import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d
from scipy import signal
from scipy import sparse
import math
import multiprocessing as mp
import cv2
import sys

import trajectory

class velocity:
    def __init__(self, id, N):
        self.id = id;
        self.N = N;
        self.X = []*self.N;
        self.Y = []*self.N;
        self.V = []*self.N;
        self.direction = 0;
        self.w = []*self.N;
        self.Vw = [None]*self.N;
        self.variation = 0;

def interp_and_smooth(data, interp = 1, wn_size = 5):
    n = len(data)
    N = n*interp
    datap = np.arange(0,n)
    datai = np.linspace(0,n,N)
    data_ = np.interp(datai, datap, data)
    return uniform_filter1d(data_, wn_size)

def interp_track(data, params, frame_size, interp = 1):
    Velocity = velocity(data.id,(data.length-1)*interp)
    for frame in range(data.start_frame, data.end_frame):
        data_trajectory = data.path[frame - data.start_frame]
        Velocity.X.append(data_trajectory[0])
        Velocity.Y.append(data_trajectory[1])
    # interpolate track
    Velocity.X = interp_and_smooth(Velocity.X, interp, params.wn_size)
    Velocity.Y = interp_and_smooth(Velocity.Y, interp, params.wn_size)
    dX = np.diff(Velocity.X)
    dY = np.diff(Velocity.Y)
    # calculate velocity of each interpolated pixel
    distance = [math.sqrt(distx**2 + disty**2) for distx,disty in zip(dX,dY)]
    distance.insert(0,0)
    Velocity.direction = np.sign(np.sum([math.atan2(distx,disty) for distx,disty in zip(dX,dY)]))
    Velocity.variation = np.std(np.array(data.path),0)
    for d in distance:
        Velocity.V.append(d/params.frame_rate*params.wl2mm)
    return Velocity

def local_weighted_average(data,index,params):
    print('spatially averaging map {0}/{1}\r'.format(index+1,len(data)), end = "\r", flush = True)
    v1 = data[index]
    for idx in range(v1.N):
        centroid = [v1.X[idx],v1.Y[idx]]
        v_sum = Z = 0
        for v2 in data:
            d = [math.sqrt((x_-centroid[0])**2 + (y_-centroid[1])**2) for x_,y_ in zip(v2.X,v2.Y)]
            d_ = [i if i<params.spatial_average_radius else np.nan for i in d]
            v2.w = [np.exp(-x/params.spatial_average_radius)**2 for x in d_]
            if ~np.isnan(v2.w).all():
                Z += np.nansum(v2.w)
                v_sum += np.nansum([u*v for u,v in zip(v2.w,v2.V)])
                v1.Vw[idx] = (v_sum/Z) if Z!= 0 else 0
    return v1

def track2grid(data, index, frame_size,interp,params):
    print('interpolating onto map {0}/{1}'.format(index+1,len(data)), end = "\r", flush = True)
    # interpolate onto 2D grid
    dd = np.zeros((frame_size[0]*interp,frame_size[1]*interp))
    x = np.linspace(0,data[index].N*interp, data[index].N)
    xi = np.linspace(0,data[index].N*interp,data[index].N*interp)
    xcoords = interp1d(x,data[index].X*interp)
    xcoords = np.round(xcoords(xi)).astype(int)
    ycoords = interp1d(x,data[index].Y*interp)
    ycoords = np.round(ycoords(xi)).astype(int)
    #zvalues = interp1d(x,[x*data[index].direction for x in data[index].Vw])
    zvalues = interp1d(x,[x for x in data[index].Vw])
    #sigma = np.std(zvalues(xi))
    #N = round(params.psf_width*interp) if round(params.psf_width*interp)%2 is False else round(params.psf_width*interp)+1
    #N = int(data[index].variation[0]*interp) if int(data[index].variation[0]*interp)%2 == 1 else int(data[index].variation[0]*interp)+1
    #M = int(data[index].variation[1]*interp) if int(data[index].variation[1]*interp)%2 == 1 else int(data[index].variation[1]*interp)+1
    #std = np.round(np.std(data[index].variation))
    #N = int(std*interp) if (std*interp)%2 is False else int(std*interp + 1)
    #k1d = signal.gaussian(N,std=data[index].variation[1]).reshape(N,1)
    #k2d = signal.gaussian(N,std=data[index].variation[0]).reshape(N,1)
    #kernel = np.outer(k1d,k2d)
    for X,Y,Z in zip(xcoords,ycoords,zvalues(xi)):
        dd[Y,X]=Z
        #dd[int(Y-(N/2)):int(Y+(N/2)),int(X-(N/2)):int(X+(N/2))] += kernel*Z
    #dd = dd/np.max(dd)*np.max(data[index].Vw)*data[index].direction
    return sparse.csr_matrix(dd)

def generate_Velocity(data, params, interp = 1, frame_size=None):
    velocities = []
    Vmap = []
    for idx, T in enumerate(data):
        print('calculating Velocity for trajectory {0}/{1}'.format(idx+1,len(data)), end = "\r", flush = True)
        velocities.append(interp_track(T,params,frame_size,interp));
    print('')
    if params.parallel_proc is True:
        mp.Pool(mp.cpu_count()).terminate()
        with mp.Pool(mp.cpu_count()) as pool:
            res = [pool.apply_async(local_weighted_average, (velocities, index, params)) for index in range(len(velocities))]
            velocities = [r.get() for r in res]
        pool.close()
        pool.join()
        print('')
        with mp.Pool(mp.cpu_count()) as pool:
            res = [pool.apply_async(track2grid, (velocities, index,frame_size, interp, params)) for index in range(len(velocities))]
            Vmap = [r.get() for r in res]
        pool.close()
        pool.join()
        print('')
    else:
        for index, V in enumerate(velocities):
            velocities[index] = local_weighted_average(velocities, index, params);
            Vmap.append(track2grid(velocities, index, frame_size, interp, params));
            print('')
    return velocities, Vmap
