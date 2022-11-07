import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from natsort import natsorted
import multiprocessing as mp
import os

# ULM modules
import parameters
import localization
import trajectory
import tracking
import visualization
from functions import *

def ULMlocalize(ULMfile,img,params,tree,reprocess = False):
    filename = tree.csd_l + '\\localizations_' + ULMfile[:-4] + '.pickle'
    #data = np.array(scipy.io.loadmat('sample.mat')['vEnvFilt_zxf'], np.float32)
    # localize bubbles
    if os.path.exists(filename) and not reprocess:
        with open(filename, "rb") as f:
            all_bubbles = pickle.load(f)
    else:
        Nframes = img.shape[2]
        all_bubbles = []
        if params.parallel_proc is True:
            mp.Pool(mp.cpu_count()).terminate()
            with mp.Pool(mp.cpu_count()) as pool:
                res = [pool.apply_async(tracking.localize_frame, (img, frame, params)) for frame in range(Nframes)]
                all_bubbles = [r.get() for r in res]
            pool.close()
            pool.join()
        else:
            all_bubbles = []
            for frame in range(Nframes):
                all_bubbles.append(tracking.localize_frame(img, frame, params))
        with open(filename, "wb") as f:
            pickle.dump(all_bubbles, f, protocol=pickle.HIGHEST_PROTOCOL)
    # localization animation
    if params.show_plots:
        fig, ax  = plt.subplots()
        bg = ax.imshow(img[:,:,0])
        sc = ax.scatter(*zip(*all_bubbles[0].positions),s=1,c='r')
        def animate(i):
            bg.set_data(img[:,:,i])
            sc.set_offsets(all_bubbles[i].positions)
        a = matplotlib.animation.FuncAnimation(fig, animate, interval = 100, frames = img.shape[2]-1)
        plt.show()
    print('\n')
    return all_bubbles

def ULMtrack(ULMfile,params,all_bubbles,tree,reprocess=False):
    filename = tree.csd_t + '\\trajectory_' + ULMfile[:-4] + '.tsv'
    # calculate trajectories
    if os.path.exists(filename) and not reprocess:
        trajs = trajectory.read_trajectories(filename)
    else:
        trajs = trajectory.build_trajectories_hungarian(all_bubbles,params)
        trajectory.write_trajectories(trajs, filename)
    return trajs

def ULMvelocity(ULMfile,trajs,params,tree,img,interp=1,reprocess=False):
    filename = tree.csd_v + '\\velocity_' + ULMfile[:-4] + '.pickle'
    # calculate trajectories
    if os.path.exists(filename) and not reprocess:
        with open(filename, "rb") as f:
            Velocity, Vmap = pickle.load(f)
    else:
        ## TODO: put in image class later
        frame_size = img.shape[:2]
        Velocity, Vmap = visualization.generate_Velocity(trajs, params, interp, frame_size = img.shape[:2])
        with open(filename, "wb") as f:
            pickle.dump([Velocity, Vmap], f, protocol=pickle.HIGHEST_PROTOCOL)
    return Velocity, Vmap

class fileTree:
    def __init__(self,load_dir,cld,cwd):
        self.csd = os.path.join(cwd,'data',load_dir)
        self.csd_l = os.path.join(self.csd, 'localizations')
        self.csd_v = os.path.join(self.csd, 'velocity')
        self.csd_t = os.path.join(self.csd, 'trajectory')
        if not os.path.exists(self.csd):
            os.mkdir(self.csd)
        if not os.path.exists(self.csd_l):
            os.mkdir(self.csd_l)
        if not os.path.exists(self.csd_v):
            os.mkdir(self.csd_v)
        if not os.path.exists(self.csd_t):
            os.mkdir(self.csd_t)
        self.load_files = natsorted(glob.glob(cld+'\*.bin'))


class ImageData:
    def __init__(self):
        self.num_frames = -1

    def __getitem__(self, index):
        frame = ImageData()
        frame.initialize(1, self.frame_size)

    def __setitem__(self, index, value):
        if value.__class__ =='ImageData':
            self.data[:,:,index] = value.data[:,:,0]
        else:
            self.data[:,:,index] = value

    def initialize(self, num_frames, frame_size):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_pixels = frame_size[0] * frame_size[1]
        self.data = np.zeros([frame_size[1],frame_size[0],num_frames])
