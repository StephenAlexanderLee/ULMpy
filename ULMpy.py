## pyULM: Ultrasound Localization Microscopy using python
## MB tracking is based on pystachio (https://github.com/ejh516/pystachio-smt)
## [https://doi.org/10.1016/j.csbj.2021.07.004]
## trajectory linking: hungarian linker

## Written by: Stephen A. Lee 2022

# ---------------------------- modules --------------------------------------- #
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import os
import pickle
import copy

# --------------------------- custom modules --------------------------------- #
import parameters
from functions import MBseparate_freq
from ULMwrappers import *
P = parameters.Parameters()

# --------------------------- main script ------------------------------------ #
load_dir = '13 M02 pre-TUS ulm'
load_date = '22-11-10'
cld = 'H:\\Research Data\\ULM\\To Stephen From Sua 2\\{0}\\ULM'.format(load_dir)
cwd = os.getcwd()                   # set current directory
tree = fileTree(load_dir,cld,cwd,load_date)   # generate file tree structure
# load enveloped data
Nframes = 500                       # TODO: put into parameters [number of frames]
#N = 281
#M = 521
N = 321                             # TODO: put into parameters [image size]
M = 161                             # TODO: put into parameters [image size]
interp_factor = 4                   # interp*4 [already interpolated by 4] times current image grid
reprocess = [False,False,True]        # [localization, trajectory, velocity]
P.show_plots = False                # verbose plotting
seperate_f = [[0,5],[2.5,10],[7.5,25]] # MB seperation frequency bins

# perform ULM - localization, tracking, velocity estimation
for f in tree.load_files:
    #f = tree.load_files[0]
    # Bubble Seperation
    for W in seperate_f:
        #W = seperate_f[0]
        ULMfile = '{0}to{1}_'.format(W[0],W[1]) + os.path.basename(f)
        print(ULMfile)
        data = MBseparate_freq(np.reshape(np.fromfile(f, dtype='<f'),(N,M,Nframes),order='F'),w=W)
        img = (((data - data.min()) / (data.max() - data.min())) * 255.9).astype(np.uint16)
        all_bubbles = ULMlocalize(ULMfile,img,P,tree,reprocess[0])
        trajs = ULMtrack(ULMfile,P,all_bubbles,tree,reprocess[1])
        print('detected {0} tracks'.format(len(trajs)))
        if len(trajs)>0:
            vv,sV = ULMvelocity(ULMfile,trajs,P,tree,img,interp_factor,reprocess[2])
# ---------------------------------------------------------------------------- #

def Vsparse_to_array(Vel,M):
    if type(Vel) is scipy.sparse.csr.csr_matrix:
        Vel = Vel.toarray()
        mask = M.toarray()
        return (np.divide(Vel,mask,where=mask!=0)), mask
    else:
        null = np.zeros((100,100))
        return null, null


dynrange = 5;#np.mean(velo[velo>0])
cmap = copy.copy(plt.cm.get_cmap('RdBu_r'))
ncmap = cmap(np.arange(cmap.N))
ncmap[:,-1] = np.abs(np.linspace(-1,1,int(cmap.N)))
ncmap = matplotlib.colors.ListedColormap(ncmap)
cmap_args_l = dict(cmap=ncmap,
                    vmin=-dynrange,
                    vmax=dynrange,
                    extent=[-.85,-2.85,2.5,6.5],
                    alpha = 0.8,
                    aspect='equal')
cmap_args_r = dict(cmap=ncmap,
                    vmin=-dynrange,
                    vmax=dynrange,
                    extent=[.85,2.85,2.5,6.5],
                    alpha = 0.8,
                    aspect='equal')
plt.style.use('seaborn-white')

load_velocity = natsorted(glob.glob(tree.csd_v+'\*.pickle'))[:20]
load_vec = [('lft' in x) for x in load_velocity]
V_l = V_r = 0
M_l = M_r = 0
for i,filename in enumerate(load_velocity):
    with open(filename, "rb") as f:
        vv , sV = pickle.load(f)
        if load_vec[i]:
            V_l += np.sum(sV,0)
            #mask += len(sV)
            M_l += np.sum([((x)!=0).astype(int) for x in sV],0)
        else:
            V_r += np.sum(sV,0)
            M_r += np.sum([((x)!=0).astype(int) for x in sV],0)

velo_l, mask_l = Vsparse_to_array(V_l,M_l)
velo_r, mask_r = Vsparse_to_array(V_r,M_r)

fig, ax = plt.subplots(1,2,figsize=(17.5,8))
ax[0].imshow(mask_l,cmap='gray_r',extent=[-.85,-2.85,2.5,6.5])
ax[0].imshow(velo_l,**cmap_args_l)
ax[0].set_xlabel('mm');ax[0].set_ylabel('mm');ax[0].set_title(load_dir + ' left')
ax[1].imshow(mask_r,cmap='gray_r',extent=[.85,2.85,2.5,6.5])
im = ax[1].imshow(velo_r,**cmap_args_r)
ax[1].set_xlabel('mm');ax[1].set_ylabel('mm');ax[1].set_title(load_dir + ' right')
h = fig.colorbar(im)
h.ax.set_ylabel('mm/s')
#plt.savefig(os.path.join(tree.csd,'velocity_map_abs.png'))
#np.save(os.path.join(tree.csd,'final_velocity'),[velo_l,M_l,velo_r,M_r])
#plt.close()
plt.show()

# ---------------------------------------------------------------------------- #

#dx = 2.5e-5/interp_factor
#ROI = [int(x/dx*1e-3) for x in [1.5, 4.0]]
#Rstart = [int(x/dx*1e-3) for x in [1.25, 2.5]]
#segment_l = velo_l[int(Rstart[1]):int(Rstart[1]+ROI[1]),int(velo_l.shape[1]/2-ROI[0]/2):int(velo_l.shape[1]/2+ROI[0]/2)]
#segment_r = velo_r[int(Rstart[1]):int(Rstart[1]+ROI[1]),int(velo_l.shape[1]/2-ROI[0]/2):int(velo_l.shape[1]/2+ROI[0]/2)]
#
#cmap_args_l2 = dict(cmap=ncmap,
#                    vmin=-0,
#                    vmax=dynrange,
#                    extent=[Rstart[0],Rstart[0]+ROI[0],Rstart[1],Rstart[1]+ROI[1]],
#                    alpha = 0.8,
#                    aspect='equal')
#cmap_args_r2 = dict(cmap=ncmap,
#                    vmin=-0,
#                    vmax=dynrange,
#                    extent=[Rstart[0],Rstart[0]+ROI[0],Rstart[1],Rstart[1]+ROI[1]],
#                    alpha = 0.8,
#                    aspect='equal')
#
#fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,4))
#rect = matplotlib.patches.Rectangle(Rstart,ROI[0],ROI[1],fill = False)
#rect2 = matplotlib.patches.Rectangle(Rstart,ROI[0],ROI[1],fill = False)
#ax1.imshow(M_l,vmax=5)
#ax1.add_patch(rect)
#ax2.imshow(segment_l,**cmap_args_l2)
#ax3.imshow(M_r,vmax=5)
#ax3.add_patch(rect2)
#ax4.imshow(segment_r,**cmap_args_r2)
#plt.savefig(os.path.join(tree.csd,'ROIvelocity_abs.png'))
#plt.show()
#
#left = segment_l[segment_l!=0]
#right = segment_r[segment_r!=0]
#
#sns.histplot(data = [left,right], kde=True, alpha = 0.5)
#plt.legend(labels=['right','left'])
#plt.xlabel('Velocity [mm/s]')
#plt.xlim((-20,20))
#plt.ylim((0,3000))
#plt.savefig(os.path.join(tree.csd,'velocity_distribution.png'))
#plt.show()
#
#np.save(os.path.join(tree.csd,'left_abs'),left)
#np.save(os.path.join(tree.csd,'right_abs'),right)
#
#L = []
#R = []
#for load_dir in dirs:
#    cld = 'H:\\Research Data\\ULM\\To Stephen From Sua 2\\{0}\\ULM'.format(load_dir)
#    cwd = os.getcwd()
#    tree = fileTree(load_dir,cld,cwd)
#    left = np.load(os.path.join(tree.csd,'left_abs.npy'))
#    right = np.load(os.path.join(tree.csd,'right_abs.npy'))
#    L.append(left)
#    R.append(right)
#
#plt.style.use('default')
#
#idx = 5
#fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
#sp1 = sns.histplot(data = L[idx:idx+2], kde=True, alpha = 0.5, bins=50, palette='magma', ax=ax1)
#ax1.set_xlabel('Velocity [mm/s]')
#ax1.set_title('left')
#ax1.set_xlim((0,20))
#ax1.set_ylim((0,10000))
#legend = ax1.get_legend()
#handles = legend.legendHandles
#legend.remove()
#ax1.legend(handles,dirs[idx:idx+2])
#sns.histplot(data = R[idx:idx+2], kde=True, alpha = 0.5, bins = 50, palette='magma', ax=ax2)
#ax2.set_xlabel('Velocity [mm/s]')
#ax2.set_title('right')
#ax2.set_xlim((0,20))
#ax2.set_ylim((0,10000))
#ax2.legend(handles,dirs[idx:idx+2])
#plt.savefig(os.path.join(tree.csd,'velocity_distribution_abs_1_'+ str(idx) +'.png'))
#plt.show()
#
#############################################
#for t in trajs:
#    plt.plot(*zip(*t.path))
#
#plt.gca().invert_yaxis()
#plt.show()
#
#vv = [v.toarray() for v in Vmap]
#V = np.cumsum(vv,0)
#cmap = copy.copy(plt.cm.get_cmap('RdBu'))
#
#fig, ax  = plt.subplots()
#args = {'cmap': cmap, 'vmin': -5, 'vmax': 5}
#bg = ax.imshow(V[0],**args)
#t = ax.text(0,0,'track {0}'.format(0))
#def animate(i):
#    bg.set_data(V[i])
#    t.set_text('track {0}'.format(i))
#
#a = matplotlib.animation.FuncAnimation(fig, animate, interval = 10, frames = len(V)-1)
#plt.show()
