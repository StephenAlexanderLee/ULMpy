
import numpy as np
import os
import csv
from localization import Localize
from scipy.optimize import linear_sum_assignment

class Trajectory:
    def __init__(self, id, bubbles, bubble_id):
        self.id = id
        self.start_frame = bubbles.frame
        self.end_frame = bubbles.frame
        self.path = [bubbles.positions[bubble_id, :]]
        self.intensity = [bubbles.spot_intensity[bubble_id]]
        self.bg_intensity = [bubbles.bg_intensity[bubble_id]]
        self.snr = [bubbles.snr[bubble_id]]
        self.length = 1
        self.stochiometry = 0
        self.converged = [bubbles.converged[bubble_id]]
        self.linked_traj = None
        self.width = [bubbles.width[bubble_id]]
        self.frame_link = [bubbles.frame]
    def extend(self, bubbles, bubble_id):
        self.end_frame = bubbles.frame
        self.path.append(bubbles.positions[bubble_id,:])
        self.frame_link.append(bubbles.frame)
        self.intensity.append(bubbles.spot_intensity[bubble_id])
        self.bg_intensity.append(bubbles.bg_intensity[bubble_id])
        self.converged.append(bubbles.converged[bubble_id])
        self.snr.append(bubbles.snr[bubble_id])
        self.length += 1
    def fix(self):
        frame_link = np.arange(self.start_frame,self.end_frame)
        missing_frames = np.setdiff1d(frame_link,self.frame_link)
        for idx, frame in enumerate(frame_link):
            if frame in missing_frames:
                self.path.insert(idx,np.array([np.interp(idx,self.frame_link,[x[0] for x in self.path]),np.interp(idx,self.frame_link,[y[1] for y in self.path])]))
                self.bg_intensity.insert(idx,np.interp(idx,self.frame_link,self.bg_intensity))
                self.snr.insert(idx,np.interp(idx,self.frame_link,self.snr))
                self.converged.insert(idx,0)
                self.frame_link.insert(idx,frame)
        self.length = len(self.path)

def build_trajectories(all_bubbles, params):
    trajectories = []
    idt = 0
    # create trajectories for all bubbles in first frame
    for i in range(all_bubbles[0].num_spots):
        trajectories.append(Trajectory(idt, all_bubbles[0], i))
        idt += 1
    # construct trajectories for the rest of the frames
    for frame in range(1,len(all_bubbles)):
        assigned_spots = []
        for bubble in range(all_bubbles[frame].num_spots):
            close_candidates = []
            for candidate in trajectories:
                if candidate.end_frame != frame-1:
                    continue
                candidate_dist = np.linalg.norm(
                    all_bubbles[frame].positions[bubble, :] - candidate.path[-1]
                )
                if candidate_dist < params.max_displacement:
                    close_candidates.append(candidate)
            if len(close_candidates) == 0:
                trajectories.append(Trajectory(idt, all_bubbles[frame], bubble))
                idt +=1
            elif len(close_candidates) == 1:
                close_candidates[0].extend(all_bubbles[frame], bubble)
            else:
                trajectories.append(Trajectory(idt, all_bubbles[frame], bubble))
                idt += 1
    filtered_trajectories = list(filter(lambda x: x.length >= params.min_traj_len, trajectories))
    actual_traj_num = 0
    for traj in filtered_trajectories:
        traj.id = actual_traj_num
        actual_traj_num += 1
    return filtered_trajectories

def write_trajectories(trajectories, filename):
    f = open(filename, "w")
    f.write(f"trajectory\tframe\tx\ty\tspot_intensity\tbg_intensity\tSNR\tconverged\twidthx\twidthy\n")
    for traj in trajectories:
        for frame in range(traj.start_frame, traj.end_frame + 1):
            i = frame - traj.start_frame
            f.write(
            f"{traj.id}\t{frame}\t{traj.path[i][0]}\t{traj.path[i][1]}\t{traj.bg_intensity[i]}\t{traj.snr[i]}\t{traj.converged[i]}\t{traj.width[0][0]}\n")
    f.close()

def read_trajectories(filename):
    trajectories = []
    prev_traj_id = -1
    if not os.path.isfile(filename):
        print('no such file {filename}')
        return None
    with open(filename, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter = '\t')
        for line in tsv_reader:
            if line[0] == 'trajectory':
                continue
            bubble = Localize(num_spots = 1)
            traj_id = int(line[0])
            bubble.frame = int(line[1])
            bubble.positions[0,:] = [float(line[2]), float(line[3])]
            bubble.bg_intensity[0] = float(line[4])
            bubble.snr[0] = float(line[5])
            bubble.converged[0] = int(float(line[6]))
            bubble.width[0,:] = [float(line[7]), float(line[7])]
            if traj_id != prev_traj_id:
                trajectories.append(Trajectory(traj_id, bubble, 0))
            else:
                trajectories[-1].extend(bubble, 0)
            prev_traj_id = traj_id
    return trajectories

def hungarianlinker(data,frame,unmatched,unmatch_range,params):
    sources = data[frame].positions
    for key in unmatch_range:
        if all(unmatched[key] != None):
            sources = np.vstack((sources,data[key].positions[unmatched[key]]))
    targets = data[frame+1].positions
    n_target = len(targets)
    n_source = len(sources)
    D = np.zeros((n_source,n_target))
    for i, point in enumerate(sources):
        diff_coords = targets - np.tile(point,(n_target,1))
        square_dist = np.sum(diff_coords**2,1)
        D[i,:] = square_dist
    D[D>params.max_displacement**2] = params.max_displacement**10
    #D = params.max_displacement**2-D
    row, col = linear_sum_assignment(D,False)
    aidx = np.where(D[row,col]!=params.max_displacement**10)
    assigned_sources = row[aidx]
    assigned_targets = col[aidx]
    unassigned_sources = np.setdiff1d(np.arange(n_source),assigned_sources)
    unassigned_targets = np.setdiff1d(np.arange(n_target),assigned_targets)
    target_dist = [np.NaN]*len(assigned_sources)
    for i, (idx,idy) in enumerate(zip(assigned_sources,assigned_targets)):
        dist = np.sqrt(D[idx,idy])
        target_dist[i] = dist if dist < params.max_displacement else np.nan
    assigned_matches = np.argwhere(~np.isnan(target_dist)).T
    sources_idx = assigned_sources[assigned_matches][0].tolist()
    target_idx = assigned_targets[assigned_matches][0].tolist()
    unassigned_sources = np.setdiff1d(np.arange(n_source),sources_idx)
    unassigned_targets = np.setdiff1d(np.arange(n_target),target_idx)
    return list(zip(sources_idx,target_idx)), [x for x in target_dist if not np.isnan(x)], unassigned_sources, unassigned_targets

def build_trajectories_hungarian(all_bubbles,params):
    Nframes = len(all_bubbles)
    trajectories = {}
    unmatched = dict.fromkeys(set(range(Nframes)))
    unmatched_t = dict.fromkeys(set(range(Nframes)))
    idt = 0
    # create trajectories for all bubbles in first frame
    for i in range(all_bubbles[0].num_spots):
        trajectories[i] = (Trajectory(idt, all_bubbles[0], i))
        idt+=1
    for frame in range(Nframes-1):
        #frame = 0
        # frame gap search range
        unmatch_range = [x for x in reversed(range(frame-params.max_frame_gap,frame)) if x >=0]
        assignments, distances, unassigned_sources, unassigned_targets = hungarianlinker(all_bubbles,frame,unmatched,unmatch_range,params)
        # extend trajectory from source to bubble target
        # then rearrange trajectories to next frame bubble order
        shuffle_traj = {}
        for idx, (source,target) in enumerate(assignments):
            trajectories[source].extend(all_bubbles[frame+1],target)
            shuffle_traj[target] = trajectories[source]
        # for all unassigned targets, create new trajectory in next frame bubble order
        for idx, utarget in enumerate(unassigned_targets):
            shuffle_traj[utarget] = Trajectory(idt, all_bubbles[frame+1],utarget)
            idt+=1
        #print('{0} trajectories should = {1} next number of bubbles'.format(len(shuffle_traj),all_bubbles[frame+1].num_spots))
        # set current unmatched frame to all unassigned sources from current frame bubble order
        idx = all_bubbles[frame].num_spots
        unmatched[frame] = np.array(unassigned_sources[unassigned_sources<idx])
        unmatched_t[frame] = np.array([trajectories[x].id for x in unmatched[frame]])
        # remove newly assigned bubbles from previous frames in gap search range
        for key in unmatch_range:
            limits = [idx, idx + len(unmatched[key])]
            idx += len(unmatched[key])
            previous_unassigned_sources = unassigned_sources[(limits[0] <= unassigned_sources)*(unassigned_sources < limits[1])]
            previous_unassigned_traj = np.array([trajectories[x].id for x in previous_unassigned_sources])
            matched_idx = np.in1d(unmatched_t[key],previous_unassigned_traj)
            #print('for frame {0}: matched {1} previously unmatched bubbles'.format(key,sum(matched_idx==False)))
            unmatched[key] = unmatched[key][matched_idx]
            unmatched_t[key] = unmatched_t[key][matched_idx]
        # move all unassigned trajectories to end of order
        for idx, leftover in enumerate(unassigned_sources):
            shuffle_traj[all_bubbles[frame+1].num_spots + idx] = trajectories[leftover]
        # reassign trajectories with shuffled trajectories
        trajectories = shuffle_traj
        #print('{0} trajectories = {1} next bubbles + {2} unmatched'.format(len(trajectories),all_bubbles[frame+1].num_spots,len(unassigned_sources)))
    filtered_trajectories = list(filter(lambda x: x.length >= params.min_traj_len, trajectories.values()))
    actual_traj_num = 0
    for traj in filtered_trajectories:
        traj.id = actual_traj_num
        actual_traj_num += 1
    # fill gaps
    [t.fix() for t in filtered_trajectories]
    return filtered_trajectories


#frame_link = np.arange(traj.start_frame,traj.end_frame)
#missing_frames = np.setdiff1d(frame_link,traj.frame_link)
#for idx, frame in enumerate(frame_link):
#    if frame in missing_frames:
#        traj.path.insert(idx,np.array([np.interp(idx,traj.frame_link,[x[0] for x in traj.path]),np.interp(idx,traj.frame_link,[y[1] for y in traj.path])]))
#        traj.bg_intensity.insert(idx,np.interp(idx,traj.frame_link,traj.bg_intensity))
#        traj.snr.insert(idx,np.interp(idx,traj.frame_link,traj.snr))
#        traj.converged.insert(idx,0)
#        traj.frame_link.insert(idx,frame)
#
#traj.length = len(traj.path)
