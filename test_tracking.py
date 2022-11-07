import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from functions import *

data = scipy.io.loadmat('sample.mat')

ifrm = 1
D = np.array(data['vEnvFilt_zxf'][:,:,ifrm], np.float32)
DD = (((D - D.min()) / (D.max() - D.min())) * 255.9).astype(np.uint8)


d = cv2.cvtColor(DD, cv2.COLOR_GRAY2BGR)

frame_size = d.shape[0:2][::-1]

struct_disk_radius = 5
disk_size = 2 * struct_disk_radius - 1
disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size,disk_size))

blurred_frame = cv2.GaussianBlur(d, (3,3), 0)

tophat_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, disk_kernel)

hist_data = cv2.calcHist([tophat_frame], [0], None, [256], [0,256])
hist_data[0] = 0

peak_width, peak_location = fwhm(hist_data)
bw_threshold_tolerance = 0.4
bw_threshold = int(peak_location + bw_threshold_tolerance * peak_width)

blurred_tophat_frame = cv2.GaussianBlur(tophat_frame, (3,3), 0)

bw_frame = cv2.threshold(blurred_tophat_frame,bw_threshold, 255, cv2.THRESH_BINARY)[1]

bw_opened = cv2.morphologyEx(bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

bw_filled = cv2.morphologyEx(
    bw_opened,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
)

spot_locations = ultimate_erode(bw_filled[:,:,0], ifrm)

num_spots = len(spot_locations)

positions = np.zeros([num_spots,2])
for i in range(num_spots):
    positions[i,:] = spot_locations[i]


#plt.scatter(*zip(*spot_locations))

### ------------ merge coincident candidates ------------- ##

new_positions = []
skip = []
for i in range(num_spots):
    tmp_positions = [positions[i,:]]
    if i in skip:
        continue
    for j in range(i + 1, num_spots):
        if sum((positions[i,:] - positions[j,:]) **2) < 4:
            skip.append(j)
            tmp_positions.append(positions[j,:])
    p = [0, 0]
    for pos in tmp_positions:
        p[0] += pos[0]
        p[1] += pos[1]
    p[0] = p[0] / len(tmp_positions)
    p[1] = p[1] / len(tmp_positions)
    new_positions.append(p)


#plt.scatter(*zip(*spot_locations),s=1);plt.scatter(*zip(*new_positions),s=1);
#plt.gca().invert_yaxis();plt.show()

plt.imshow(DD);
plt.scatter(*zip(*new_positions),s=1,c='r')
plt.show()


##

num_spots = len(new_positions)

positions = np.zeros([num_spots,2])
for i in range(num_spots):
    positions[i,:] = new_positions[i]

SNR = np.zeros([num_spots])
## refine spot centers
r = 8
gauss_mask_max_iter = 1000
inner_mask_radius = 5
gauss_mask_sigma = 2.0#2.0
snr_filter_cutoff = 0.4

for i in range(num_spots):
    N = 2 * r + 1
    p_estimate = positions[i, :]
    for d_ in (0,1):
        if round(p_estimate[d_]) < r:
            p_estimate[d_] = r
        elif round(p_estimate[d_]) > frame_size[d_]-r-1:
            p_estimate[d_] = frame_size[d_] - r - 1
    # create sub-Image
    spot_region = np.array(
    [[round(p_estimate[0]) - r, round(p_estimate[0]) + r],
    [round(p_estimate[1]) - r, round(p_estimate[1]) + r]]).astype(int)
    spot_pixels = DD[
    spot_region[1, 0] : spot_region[1, 1] + 1,
    spot_region[0, 0] : spot_region[0, 1] + 1]
    coords = np.mgrid[
    spot_region[0, 0] : spot_region[0, 1] + 1,
    spot_region[1, 0] : spot_region[1, 1] + 1]
    Xs, Ys = np.meshgrid(
    range(spot_region[0,0], spot_region[0, 1] + 1),
    range(spot_region[1,0], spot_region[1, 1] + 1))
    converged = False
    iteration = 0
    clipping = False
    spot_intensity = 0
    bg_intensity = 0
    snr = 0
    while not converged and iteration < gauss_mask_max_iter:
        iteration += 1
        # Generate the inner mask
        inner_mask = np.where(
        (coords[0, :, :] - p_estimate[0]) ** 2
        + (coords[1, :, :] - p_estimate[1]) ** 2
        <= inner_mask_radius **2,1,0)
        mask_pixels = np.sum(inner_mask)
        # generate gaussian mask
        coords_sq = (coords[:,:,:] - p_estimate[:, np.newaxis, np.newaxis]) ** 2
        exponent = -(coords_sq[0,:,:] + coords_sq[1,:,:]) / (2 * gauss_mask_sigma **2)
        gauss_mask = np.exp(exponent)
        # normalize
        if np.sum(gauss_mask) != 0:
            gauss_mask /= np.sum(gauss_mask)
        bg_mask = 1 - inner_mask
        # calculate local background intensity and subtract from sub Image
        spot_bg = spot_pixels * bg_mask
        num_bg_spots = np.sum(bg_mask)
        bg_average = np.sum(spot_bg) / num_bg_spots
        bg_corr_spot_pixels = spot_pixels - bg_average
        # calculate revised position estimate
        spot_gaussian_product = bg_corr_spot_pixels * gauss_mask
        p_estimate_new = np.zeros(2)
        p_estimate_new[0] = np.sum(spot_gaussian_product * Xs) / np.sum(spot_gaussian_product)
        p_estimate_new[1] = np.sum(spot_gaussian_product * Ys) / np.sum(spot_gaussian_product)
        estimate_change = np.linalg.norm(p_estimate - p_estimate_new) # distance between two points
        if not np.isnan(p_estimate_new).any():
            p_estimate = p_estimate_new
        else:
            print('WARNING: Position estimate is NaN, failed to converge')
            break
        spot_intensity = np.sum(bg_corr_spot_pixels * inner_mask)
        bg_std = np.std(spot_bg[bg_mask==1])
        if estimate_change < 1e-6:
            converged = True
        snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))
        if snr <= snr_filter_cutoff:
            break
    positions[i,:] = p_estimate
    SNR[i] = snr

#plt.imshow(DD);
#plt.scatter(p_estimate[0],p_estimate[1],c='r',s=40);
#plt.show()

plt.imshow(DD);
plt.scatter(*zip(*new_positions),s=1,c='y')
plt.scatter(*zip(*positions),s=1,c='r')
plt.show()

## filter candidates
filt_pos = []
subarray_halfwidth = 8
for i in range(num_spots):
    if SNR[i] <= snr_filter_cutoff:
        continue
    if positions[i,0] < subarray_halfwidth \
        or positions[i,0] >= frame_size[0] - subarray_halfwidth \
        or positions[i,1] < subarray_halfwidth \
        or positions[i,1] >= frame_size[1] - subarray_halfwidth:
        continue
    filt_pos.append(positions[i,:])



plt.imshow(DD);
plt.scatter(*zip(*filt_pos),s=1,c='y')
plt.show()

pos1 = filt_pos
pos2 = filt_pos

plt.scatter(*zip(*pos1),s=1,c='y')
plt.scatter(*zip(*pos2),s=1,c='r')
plt.show()
