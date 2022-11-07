import numpy as np
import matplotlib.pyplot as plt
import cv2

from functions import *


class Localize:
    def __init__(self, num_spots = 0, frame = 0):
        self.num_spots =  num_spots
        if num_spots > 0:
            self.positions = np.zeros([num_spots, 2])
            self.bg_intensity = np.zeros(num_spots)
            self.spot_intensity = np.zeros(num_spots)
            self.frame = frame
            self.traj_num = [-1] * self.num_spots
            self.snr = np.zeros([num_spots])
            self.converged = np.zeros([num_spots], dtype=np.int8)
            self.exists = True
            self.width = np.zeros((num_spots,2))
        else:
            self.frame = frame
            self.exists = False

    def set_positions(self, positions):
        self.num_spots = len(positions)
        self.positions = np.zeros([self.num_spots, 2])
        self.clipping = [False] * self.num_spots
        self.bg_intensity = np.zeros(self.num_spots)
        self.spot_intensity = np.zeros(self.num_spots)
        self.center_intensity = np.zeros(self.num_spots)
        self.width = np.zeros([self.num_spots, 2])
        self.traj_num = [-1] * self.num_spots
        self.snr = np.zeros([self.num_spots])
        self.converged = np.zeros([self.num_spots], dtype=np.int8)
        for i in range(self.num_spots):
            self.positions[i, :] = positions[i]

    def localize_bubbles(self, data, params):
        d = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        disk_size = 2 * params.disk_radius - 1
        disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))
        blurred_frame = cv2.GaussianBlur(d, (3,3), 0)
        tophat_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, disk_kernel)
        hist_data = cv2.calcHist([tophat_frame], [0], None, [256], [0,256])
        hist_data[0] = 0
        peak_width, peak_location = fwhm(hist_data)
        bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)
        blurred_tophat_frame = cv2.GaussianBlur(tophat_frame, (3,3), 0)
        bw_frame = cv2.threshold(blurred_tophat_frame, bw_threshold, 255, cv2.THRESH_BINARY)[1]
        bw_opened = cv2.morphologyEx(bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
        bw_filled = cv2.morphologyEx(bw_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        spot_locations = ultimate_erode(bw_filled[:,:,0], data)
        self.set_positions(spot_locations)

    def merge_coincident_bubbles(self, spacing_thresh = 4):
        new_positions = []
        skip = []
        for i in range(self.num_spots):
            tmp = [self.positions[i, :]]
            if i in skip:
                continue
            for j in range(i + 1, self.num_spots):
                if sum((self.positions[i, :] - self.positions[j, :]) ** 2) < spacing_thresh:
                    skip.append(j)
                    tmp.append(self.positions[j,:])
            p = [0, 0]
            for pos in tmp:
                p[0] += pos[0]
                p[1] += pos[1]
            p[0] = p[0] / len(tmp)
            p[1] = p[1] / len(tmp)
            new_positions.append(p)
        self.set_positions(new_positions)

    def refine_centers(self, data, params):
        frame_size = data.shape[0:2][::-1]
        for idx in range(self.num_spots):
            r = params.subarray_halfwidth
            N = 2 * r + 1
            # get center estimate + fit within frame
            p_estimate = self.positions[idx, :]
            for d in (0, 1):
                if round(p_estimate[d]) < r:
                    p_estimate[d] = r
                elif round(p_estimate[d]) > frame_size[d] - r - 1:
                    p_estimate[d] = frame_size[d] - r - 1
            # create sub_image
            sub_image = np.array(
            [[round(p_estimate[0]) - r, round(p_estimate[0]) + r],
             [round(p_estimate[1]) - r, round(p_estimate[1]) + r]]).astype(int)
            sub_pixels = data[
            sub_image[1,0] : sub_image[1,1] + 1,
            sub_image[0,0] : sub_image[0,1] + 1]
            coords = np.mgrid[
            sub_image[0,0] : sub_image[0,1] + 1,
            sub_image[1,0] : sub_image[1,1] + 1]
            Xs, Ys = np.meshgrid(
            range(sub_image[0,0], sub_image[0,1] + 1),
            range(sub_image[1,0], sub_image[1,1] + 1))
            # iteratively refine centers
            converged = False
            iteration = 0
            clipping = False
            spot_intensity = 0
            bg_intensity = 0
            snr = 0
            while not converged and iteration < params.gauss_mask_max_iter:
                iteration += 1
                # generate the inner mask circle
                inner_mask = np.where(
                (coords[0,:,:] - p_estimate[0]) ** 2
                + (coords[1,:,:] - p_estimate[1]) ** 2
                <= params.inner_mask_radius **2, 1, 0)
                # generate gaussian mask
                coords_sq = (coords[:,:,:] - p_estimate[:, np.newaxis, np.newaxis]) ** 2
                exponent = -(coords_sq[0,:,:] + coords_sq[1,:,:]) / (2*params.gauss_mask_sigma **2)
                gauss_mask = np.exp(exponent)
                # normalize mask
                if np.sum(gauss_mask) != 0:
                    gauss_mask /= np.sum(gauss_mask)
                bg_mask = 1 - inner_mask
                # calculate local background intensity and subtract from sub image
                sub_bg = sub_pixels * bg_mask
                num_bg_pixels = np.sum(bg_mask)
                bg_average = np.sum(sub_bg) / num_bg_pixels
                bg_corr_sub_pixels = sub_pixels - bg_average
                # calculate revised position estimate
                sub_gaussian_product = bg_corr_sub_pixels * gauss_mask
                p_estimate_new = np.zeros(2)
                p_estimate_new[0] = np.sum(sub_gaussian_product * Xs) / np.sum(sub_gaussian_product)
                p_estimate_new[1] = np.sum(sub_gaussian_product * Ys) / np.sum(sub_gaussian_product)
                # calculate distance between old and new estimate
                estimate_change = np.linalg.norm(p_estimate - p_estimate_new)
                if not np.isnan(p_estimate_new).any():
                    p_estimate = p_estimate_new
                else:
                    print('WARNING: Position estimate has NaN, failed to converge')
                    break
                spot_intensity = np.sum(bg_corr_sub_pixels * inner_mask)
                bg_std = np.std(sub_bg[bg_mask == 1])
                if estimate_change < 1e-6:
                    converged = True
                # calculate spot SNR | dont bother iterating if SNR is too low
                snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))
                if snr <= params.snr_filter_cutoff:
                    break

            self.bg_intensity[idx] = bg_average
            self.spot_intensity[idx] = spot_intensity
            self.snr[idx] = snr
            self.converged[idx] = converged
            self.positions[idx, :] = p_estimate

    def filter_candidates(self, data, params):
        frame_size = data.shape[0:2][::-1]
        positions = []
        clipping = []
        bg_intensity = []
        spot_intensity = []
        center_intensity = []
        width = []
        traj_num = []
        snr = []

        for i in range(self.num_spots):
            # Filter spots that are too noisy
            if self.snr[i] <= params.snr_filter_cutoff:
                continue

            # Filter spots too close to edge
            if self.positions[i,0] < params.subarray_halfwidth \
                or self.positions[i,0] >= frame_size[0] - params.subarray_halfwidth \
                or self.positions[i,1] < params.subarray_halfwidth \
                or self.positions[i,1] >= frame_size[1] - params.subarray_halfwidth:
                continue

            positions.append(self.positions[i,:])
            clipping.append(self.clipping[i])
            bg_intensity.append(self.bg_intensity[i])
            spot_intensity.append(self.spot_intensity[i])
            center_intensity.append(self.center_intensity[i])
            width.append(self.width[i,:])
            traj_num.append(self.traj_num[i])
            snr.append(self.snr[i])

        self.num_spots = len(clipping)
        self.positions = np.array(positions)
        self.clipping = np.array(clipping)
        self.bg_intensity = np.array(bg_intensity)
        self.spot_intensity = np.array(spot_intensity)
        self.center_intensity = np.array(center_intensity)
        self.width = np.array(width)
        self.traj_num = np.array(traj_num)
        self.snr = np.array(snr)

    def get_bubble_widths(self, data, params):
        for i in range(self.num_spots):
            x = round(self.positions[i,0]).astype(int)
            y = round(self.positions[i,1]).astype(int)
            # create temporary array with center of spot
            tmp = data[
            y - params.subarray_halfwidth : y + params.subarray_halfwidth + 1,
            x - params.subarray_halfwidth : x + params.subarray_halfwidth + 1]

            spotmask = np.zeros(tmp.shape)
            cv2.circle(spotmask,
            (params.subarray_halfwidth, params.subarray_halfwidth),
            params.inner_mask_radius, 1, -1)
            bg_intensity = np.mean(tmp[spotmask == 0])
            tmp = tmp - bg_intensity
            p, succ = fit2Dgaussian(tmp)
            if succ == 1:
                self.width[i,0] = p[3]
                self.width[i,1] = p[4]
            else: #something is wrong
                self.width[i,0] = params.psf_width
                self.width[i,1] = params.psf_width
