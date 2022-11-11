



default_parameters = {
    # Image parameters
    'psf_width' :
        {'description' : 'point spread function of bubble',
        'class' : 'image',
        'default' : 5},
    # Tracking Parameters
    'disk_radius' :
        {'description' : 'Radius of the cv2 disc structural element',
         'class' : 'tracking',
         'default' : 5},
    'N_merge' :
        {'description' : 'merge coincident bubble centroids',
         'class' : 'tracking',
         'default' : 25},
    'bw_threshold_tolerance' :
        {'description' : 'threshold for generating the b/w image relative to peak intensity',
        'class' : 'tracking',
        'default' : 0.4},
    'max_displacement' :
        {'description' : 'maximum displacement allowed for spots between frames',
         'class': 'tracking',
         'default' :5},
    'min_traj_len' :
        {'description' : 'Minimum number of frames needed to define a trajectory',
         'class' : 'tracking',
         'default' : 10},
    'max_frame_gap' :
        {'description' : 'max frame search for gaps in links',
         'class' : 'tracking',
         'default' : 3},
    'subarray_halfwidth':
        {'description' : 'Halfwidth of the sub-image for analysing individual spots',
         'class' : 'tracking',
         'default' : 8},
     'gauss_mask_max_iter' :
         {'description' : 'maximum number of iterations for center refinement',
          'class' : 'tracking',
          'default' : 1000},
     'inner_mask_radius':
         {'description' : 'radius of the mask used for calculating spot intensities',
          'class' : 'tracking',
          'default' : 5},
     'gauss_mask_sigma' :
         {'description' : 'width of the gaussian used for center refinement',
         'class' : 'tracking',
         'default' : 3.0},
     'snr_filter_cutoff' :
         {'description' : 'cutoff value when filtering spots based on SNR',
         'class' : 'tracking',
         'default' : 1.0},
    # velocity parameters
    'frame_rate' :
        {'description' : 'ultrasound frame rate for calculating velocities',
        'class' : 'velocity',
        'default' : 1e-3},
    'wn_size' :
        {'description' : 'smoothing filter point size for track interp',
        'class' : 'velocity',
        'default' : 15},
    'wl2mm' :
        {'description' : 'conversion from wavelengths to mm with interp factor',
        'class' : 'velocity',
        'default' : 9.8560e-5/4*1e3},
    'spatial_average_radius' :
        {'description' : 'defined radius for spatial averaging in mm',
        'class' : 'velocity',
        'default' : 40e-6/9.8560e-5*4*4*4}, # 4 ppw : 4 interp bin : 4 interp now
    'parallel_proc' :
        {'description' : 'enable parallel pooling',
        'class' : 'general',
        'default' : True},
    # post processing parameters
    'show_plots':
        {'desription': 'display figures or not',
        'class' : 'postprocessing',
        'default': False}
}


class Parameters:
    def __init__(self, initial=default_parameters):
        self._params = initial
        for p in self._params.keys():
            self._params[p]['value'] = self._params[p]['default']
    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self,name)
        else:
            try:
                return object.__getattribute__(self, "_params")[name]['value']
            except KeyError as exc:
                print("error: no such key")
                raise exc
    def __setattribute__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._params[name]['value'] = value
