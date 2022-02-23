import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.ndimage.filters as fi

import itertools

import sklearn as sk
from sklearn import svm
import pandas as pd
import os
import numpy as np
import pickle # for saving the model if desired

from matplotlib import animation, rc
from IPython.display import HTML

import util

def bin_path(settings, ds):
    # Interpolate the position to the same timestamps as the activity
    interp_x = interpolate.interp1d(ds['pos_t_ms'], ds['pos_x_cm'], axis=0)
    interp_y = interpolate.interp1d(ds['pos_t_ms'], ds['pos_y_cm'], axis=0)
    i_pos_x_cm = interp_x(ds['activity_t_ms'])
    i_pos_y_cm = interp_y(ds['activity_t_ms'])

    # Bounds for the arena in the canonical coordinates
    x_bounds = [0, ds['arena_width_cm']]
    y_bounds = [0, ds['arena_height_cm']]

    # Bin the position data
    pos_x_binned, pos_x_bin_centers_cm, pos_x_out_of_bounds_sample_indices = util.bin_position(x_bounds,
                                                                                             settings['n_pos_x_bins'],
                                                                                             i_pos_x_cm)
    pos_y_binned, pos_y_bin_centers_cm, pos_y_out_of_bounds_sample_indices = util.bin_position(y_bounds,
                                                                                             settings['n_pos_y_bins'],
                                                                                             i_pos_y_cm)

    # Y-dimension comes first, THEN X-dimension
    arena_size_binned = (settings['n_pos_y_bins'], settings['n_pos_x_bins'])

    pos_xy_binned_linear = np.ravel_multi_index((pos_y_binned, pos_x_binned), arena_size_binned, mode='raise',
                                                order='C')

    # Store
    ds['i_pos_x_cm'] = i_pos_x_cm
    ds['i_pos_y_cm'] = i_pos_y_cm
    ds['pos_x_bin_centers_cm'] = pos_x_bin_centers_cm
    ds['pos_y_bin_centers_cm'] = pos_y_bin_centers_cm
    ds['pos_x_binned'] = pos_x_binned
    ds['pos_y_binned'] = pos_y_binned

    ds['pos_x_bounds'] = x_bounds
    ds['pos_y_bounds'] = y_bounds

    ds['pos_x_out_of_bounds_sample_indices'] = pos_x_out_of_bounds_sample_indices
    ds['pos_y_out_of_bounds_sample_indices'] = pos_y_out_of_bounds_sample_indices

    ds['arena_size_binned'] = arena_size_binned
    ds['arena_total_bins'] = arena_size_binned[0] * arena_size_binned[1]
    ds['pos_xy_binned_linear'] = pos_xy_binned_linear



