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


def load_cell_use_data(data_folder, num_cells):
    filename = os.path.join(data_folder, 'cell_use.mat')

    if os.path.exists(filename):
        data = h5py.File(filename, 'r')

        cell_use = np.array(data['cell_use']).flatten()
        data.close()

        cell_use = (cell_use == 1)
    else:
        cell_use = np.full(num_cells, True)

    return cell_use


def load_scope_timestamps(dataFolder):
    scopeFilename = os.path.join(dataFolder, 'scope.hdf5')
    scope = h5py.File(scopeFilename, 'r')

    activity_t_ms = np.array(scope['timestamp_ms'])
    num_time_samples = len(activity_t_ms)

    scope.close()

    return activity_t_ms, num_time_samples


def load_movement_data(dataFolder):
    movementFilename = os.path.join(dataFolder, 'movement.mat')
    movement = h5py.File(movementFilename, 'r')

    pos_x_cm, pos_y_cm, pos_t_ms = np.array(movement['movement']['x_cm']), np.array(
        movement['movement']['y_cm']), np.array(movement['movement']['timestamps_ms'])
    speed_smoothed_cm_per_s = np.array(movement['movement']['speed_smoothed_cm_per_s'])

    pos_x_cm = pos_x_cm.flatten()
    pos_y_cm = pos_y_cm.flatten()
    pos_t_ms = pos_t_ms.flatten()
    speed_smoothed_cm_per_s = speed_smoothed_cm_per_s.flatten()

    movement.close()

    return pos_x_cm, pos_y_cm, pos_t_ms, speed_smoothed_cm_per_s


def load_neuron_data(dataFolder, ACTIVITY_TYPE_TO_USE='trace_filt'):
    neuronFilename = os.path.join(dataFolder, 'neuron.hdf5')
    neuron = h5py.File(neuronFilename, 'r')

    num_neurons = int(neuron.attrs['num_neurons'][0])

    if 'num_time_samples' in neuron.attrs.keys():
        num_time_samples = int(neuron.attrs['num_time_samples'][0])
    else:
        num_time_samples = -1
    
    activity = []
    for nid in range(1, num_neurons + 1):
        neuron_name = 'neuron_%d' % (nid)
        ndata = np.array(neuron[neuron_name][ACTIVITY_TYPE_TO_USE])
        activity.append(ndata)
    activity = np.array(activity)

    neuron.close()

    #if activity.shape != (num_neurons, num_time_samples):
    #    raise

    return activity, num_neurons, num_time_samples

def load_trace_scores(dataset_folder, num_time_samples):
    data_filename = os.path.join(dataset_folder, 'trace_scores.xlsx')
    if os.path.exists(data_filename):
        df = pd.read_excel( data_filename )
        return df['trace_score'].values
    else:
        scores = np.ones(num_time_samples)
        scores = [int(value) for value in scores]
        return scores


# Load all of the data that we need
def load_dataset(settings, dataset_folder):
    ds = {}

    activity, num_neurons, num_time_samples_1 = load_neuron_data(dataset_folder, settings['activity_type_to_use'])
    activity_t_ms, num_time_samples_2 = load_scope_timestamps(dataset_folder)

    if num_time_samples_1 == -1:
        num_time_samples = num_time_samples_2

    pos_x_cm, pos_y_cm, pos_t_ms, speed_smoothed_cm_per_s = load_movement_data(dataset_folder)

    include_cell = load_cell_use_data(dataset_folder, num_neurons)

    # interpolate the speed
    interp_s = interpolate.interp1d(pos_t_ms, speed_smoothed_cm_per_s, axis=0)
    i_speed_smoothed_cm_per_s = interp_s(activity_t_ms)

    # load the trace scores so we can filter based on them
    trace_scores = load_trace_scores(dataset_folder, num_time_samples)

    ds['dataset_folder'] = dataset_folder
    ds['activity'] = activity
    ds['num_neurons'] = num_neurons
    ds['num_time_samples'] = num_time_samples
    ds['activity_t_ms'] = activity_t_ms
    ds['pos_x_cm'] = pos_x_cm
    ds['pos_y_cm'] = pos_y_cm
    ds['pos_t_ms'] = pos_t_ms
    ds['speed_smoothed_cm_per_s'] = speed_smoothed_cm_per_s
    ds['i_speed_smoothed_cm_per_s'] = i_speed_smoothed_cm_per_s
    ds['include_cell'] = include_cell
    ds['trace_scores'] = trace_scores

    # Should not be hard coded
    # arena_x_min = np.min(ds['pos_x_cm'])
    # arena_x_max = np.max(ds['pos_x_cm'])
    # arena_y_min = np.min(ds['pos_y_cm'])
    # arena_y_max = np.max(ds['pos_y_cm'])

    # ds['arena_width_cm']  = arena_x_max - arena_x_min
    # ds['arena_height_cm'] = arena_y_max - arena_y_min
    # ds['arena_x_min_cm'] = arena_x_min
    # ds['arena_x_max_cm'] = arena_x_max
    # ds['arena_y_min_cm'] = arena_y_min
    # ds['arena_y_max_cm'] = arena_y_max
    # ds['arena_size_cm'] = (ds['arena_height_cm'], ds['arena_width_cm'])

    return ds




