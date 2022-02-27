import numpy as np
import h5py

# Load all of the data that we need
def load_dataset(settings, dataset_filename):
    #dataset_filename = 'R:\calcium_decoding\datasets\kinsky_mouse1_day1_1octagon.mat'

    data = h5py.File(dataset_filename, 'r')

    activity = np.array(data['RawTrace']).T
    num_neurons = activity.shape[0]
    num_time_samples = activity.shape[1]
    # Kinksy times are in seconds so convert to milliseconds
    activity_t_ms = np.array(data['time_interp']).flatten().T*1000
    activity_t_ms = activity_t_ms - activity_t_ms[0]
    pos_x_cm = np.array(data['x_adj_cm']).flatten().T
    pos_x_cm = pos_x_cm - np.min(pos_x_cm)
    pos_y_cm = np.array(data['y_adj_cm']).flatten().T
    pos_y_cm = pos_y_cm - np.min(pos_y_cm)
    pos_t_ms = np.array(data['time_interp']).flatten().T*1000
    pos_t_ms = pos_t_ms - pos_t_ms[0]
    speed_smoothed_cm_per_s = np.array(data['speed']).flatten().T
    i_speed_smoothed_cm_per_s = np.array(data['speed']).flatten().T
    include_cell = np.ones(num_neurons)
    trace_scores = np.ones(num_neurons)
    arena_width_cm = 35
    arena_height_cm = 35

    data.close()

    # Store the data
    ds = {}

    ds['dataset_folder'] = ''
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
    ds['trace_scores'] = []

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
