import numpy as np
import datetime

def gen_datetimetag():
    x = str(datetime.datetime.now())
    x = x.replace(' ', '_')
    x = x.replace('-', '_')
    x = x.replace(':', '_')
    x = x.replace('.', '_')
    return x


def bin_position(x_bounds, n_bins, x_samples):
    # Bins are defined by edges
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)

    # Centers of the bins are between two edges
    x_bin_centers = np.zeros(n_bins)
    for i in range(len(x_bin_centers)):
        x_bin_centers[i] = x_edges[i] + (x_edges[i + 1] - x_edges[i]) / 2

    # Bin the data
    x_binned = np.digitize(x_samples, x_edges)

    # If points are outside the edges, so the binned value to zero
    # and record the bad index
    out_of_bounds_sample_indices = []
    for i in range(len(x_binned)):
        xs = x_binned[i]
        if xs == 0 or xs == len(x_edges):
            x_binned[i] = -1
            out_of_bounds_sample_indices.append(i)

    out_of_bounds_sample_indices = np.array(out_of_bounds_sample_indices)

    # Convert from bin numbers like 1,2,3, to indices 0, 1, 2
    x_binned = x_binned - 1

    return x_binned, x_bin_centers, out_of_bounds_sample_indices


def accumulate_map_create(siz, pos_x_binned, pos_y_binned, value=1):
    accumulate_map = np.zeros(siz)
    for index, ind in enumerate(list(zip(pos_y_binned, pos_x_binned))):
        if type(value) == np.ndarray or type(value) == list:
            accumulate_map[ind] += value[index]
        else:
            accumulate_map[ind] += value
    return accumulate_map

