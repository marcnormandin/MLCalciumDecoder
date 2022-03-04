import numpy as np

import dataset
#from dataset.kinsky_calcium import load_dataset

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

from util import gen_datetimetag

#from dataset.muzziolab_calcium import load_dataset
from dataset.kinsky_calcium import load_dataset as load_dataset_kinsky
from dataset.muzziolab_calcium import load_dataset as load_dataset_muzzio


import decoder.decoder as decoder

if __name__ == '__main__':

    # Specify a folder that comes from one of the trial analysis folder from cheng's task 2c CA1 project
    settings = {
        'activity_type_to_use': 'trace_filt',  # or trace_raw, spikes
        'n_pos_x_bins': 41,
        'n_pos_y_bins': 18,
        'training_size_fraction': 0.6,
        'filter_speed_threshold_cm_per_s': 2.0,
        'filter_discard_initial_s': 0,
        'output_directory': os.path.join('R:\\calcium_decoding\\analysis', gen_datetimetag())
    } 

    IS_MUZZIO_DATA = False

    if IS_MUZZIO_DATA:
        #dataset_folder = 'R:\\calcium_decoding\\datasets\\CMG169_CA1_s1_trial_1'
        #dataset_folder = 'R:\\calcium_decoding\\datasets\\CMG169_CA1_s3_trial_2'
        dataset_folder = 'R:\\calcium_decoding\\datasets\\MRL016_CA1_transgenic_forage_cylinder_H15_M25_S14'
        dataset_label = dataset_folder
        print('Loading Muzzio Lab Data:', dataset_folder);
        
        ds = load_dataset_muzzio(settings, dataset_folder)
    else:
        #dataset_filename = 'R:\calcium_decoding\datasets\kinsky_mouse1_day1_1octagon.mat'
        dataset_filename = 'R:\calcium_decoding\datasets\Kinsky_Mouse1_G30_day6_2env180_1octagon.mat'
        dataset_label = dataset_filename
        print('Loading Kinsky Data:', dataset_filename)

        ds = load_dataset_kinsky(settings, dataset_filename)

    
    dataset.bin_path(settings, ds)

    # Prepare the training data
    X_all, y_all, model_cell_indices = decoder.prepare_model_data(settings, ds)
    X_train, y_train, X_predict, y_true = decoder.split_data(X_all, y_all, settings['training_size_fraction'])

    # Create the model (train it)
    print('Training')
    model = decoder.train(X_train, y_train, ds['arena_size_binned'])

    # These are predictions because they don't decode any data trained on.
    print('Predicting')
    y_predict, prediction_maps = decoder.predict(model, X_predict)

    # Now decode the entire data so we can see the total performance
    print('Decoding')
    X_decode, y_decode_true = decoder.prepare_decoding_data(settings, ds, model_cell_indices)
    y_decode_predict, decode_maps = decoder.predict(model, X_decode)

    # Save the results
    print('Saving')
    os.makedirs(settings['output_directory'], exist_ok=True)
    pickle.dump([IS_MUZZIO_DATA, dataset_label, settings, ds, model_cell_indices, X_train, y_train, X_predict, y_true, model, y_predict, prediction_maps, 
        X_decode, y_decode_true, y_decode_predict, decode_maps], open(os.path.join(settings['output_directory'], 'analysis.p'),  'wb') )

    print('Decoder output saved to ', settings['output_directory'])

    # # Show the true and predicted bins as a function of time sample
    # plt.figure(figsize=(18, 9))
    # plt.plot(np.arange(len(y_decode)), y_decode, 'b.')
    # plt.plot(np.arange(len(y_decode)), y_predict, 'ro')
    # plt.show()

    # input('Press any key to continue...')
