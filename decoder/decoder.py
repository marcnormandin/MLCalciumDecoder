# 2022-02-08. Attempt to implement the algorithm outlined in
#            A distributed neural code in the dentate gyrus and in CA1 (Fusi, 2020)
# 2022-02-09. Made the code use dictionaries and smaller functions so
#            that we can have more than one dataset loaded at a time for when we do prediction.
# 2022-02-10. Made my own version of np.digitize because I need to handle the out of bounds point differently.
#            For debugging, I wrote code to create synthetic activity data based on simulated placefield
#
#            I divide the bin votes by the number of times a BIN could have been selected, which gives a "vote rate"
#            The predictions for the synthetic are almost perfect (basically perfect)
#
#            Added loading of cell_use data to be used to either include or exclude a cell used for the predictions
#            Added loading of the speed of the animal so that we can exclude data from low-speeds
#
# 2022-02-14. Started this version as a copy of Version 2.
# 2022-02-14. Made this separate version to load in Kinsky's data from Medeley
# 2022-02-16. Copied the code out of the notebook into separate python files.

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

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def score_data(settings, ds):
    # Some of the binned samples may be junk, so we need to clean them up while maintaining the data consistency
    #bad_sample_indices = np.concatenate((ds['pos_x_out_of_bounds_sample_indices'], ds['pos_y_out_of_bounds_sample_indices'],
    #                                    [index for index, value in enumerate(ds['i_speed_smoothed_cm_per_s']) if value <= 50],
    #                                    [index for index, value in enumerate(ds['activity_t_ms']) if value <= discard_seconds*1000]))
    bad_sample_indices = np.concatenate((
                                         ds['pos_x_out_of_bounds_sample_indices'], 
                                         ds['pos_y_out_of_bounds_sample_indices'],
                                         [index for index, value in enumerate(ds['activity_t_ms']) if value <= settings['filter_discard_initial_s']*1000],
                                         [index for index, value in enumerate(ds['i_speed_smoothed_cm_per_s']) if value <= settings['filter_speed_threshold_cm_per_s']]
                                         ))
    bad_sample_indices = np.unique(bad_sample_indices)
    bad_sample_indices = [int(x) for x in bad_sample_indices]
    

    # Some of the cells may be junk, so we need to clean them up while maintaining the data consistency
    bad_cell_indices = np.concatenate((
                                        [index for index, value in enumerate(ds['include_cell']) if value == False],
                                        [index for index, value in enumerate(ds['trace_scores']) if value == 0]
                                        ))
    bad_cell_indices = np.unique(bad_cell_indices)
    bad_cell_indices = [int(x) for x in bad_cell_indices]

    return bad_sample_indices, bad_cell_indices


def prepare_model_data(settings, ds):
    # We will first put the data in the correct dimenions, and then we will extract the segments that correspond to either of two
    # bin locations
    # X needs to be (n_samples, n_features)
    # y needs to be (n_samples, ) <-- second is empty
    activity = ds['activity']
    true_bins_linear = ds['pos_xy_binned_linear']

    num_original_cells = activity.shape[0]
    #num_original_time_samples = activity.shape[1]

    valid_cell_indices = np.arange(num_original_cells)
    print('There are the valid cell indices: ', valid_cell_indices)

    # We only want to train the decoder on good data
    bad_sample_indices, bad_cell_indices = score_data(settings, ds)

    print(bad_sample_indices)
    print(bad_cell_indices)

    print('Out of %d total samples, %d will not be used.' %(ds['num_time_samples'], len(bad_sample_indices)))
    print('Out of %d total cells, %d will not be used.' %(ds['num_neurons'], len(bad_cell_indices)))


    print(activity.shape)

    if len(bad_sample_indices):
        activity = np.delete(activity, bad_sample_indices, axis = 1) # columns which are the samples
        true_bins_linear = np.delete(true_bins_linear, bad_sample_indices)

    print(activity.shape)
    
    if len(bad_cell_indices):
        activity = np.delete(activity, bad_cell_indices, axis = 0) # rows which are the cell traces
        valid_cell_indices = np.delete(valid_cell_indices, bad_cell_indices)

    print(activity.shape)

    X_all = activity.T
    y_all = true_bins_linear

    # Scale the data
    X_scaler = sk.preprocessing.StandardScaler().fit(X_all)
    X_all = X_scaler.transform(X_all)

    return X_all, y_all, valid_cell_indices

def split_data(X_all, y_all, training_size_fraction=0.8):
    # Split the data so that we can predict on data that we haven't trained with
    num_time_samples = X_all.shape[0]

    train_sample_max_index = round(num_time_samples * training_size_fraction)

    train_time_samples_inds = np.arange(train_sample_max_index)

    decode_time_samples_inds = np.arange(train_sample_max_index+1, num_time_samples)

    X_train = X_all[train_time_samples_inds,:]
    y_train = y_all[train_time_samples_inds]

    X_decode = X_all[decode_time_samples_inds,:]
    y_decode = y_all[decode_time_samples_inds]

    return X_train, y_train, X_decode, y_decode


def train_model_for_two_bins(X_all, y_all, loc_a_lind, loc_b_lind):
    # print('Testing a: %d, b: %d' % (loc_a_lind, loc_b_lind))

    # Get indices into the data that are associated with either of the two locations
    loc_a_sample_linds = [i for i, x in enumerate(y_all) if x == loc_a_lind]
    loc_b_sample_linds = [i for i, x in enumerate(y_all) if x == loc_b_lind]

    if len(loc_a_sample_linds) <= 2 or len(loc_b_sample_linds) <= 2:
        return [], 0, False

    try:
        # Extract only the samples that correspond to either of those locations
        loc_sample_linds = []
        loc_sample_linds.extend(loc_a_sample_linds)
        loc_sample_linds.extend(loc_b_sample_linds)

        y_samples = np.concatenate((np.zeros(len(loc_a_sample_linds)), np.ones(len(loc_b_sample_linds))))
        X_samples = X_all[loc_sample_linds, :]

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_samples, y_samples, test_size=0.50,
                                                                               random_state=42)

        # Config
        MAX_ITERS = 1000

        # Create the classification model
        SVM = svm.LinearSVC(max_iter=MAX_ITERS, class_weight='balanced')
        SVM.fit(X_train, y_train)
        SVM.predict(X_test)

        score = SVM.score(X_test, y_test)

        if SVM.n_iter_ != MAX_ITERS:
            is_valid = True
        else:
            is_valid = False
    except ValueError as e:
        return [], 0, False

    return SVM, score, is_valid

def train(X_train, y_train, arena_size_binned):
    # Linearize the bin combinations so that we can treat that as objects.
    num_linear_bins = arena_size_binned[0] * arena_size_binned[1]
    linear_bin_combinations = list(itertools.combinations(np.arange(num_linear_bins), 2))

    # Train the models using the training data
    SVM_model = []
    SVM_scores = []
    SVM_is_valid = []

    # TRAIN THE MODELS
    #     scaler = sk.preprocessing.StandardScaler().fit(X_train)
    #     X_train_scaled = scaler.transform(X_train)
    for index, comb in enumerate(linear_bin_combinations):
        loc_a_lind = comb[0]
        loc_b_lind = comb[1]
        SVM, score, is_valid = train_model_for_two_bins(X_train, y_train, loc_a_lind, loc_b_lind)

        # Store
        SVM_model.append(SVM)
        SVM_scores.append(score)
        SVM_is_valid.append(is_valid)

    # Use only the models that are valid
    SVM_to_use = [i for i, x in enumerate(SVM_is_valid) if x == True]

    # Compute the chances that each bin location had to be selected
    linear_chances = np.zeros(num_linear_bins)
    for index, comb in enumerate(linear_bin_combinations):
        loc_a_lind = comb[0]
        loc_b_lind = comb[1]

        if not SVM_is_valid[index]:
            continue

        linear_chances[loc_a_lind] += 1
        linear_chances[loc_b_lind] += 1

    # Store
    model = {}
    model['arena_size_binned'] = arena_size_binned
    model['svm'] = SVM_model
    model['scores'] = SVM_scores
    model['is_valid'] = SVM_is_valid
    model['chances'] = linear_chances
    model['num_linear_bins'] = num_linear_bins
    model['linear_bin_combinations'] = linear_bin_combinations

    return model


def predict(model, X_predict):
    # Each row are votes for a sample, and columns are the linear
    num_samples = X_predict.shape[0]
    linear_votes_per_sample = np.zeros((num_samples, model['num_linear_bins']))

    # Perform the prediction
    for index, comb in enumerate(model['linear_bin_combinations']):
        loc_a_lind = comb[0]
        loc_b_lind = comb[1]
        if not model['is_valid'][index]:
            continue

        predicted_classes = model['svm'][index].predict(X_predict)
        for sample_index, predicted_class in enumerate(predicted_classes):
            if predicted_class == 0:
                predicted_linear_bin_index = loc_a_lind
            else:
                predicted_linear_bin_index = loc_b_lind

            linear_votes_per_sample[sample_index, predicted_linear_bin_index] += model['scores'][index]

    # Normalize by the number of chances that each location had, otherwise the results are biased.
    linear_votes_per_sample_weighted = np.zeros(linear_votes_per_sample.shape)
    for i in range(linear_votes_per_sample_weighted.shape[0]):
        for j in range(linear_votes_per_sample_weighted.shape[1]):
            chances = model['chances'][j]
            if chances != 0:
                linear_votes_per_sample_weighted[i, j] = linear_votes_per_sample[i, j] / (1.0 * chances)
            else:
                linear_votes_per_sample_weighted[i, j] = 0

    # Return one prediction per sample
    y_predicted = np.argmax(linear_votes_per_sample_weighted, axis=1)

    # Create a list of 2D prediction maps
    prediction_maps = []
    for sample_index in range(num_samples):
        votes = linear_votes_per_sample_weighted[sample_index, :]

        [ai, aj] = np.unravel_index(np.arange(len(votes)), model['arena_size_binned'], order='C')
        prediction_map = util.accumulate_map_create(model['arena_size_binned'], aj, ai, value=votes)
        prediction_map = np.flipud(prediction_map)

        prediction_maps.append(prediction_map)

    return y_predicted, prediction_maps
    


    