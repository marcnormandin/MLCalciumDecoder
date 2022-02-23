def simulate_placefield(sz, sigma):
    A = np.zeros(sz)
    i = np.random.randint(A.shape[0])
    j = np.random.randint(A.shape[1])
    A[i, j] = 1
    B = fi.gaussian_filter(A, sigma)
    B = B / np.max(B)
    return B


def simulate_neuron_activity(x_bounds, y_bounds, n_pos_x_bins, n_pos_y_bins, i_pos_x_cm, i_pos_y_cm):
    # Bin the position data
    pos_x_binned, pos_x_bin_centers_cm, pos_x_out_of_bounds_sample_indices = ml_bin_position(x_bounds, n_pos_x_bins,
                                                                                             i_pos_x_cm)
    pos_y_binned, pos_y_bin_centers_cm, pos_y_out_of_bounds_sample_indices = ml_bin_position(y_bounds, n_pos_y_bins,
                                                                                             i_pos_y_cm)

    # Y-dimension comes first, THEN X-dimension
    arena_size_binned = (n_pos_y_bins, n_pos_x_bins)

    bin_x_spacing_cm = abs(pos_x_bin_centers_cm[1] - pos_x_bin_centers_cm[0])
    bin_y_spacing_cm = abs(pos_y_bin_centers_cm[1] - pos_y_bin_centers_cm[0])
    bin_spacing_cm = (bin_x_spacing_cm + bin_y_spacing_cm) / 2

    sigma_cm = 5
    sigma_bin = sigma_cm / bin_spacing_cm

    place_field = simulate_placefield(arena_size_binned, sigma_bin)

    sim_activity = np.zeros(len(pos_x_binned))

    for index, pos in enumerate(zip(pos_x_binned, pos_y_binned)):
        sim_activity[index] = place_field[pos[1], pos[0]]

    return sim_activity
