import numpy as np


def align_and_concatenate(old_data, new_data, old_range=(0), new_range=(0)):
    """Align old and new datasets according to the ranges given. For any dimension change, NaN is added

    Args:
        old_data (Float): Original data matrix
        new_data (Float): New data matrix to be appended
        old_range (tuple): the data range for each dimension>0 of old_data. Uses form (idx1_range,idx2_range,...)
        new_range (tuple): the data range for each dimension>0 of new_data. Uses form (idx1_range,idx2_range,...)

    Returns:
        (Float,tuple): Concatenated_data,combined_range
    """
    # Determine the number of dimensions (excluding the 'iter' dimension)
    num_dims = old_data.ndim - 1

    # add to list if input is just a 1d array
    old_range = [old_range] if num_dims <= 1 else list(old_range)
    new_range = [new_range] if num_dims <= 1 else list(new_range)
    # old_range = [old_range] if isinstance(old_range,np.ndarray) else old_range
    # new_range = [new_range] if isinstance(new_range,np.ndarray) else new_range

    # Find first row that contains all NaN
    nan_mask = np.isnan(new_data)
    is_nan_row = np.all(nan_mask, axis=1) if num_dims > 0 else nan_mask
    first_nan_row = (
        np.max(np.argmax(is_nan_row, axis=0)) if np.any(is_nan_row) else new_data.shape[0]
    )
    new_data = new_data[:first_nan_row, ...]

    # Determine the union of ranges for each dimension
    union_ranges = []
    old_index_maps = []
    new_index_maps = []

    for dim in range(num_dims):
        old_list = list(old_range[dim])
        new_list = list(new_range[dim])
        union_list = sorted(set(old_list).union(set(new_list)))
        union_ranges.append(union_list)

        # Create index maps for this dimension
        old_index_map = [union_list.index(x) for x in old_list]
        new_index_map = [union_list.index(x) for x in new_list]
        old_index_maps.append(old_index_map)
        new_index_maps.append(new_index_map)

    # Create padded arrays filled with NaN
    union_shape = [old_data.shape[0]] + [len(r) for r in union_ranges]
    old_data_padded = np.full(union_shape, np.nan)
    union_shape = [new_data.shape[0]] + [len(r) for r in union_ranges]
    new_data_padded = np.full(union_shape, np.nan)

    # Fill the padded arrays with the original data
    old_slices = tuple(np.ix_(*old_index_maps))
    new_slices = tuple(np.ix_(*new_index_maps))

    old_data_padded[(slice(None),) + old_slices] = old_data
    new_data_padded[(slice(None),) + new_slices] = new_data

    # Concatenate the padded arrays along the 'iter' dimension
    result = np.concatenate((old_data_padded, new_data_padded), axis=0)

    return result, union_ranges[0] if num_dims == 1 else union_ranges


def append_data(old_data, old_T_range, new_data, new_T_range):
    """Old method for combining data sets by aligning and zero padding such that the T ranges match

    Args:
        old_data (_type_): numpy dataset 1, shape(n_sample,range)
        old_T_range (_type_): values associated with the 2nd dimension of old_data
        new_data (_type_): numpy dataset 2, shape(n_sample,range)
        new_T_range (_type_): values associated with the 2nd dimension of new_data

    Returns:
        _type_: concatenated data
    """
    # Find first row that contains all NaN
    nan_mask = np.isnan(new_data)
    is_nan_row = np.all(nan_mask, axis=1)
    first_nan_row = np.argmax(is_nan_row) if np.any(is_nan_row) else -1
    new_data = new_data[:first_nan_row, :]

    # Insert min_diff columns of NaN at beginning of matrix to align
    min_diff = min(old_T_range) - min(new_T_range)
    if min_diff < 0:  # Pad new data
        new_data = np.insert(new_data, [0] * abs(min_diff), np.nan, axis=1)
    elif min_diff > 0:  # Pad old data
        old_data = np.insert(old_data, [0] * min_diff, np.nan, axis=1)

    # Insert max_diff columns of NaN at end of matrix to align
    max_diff = max(old_T_range) - max(new_T_range)
    if max_diff > 0:  # Pad new data
        new_data = np.insert(new_data, [new_data.shape[1]] * max_diff, np.nan, axis=1)
    elif max_diff < 0:  # Pad old data
        old_data = np.insert(old_data, [old_data.shape[1]] * abs(max_diff), np.nan, axis=1)

    # Update total data and range
    tot_data = np.append(old_data, new_data, axis=0)
    tot_T_range = range(
        min(old_T_range[0], new_T_range[0]), max(old_T_range[-1], new_T_range[-1]) + 1
    )

    return tot_data, tot_T_range


def remove_outlier(data, num_std=3):
    """Remove any outliers in the data set outside of the given number of standard deviations

    Args:
        data (_type_): data array
        num_std (int, optional): number of std before considered an outlier. Defaults to 3.

    Returns:
        _type_: data with outliers removed
    """
    num_outlier = 1
    while num_outlier > 0:
        std = np.nanstd(data, axis=0)
        mean = np.nanmean(data, axis=0)
        # check if any value outside of 3*std
        idx = (data > mean + num_std * std) | (data < mean - num_std * std)
        num_outlier = np.count_nonzero(idx)
        data[idx] = np.NaN
    return data


def clean_data(data):
    """Remove any invalid points in the data such as NaN

    Args:
        data (_type_): data array
    """
    data[np.isinf(data)] = np.nan
