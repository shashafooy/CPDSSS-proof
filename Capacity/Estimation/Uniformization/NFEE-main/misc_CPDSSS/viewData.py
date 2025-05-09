import re
import numpy as np
import os
import util.io


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
    if not isinstance(new_data, np.ndarray):
        old_list = list(old_range)
        new_list = list(new_range)
        union_list = sorted(set(old_list).union(set(new_list)))

        return new_data, union_list
    # Determine the number of dimensions (excluding the 'iter' dimension)
    num_dims = old_data.ndim - 1

    # add to list if input is just a 1d array
    if isinstance(old_range, tuple):
        old_range = list(old_range)
        new_range = list(new_range)
    else:
        old_range = [old_range]
        new_range = [new_range]
    num_range_dim = len(old_range)
    # old_range = [old_range] if not isinstance(old_range, tuple) else list(old_range)
    # new_range = [new_range] if not isinstance(new_range, tuple) else list(new_range)
    # num_range_dim = len(old_range)
    # if num_range_dim>num_dims:
    #     old_range = old_range * (num_dims/num_range_dim)
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

    for dim in range(min(num_dims, num_range_dim)):
        # dim = min(dim, len(old_range) - 1)
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
    union_shape = (
        union_shape + [old_data.shape[2]] if num_range_dim == 1 and num_dims > 1 else union_shape
    )
    old_data_padded = np.full(union_shape, np.nan)
    union_shape = [new_data.shape[0]] + [len(r) for r in union_ranges]
    union_shape = (
        union_shape + [old_data.shape[2]] if num_range_dim == 1 and num_dims > 1 else union_shape
    )
    new_data_padded = np.full(union_shape, np.nan)

    # Fill the padded arrays with the original data
    old_slices = tuple(np.ix_(*old_index_maps))
    new_slices = tuple(np.ix_(*new_index_maps))

    old_data_padded[(slice(None),) + old_slices] = old_data
    new_data_padded[(slice(None),) + new_slices] = new_data

    # Concatenate the padded arrays along the 'iter' dimension
    result = np.concatenate((old_data_padded, new_data_padded), axis=0)

    return result, union_ranges[0] if num_dims == 1 or num_range_dim == 1 else union_ranges


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
    if not isinstance(data, np.ndarray):
        return data
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
    if not isinstance(data, np.ndarray):
        return data
    data[np.isinf(data)] = np.nan


def read_data(filepath, remove_outliers=True, outlier_std=3):
    init = False
    for filename in os.listdir(filepath):
        if re.search(r"-1_iter|\.old", filename):
            continue
        filename = os.path.splitext(filename)[0]  # remove extention
        file_items = util.io.load(os.path.join(filepath, filename))
        _T_range = file_items[0]

        if not init:
            init = True
            arrays = []

            # create empty arrays with the same shape minus first dimension. First dimension in the sample index
            for item in file_items[1:]:
                if isinstance(item, np.ndarray):
                    arrays.append(np.empty((0, *item.shape[1:])))
                else:
                    arrays.append(item)
            T_range = _T_range

        for i, item in enumerate(file_items[1:]):
            arrays[i], new_range = align_and_concatenate(arrays[i], item, T_range, _T_range)
        T_range = new_range

    for item in arrays:
        if remove_outliers:
            remove_outlier(item, num_std=outlier_std)
        clean_data(item)

    return T_range, arrays


class Data:
    def __init__(self, data):
        self.data = data
        self.refresh_stats()

    def refresh_stats(self):
        self.mean = np.nanmean(self.data, axis=0)
        self.var = np.nanvar(self.data, axis=0)
        self.std = np.nanstd(self.data, axis=0)
        self.MSE = np.nanmean(self.data**2, axis=0)
