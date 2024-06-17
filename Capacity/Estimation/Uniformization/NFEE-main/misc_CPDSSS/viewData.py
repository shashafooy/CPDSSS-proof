import numpy as np

def append_data(old_data, old_T_range, new_data, new_T_range):
    """
    Adjust both datasets to match their T_range before appending
    """
    #Find first row that contains all NaN
    nan_mask = np.isnan(new_data)
    is_nan_row = np.all(nan_mask,axis=1)
    first_nan_row = np.argmax(is_nan_row) if np.any(is_nan_row) else -1
    new_data = new_data[:first_nan_row,:]
    #Insert min_diff columns of NaN at beginning of matrix to align
    min_diff = min(old_T_range) - min(new_T_range)
    if(min_diff < 0): #Pad new data     
        new_data = np.insert(new_data,[0]*abs(min_diff),np.nan,axis=1)
    elif(min_diff > 0): #Pad old data
        old_data = np.insert(old_data,[0]*min_diff,np.nan,axis=1)

    #Insert max_diff columns of NaN at end of matrix to align
    max_diff = max(old_T_range) - max(new_T_range)
    if(max_diff > 0): #Pad new data
        new_data = np.insert(new_data,[new_data.shape[1]]*max_diff,np.nan,axis=1)
    elif(max_diff < 0): #Pad old data
        old_data = np.insert(old_data,[old_data.shape[1]]*abs(max_diff),np.nan,axis=1)
    
    #Update total data and range
    tot_data = np.append(old_data,new_data,axis=0)
    tot_T_range = range(min(old_T_range[0],new_T_range[0]), max(old_T_range[-1],new_T_range[-1])+1)

    return tot_data, tot_T_range

def remove_outlier(data, num_std=3):
    num_outlier = 1
    while num_outlier > 0:
        std = np.nanstd(data,axis=0)
        mean = np.nanmean(data,axis=0)
        #check if any value outside of 3*std
        idx = (data > mean +num_std*std) |  (data < mean - num_std*std)
        num_outlier = np.count_nonzero(idx)
        data[idx]=np.NaN
    return data

def clean_data(data):
    '''
    Remove any invalid data such as Inf and replace with NaN
    '''
    data[np.isinf(data)]=np.nan
