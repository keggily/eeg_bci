import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.datasets import eegbci
from sklearn.preprocessing import RobustScaler
import joblib
import os

def preprocess_single_trial(single_trial, sfreq, channel_names):
    """
    Preprocess a single trial and maintain original shape.
    
    Parameters:
    single_trial (numpy.ndarray): The single trial data, shape (samples, channels).
    sfreq (float): Sampling frequency of the data.
    channel_names (list): List of channel names.

    Returns:
    numpy.ndarray: The preprocessed single trial data in original shape (samples, channels).
    """

    # Transpose single_trial to shape (channels, samples) for RawArray
    single_trial_transposed = np.transpose(single_trial)

    # Create an Info object
    info = create_info(ch_names=channel_names, sfreq=sfreq, ch_types=['eeg'] * len(channel_names))
    
    # Create a RawArray object
    single_trial_raw = RawArray(single_trial_transposed, info)

    # Apply preprocessing steps such as filtering
    single_trial_raw.filter(7.0, 30.0, fir_design='firwin')

    # Extract preprocessed data from the Raw object
    preprocessed_data = single_trial_raw.get_data()
    
    # Normalize data
    preprocessed_data = robust_scale(preprocessed_data)

    # Transpose the data back to original shape (samples, channels) before returning
    return np.transpose(preprocessed_data)


import numpy as np

def robust_scale(X):
    """
    Applies robust scaling to each channel of EEG data independently.
    
    Parameters:
    - X (numpy.ndarray): EEG data of shape (samples, channels).
    
    Returns:
    - Scaled EEG data with the same shape as X.
    """
    scaled_X = np.zeros_like(X)
    for channel in range(X.shape[1]):
        # Compute the median and interquartile range for each channel
        median = np.median(X[:, channel])
        q75, q25 = np.percentile(X[:, channel], [75 ,25])
        iqr = q75 - q25
        
        # Scale the channel data
        if iqr > 0:
            scaled_X[:, channel] = (X[:, channel] - median) / iqr
        else:
            # Handle the case where IQR is 0 to avoid division by zero
            scaled_X[:, channel] = X[:, channel] - median
            
    return scaled_X


