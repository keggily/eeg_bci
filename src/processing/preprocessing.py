import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.datasets import eegbci

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

    # Transpose the data back to original shape (samples, channels) before returning
    return np.transpose(preprocessed_data)


