import numpy as np
from mne import create_info
from mne.io import RawArray

def preprocess_single_trial(single_trial, sfreq, channel_names):
    """
    Preprocess a single trial using channel names extracted from EEGBCI dataset.
    
    Parameters:
    single_trial (numpy.ndarray): The single trial data, shape (samples, channels).
    sfreq (float): Sampling frequency of the data.

    Returns:
    numpy.ndarray: The preprocessed single trial data.
    """
    single_trial_transposed = np.transpose(single_trial)
    
    # Create an Info object with the extracted channel names
    info = create_info(ch_names=channel_names, sfreq=sfreq, ch_types=['eeg'] * len(channel_names))
    
    # Create a RawArray with the transposed data and newly created info
    single_trial_raw = RawArray(single_trial_transposed, info)

    # Set channel names
    eegbci.standardize(single_trial_raw) 

    # Set channel locations
    single_trial_raw.set_montage('standard_1005')   
    
    # Apply bandpass filter to the data
    single_trial_raw.filter(7.0, 30.0, fir_design='firwin')

    # Make shape (1, channels, samples) for compatibility with the model
    single_trial_raw = single_trial_raw.get_data()[None, :, :]
 
    # Return the preprocessed data
    return single_trial_raw

