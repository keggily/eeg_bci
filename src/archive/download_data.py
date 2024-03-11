from mne.datasets import eegbci
import os

def download_eeg_data(subjects, runs, data_path):
    """
    Download EEG data for the specified subjects and runs into the given data_path.
    
    Parameters:
    subjects : list of int
        Subject numbers for which to download the data.
    runs : list of int
        Run numbers for which to download the data.
    data_path : str
        Path to the directory where the data will be stored.
    """
    
    # Make sure the data_path exists
    os.makedirs(data_path, exist_ok=True)

    # Download data for each subject and run
    for subject in subjects:
        eegbci.load_data(subject, runs, path=data_path)
        print(f'Downloaded data for subject {subject}.')

# Settings for downloading
subjects = [1] 
runs = [4, 8, 12]  # motor imagery: open / close fists

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, 'data')

# Download to data file
download_eeg_data(subjects, runs, data_path)
