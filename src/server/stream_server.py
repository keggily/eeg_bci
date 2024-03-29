
"""
Source: https://github.com/labstreaminglayer/pylsl/tree/master
Author: Christian Kothe
Modified HD
"""

import sys
import getopt
from pylsl import StreamInfo, StreamOutlet, local_clock
import time
import mne
import os



srate = 160  # Sampling rate
name = 'BioSemi'  # Stream name
type = 'EEG'  # Stream type
n_channels = 64  


##If loading from repository
# subject = 'S001'  # Example subject
# run = 'R14'  # validation run for MI hands/feet
# id = subject + '_' + run

# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# data_path = 'data/MNE-eegbci-data/files/eegmmidb/1.0.0/' 
# help_string = 'stream_server.py -s <sampling_rate> -n <stream_name> -t <stream_type> -p <data_path>'

#file_path = os.path.join(root_dir, data_path, f"{subject}/{subject}{run}.edf")
#raw = mne.io.read_raw_edf(file_path, preload=True)


#If directly loading from mne
subject_id = 51 # Test on held out subject
runs = [6] # First run motor imagery: hands vs feet
path = mne.datasets.eegbci.load_data(
    subject_id, runs, update_path=False)
raw = mne.io.read_raw_edf(path[0], preload=True)



#raw.resample(srate)  # might want to add this to reduce computation time
n_channels = len(raw.info['ch_names'])

# Create a new StreamInfo and an outlet
info = StreamInfo(name, type, n_channels, srate, 'float32')
outlet = StreamOutlet(info)

print("now sending data...")
start_time = local_clock()

# Get data as numpy array
data, times = raw[:, :]

current_sample = 0
while current_sample < len(times):
    elapsed_time = local_clock() - start_time
    required_samples = int(srate * elapsed_time) - current_sample
    for sample_ix in range(required_samples):
        if current_sample >= len(times):
            break  # Stop if we have sent all data
        mysample = data[:, current_sample].tolist()  # Get the current sample
        outlet.push_sample(mysample)  # Stream the sample
        current_sample += 1
    time.sleep(0.01)  # Sleep a bit before next iteration
