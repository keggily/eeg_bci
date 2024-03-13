from pylsl import StreamInlet, resolve_stream
import numpy as np
import joblib  # Used for loading sklearn models
import sys
import os
import torch

sys.path.append('./src/processing')
from preprocessing import *

sys.path.append('./models')
from eegconformer import EEGConformer


models_dir = './models/trained/'
results_dir = './results/'

# Configuration
srate = 160  #Sampling rate of the EEG data
epoch_length_sec = 5  # Length of the desired sample in seconds
samples_needed = srate * epoch_length_sec  # Number of samples needed for ~5 seconds
n_chans = 64  # Number of EEG channels
# Manually define from eegbci dataset
channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']



# Online decoding for EEG Conformer

print("Looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

def load_eegconformer_model(model_path, n_chans, srate, n_times):
    """
    Initialize and load the pre-trained EEGConformer model from a checkpoint.
    """
    model = EEGConformer(
    n_outputs= 2,
    n_chans = n_chans,
    sfreq= srate,
    n_times = n_times,
    n_filters_time=40, 
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    drop_prob=0.7,
    att_depth=3,
    att_heads=10,
    att_drop_prob=0.7,
    final_fc_length='auto', # could be 'auto' or int
    return_features=False, # returns the features before the last classification layer if True
    chs_info=None,
    input_window_seconds=None,
    add_log_softmax=True,
)
    
    # Load the model checkpoint

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def online_decode(inlet, samples_per_epoch, model, srate, channel_names, results_dir):
    """
    Continuously pull samples from the LSL stream, decode them using the provided model,
    and save the predictions to a text file.
    """
    buffer = []  # Initialize buffer for accumulating samples

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Open a text file to save predictions
    with open(os.path.join(results_dir, 'predictions.txt'), 'w') as pred_file:
        while True:
            # Pull sample from LSL stream
            sample, timestamp = inlet.pull_sample()
            if sample:  # Ensure sample is not None
                buffer.append(sample)

            # Check if buffer has enough samples to form an epoch
            if len(buffer) >= samples_per_epoch:
                epoch = np.array(buffer[:samples_per_epoch])
                buffer = buffer[samples_per_epoch:]  # Reset the buffer

                # Decode the epoch
                prediction = decode_sample(epoch, model, srate, channel_names)

                print(f"Timestamp: {timestamp}, Prediction: {prediction}")
                
                # Append the prediction and its timestamp to the text file
                pred_file.write(f"Timestamp: {timestamp}, Prediction: {prediction}\n")
                pred_file.flush()  # Ensure the prediction is immediately written to the file

def decode_sample(epoch, model, srate, channel_names):
    """
    Process and decode a single epoch of EEG data using the EEGConformer model.
    """
    # Preprocess the epoch
    epoch = preprocess_single_trial(epoch, srate, channel_names)
    epoch = epoch.T
    #add 1 dimension at front
    epoch = np.expand_dims(epoch, axis=0)
    epoch = torch.from_numpy(epoch).to(torch.float32)
    
    
    model.eval()
    with torch.no_grad():
        logits = model(epoch)
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction


loaded_model = load_eegconformer_model(os.path.join(models_dir, 'conformer.pth'), n_chans, srate, samples_needed)
online_decode(inlet, samples_needed, loaded_model, srate, channel_names, results_dir)



# Sample online decoder for simple decoding (not recommended due to poor performance cross subject)


# def online_decode(inlet, samples_per_epoch, loaded_model, srate, channel_names, results_dir):
#     """
#     Continuously pull samples from the LSL stream and decode them.
#     """
#     buffer = []  # Initialize buffer for accumulating samples
#     pred_hist = []  # History of predictions

#     # Open a text file to save predictions
#     with open(os.path.join(results_dir, 'predictions.txt'), 'w') as pred_file:
#         while True:
#             # Pull sample from LSL stream
#             sample, timestamp = inlet.pull_sample()
#             if sample:  # Ensure sample is not None
#                 buffer.append(sample)

#             # Check if buffer has enough samples to form an epoch
#             if len(buffer) >= samples_per_epoch:
#                 epoch = np.array(buffer[:samples_per_epoch])
#                 buffer = buffer[samples_per_epoch:]  # Remove the processed samples from the buffer

#                 # Decode the epoch
#                 prediction = decode_sample(epoch, loaded_model, srate, channel_names)
#                 pred_hist.append(prediction)  # Append the prediction to the history
#                 print(f"Timestamp: {timestamp}, Prediction: {prediction}")

#                 # Write the prediction to the text file
#                 pred_file.write(f"{timestamp}, {prediction}\n")
#                 pred_file.flush()  # Ensure the prediction is written immediately

#             # # Condition to stop after decoding 15 epochs
#             # if len(pred_hist) >= 15:
#             #     print("Saved 15 epochs and their predictions.")
#             #     break  # Exit the while loop

# def decode_sample(epoch, loaded_model, srate, channel_names):
#     """
#     Process and decode an epoch of EEG data.
#     """
#     # Assuming preprocess_single_trial is implemented as previously discussed
#     preprocessed_epoch = preprocess_single_trial(epoch, srate, channel_names) 
#     prediction = loaded_model.predict(preprocessed_epoch)  
#     return prediction


# # Simple decoder using CSP and logistic regression
# model_path = os.path.join(models_dir, 'csp_logistic.pkl')
# loaded_model = joblib.load(model_path)

# online_decode(inlet, samples_needed, loaded_model, srate, channel_names, results_dir)




