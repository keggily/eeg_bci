# EEG BCI Online Decoding Project

This project demonstrates a pipeline for streaming EEG data using the Lab Streaming Layer (pylsl), processing the streamed data in real-time, and decoding it using machine learning models. The pipeline includes a server component that streams EEG data (simulated using a dataset from MNE), and a client component that receives the streamed data, processes it, and decodes it using a pre-trained model.

## Dataset

This project uses the EEGBCI Motor Imagery dataset from MNE. Due to its size and license, it's not included directly in the repository.

## Getting Started


#### Clone the Repository:
First, clone the repository to your local machine using the command:
```
git clone https://github.com/keggily/eeg_bci
```
#### Install Required Packages:
Install all required packages listed in the requirements.txt file. You may want to create a virtual environment. Ensure you are in the project directory then run:
```
pip install -r requirements.txt
```
## Running the Online Decoder
With the environment set up and dependencies installed, you are now ready to run the server and client components.


#### Start the EEG Data Server
Navigate to the server directory.
Run the server script:


```
cd src/server
python stream_server.py
```
This script simulates EEG data streaming by loading the EEGBCI Motor Imagery dataset from MNE and streaming it using pylsl.
#### Run the Online Decoder:
Open a new terminal window or tab.
Activate your virtual environment if not already activated.
From the root directory, run the online decoding script:


```
python online_decoder.py
```
The client script receives the streamed EEG data, processes it in real-time, and decodes it using a pre-trained machine learning model.



