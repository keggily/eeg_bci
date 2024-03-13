from sklearn.base import BaseEstimator, TransformerMixin
import pywt
import numpy as np

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq, wavelet='db4', num_levels=4):
        self.sfreq = sfreq
        self.wavelet = wavelet
        self.num_levels = num_levels
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_wavelet_features(X, self.sfreq, self.wavelet, self.num_levels)
    

def extract_wavelet_features(epochs, sampling_freq, wavelet='db4', num_levels=4):
    n_epochs, n_channels, n_samples = epochs.shape
    features = []

    for epoch in epochs:
        epoch_features = []
        for channel in epoch:
            coeffs = pywt.wavedec(channel, wavelet, level=num_levels)
            channel_features = []
            for coeff in coeffs[1:]:  
                channel_features.extend([
                    np.mean(np.abs(coeff)),  
                    np.std(coeff),  
                    np.max(np.abs(coeff)),
                ])
            epoch_features.append(channel_features)

        features.append(np.concatenate(epoch_features))

    return np.array(features)
