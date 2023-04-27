import librosa
import cv2
import numpy as np

# Load the high-quality and low-quality audio files
hq_audio, sr = librosa.load('hq/1.wav')
lq_audio, sr = librosa.load('lq/1.wav')

# Compute the spectrograms for both audio files
hq_spectrogram = librosa.feature.melspectrogram(hq_audio, sr=sr)
lq_spectrogram = librosa.feature.melspectrogram(lq_audio, sr=sr)

# Resize the low-quality spectrogram to match the dimensions of the high-quality spectrogram
lq_spectrogram = cv2.resize(lq_spectrogram, (hq_spectrogram.shape[1], hq_spectrogram.shape[0]))

# Normalize the spectrograms to the range [-1, 1]
hq_spectrogram = (hq_spectrogram - np.max(hq_spectrogram) / 2) / (np.max(hq_spectrogram) / 2)
lq_spectrogram = (lq_spectrogram - np.max(lq_spectrogram) / 2) / (np.max(lq_spectrogram) / 2)

# Create a NumPy array to store the spectrograms
spectrograms = np.stack([lq_spectrogram, hq_spectrogram], axis=-1)

# Save the spectrograms to disk
np.save('spectrograms.npy', spectrograms)