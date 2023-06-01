import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from pydub import AudioSegment

# Load the MP3 file
audio = AudioSegment.from_file("hq/mp3/1.mp3", format="mp3")

# Export the MP3 file to WAV format
audio.export("hq/wav/1.wav", format="wav")

# Load the WAV file
sample_rate, data = wavfile.read("hq/wav/1.wav")

# Define the new sample rate you want to downsample to
new_sample_rate = 8000

# Use the scipy.signal.resample function to downsample the data
data_downsampled = signal.resample(data, int(len(data) * float(new_sample_rate) / sample_rate))

# Normalize the audio data to keep volume levels
data_normalized = data_downsampled / np.max(np.abs(data_downsampled))

# Write the downsampled and normalized data to a new WAV file
wavfile.write('lq/mp3/1.mp3', new_sample_rate, data_normalized)