import scipy.io.wavfile as wavfile
import scipy.signal as signal

# Load the WAV file
sample_rate, data = wavfile.read('your_file.wav')

# Define the new sample rate you want to downsample to
new_sample_rate = 8000

# Use the scipy.signal.resample function to downsample the data
data_downsampled = signal.resample(data, int(len(data) * float(new_sample_rate) / sample_rate))

# Write the downsampled data to a new WAV file
wavfile.write('your_downsampled_file.wav', new_sample_rate, data_downsampled)