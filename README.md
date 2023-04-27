# Spectrogram Upscaling using Generative Adversarial Networks (GANs)
## Overview
This project is an implementation of a Generative Adversarial Network (GAN) for spectrogram upscaling. The goal of the project is to train a GAN model that takes a low-resolution spectrogram as input and generates a high-resolution spectrogram as output. The model is trained using a dataset of paired low-quality and high-quality spectrograms, and the generated high-resolution spectrograms are expected to be similar to the original high-quality spectrograms.

## Requirements
To run this project, you will need to have the following software installed:

- Python 3.7 or later
- TensorFlow 2.0 or later
- NumPy
- Librosa
- OpenCV

# Environment Setup

To use this repository, you will need to create and activate a `conda` environment with the necessary dependencies.

1. Clone this repository to your local machine.
2. Create a new `conda` environment using the `upsampler.yml` file: conda env create -f upsampler.yml
3. Activate the environment: conda activate upsampler