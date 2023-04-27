import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

# Define the generator network
def build_generator(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder layers
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv2 = Conv2D(128, 3, strides=(2,2), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv3 = Conv2D(256, 3, strides=(2,2), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    # Decoder layers
    up1 = UpSampling2D(size=(2,2))(conv3)
    up1 = Conv2D(128, 3, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.2)(up1)
    merge1 = Concatenate()([up1, conv2])
    up2 = UpSampling2D(size=(2,2))(merge1)
    up2 = Conv2D(64, 3, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.2)(up2)
    merge2 = Concatenate()([up2, conv1])
    outputs = Conv2D(1, 3, activation='tanh', padding='same')(merge2)
    
    model = Model(inputs, outputs)
    return model

# Define the discriminator network
def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv2 = Conv2D(128, 3, strides=(2,2), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv3 = Conv2D(256, 3, strides=(2,2), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    outputs = Conv2D(1, 3, activation='sigmoid', padding='same')(conv3)
    
    model = Model(inputs, outputs)
    return model

# Define the GAN network
def build_gan(generator, discriminator, input_shape):
    discriminator.trainable = False
    inputs = Input(shape=input_shape)
    generated_spectrogram = generator(inputs)
    outputs = discriminator(generated_spectrogram)
    
    model = Model(inputs, outputs)
    return model

# Define loss function and optimizer
def binary_cross_entropy_loss(y_true, y_pred):
    return BinaryCrossentropy()(y_true, y_pred)

generator = build_generator(input_shape=(256, 256, 1))

# Define the optimizer
optimizer = Adam(lr=0.0002, beta_1=0.5)

# Compile the discriminator network
discriminator = build_discriminator(input_shape=(512, 512, 1))
discriminator.compile(loss=binary_cross_entropy_loss, optimizer=optimizer, metrics=['accuracy'])

# Compile the GAN network
generator = build_generator(input_shape=(256, 256, 1))
discriminator.trainable = False
gan = build_gan(generator, discriminator, input_shape=(256, 256, 1))
gan.compile(loss=binary_cross_entropy_loss, optimizer=optimizer)

# Load the spectrogram data
spectrograms = np.load('spectrograms.npy')

# Normalize the data between -1 and 1
spectrograms = (spectrograms / 127.5) - 1.

# Train the GAN network
batch_size = 32
epochs = 1000

for epoch in range(epochs):
    # Select a random batch of spectrograms
    idx = np.random.randint(0, spectrograms.shape[0], batch_size)
    real_spectrograms = spectrograms[idx]
    
    # Generate a batch of high-quality spectrograms
    noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))
    generated_spectrograms = generator.predict(noise)
    
    # Train the discriminator network
    d_loss_real = discriminator.train_on_batch(real_spectrograms, np.ones((batch_size, 512, 512, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_spectrograms, np.zeros((batch_size, 512, 512, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the GAN network
    noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 512, 512, 1)))
    
    # Print the progress
    print('Epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
    # Save the generated spectrograms
    if epoch % 100 == 0:
        generated_spectrograms = 0.5 * generator.predict(np.random.normal(0, 1, (9, 256, 256, 1))) + 0.5
        generated_spectrograms = np.squeeze(generated_spectrograms)
        np.save('generated_spectrograms_epoch_%d.npy' % epoch, generated_spectrograms)
