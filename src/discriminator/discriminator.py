from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, Flatten, Dense, Input

__all__ = ["build_discriminator"]


def build_discriminator():
    # Instantiate sequential model
    discriminator = Sequential()

    # Convolutional layer
    discriminator.add(Conv2D(filters=32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))  # -> 14x14

    # LeakyRELU activation function
    discriminator.add(LeakyReLU(alpha=0.2))

    # Dropout layer with 25% dropout rate
    discriminator.add(Dropout(rate=0.25))

    # Second convolutional layer with Zero padding to change dimension from 7x7 to 8x8
    discriminator.add(Conv2D(filters=64, kernel_size=3, strides=2, padding="same"))  # -> 7x7
    discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))  # -> 8x8

    # Batch normalization for faster learning and higher accuracy
    discriminator.add(BatchNormalization(momentum=0.8))

    # Leaky RELU and Dropout
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(rate=0.25))

    # Third convolutional layer with Batch Normalization, Leaky RELU and Dropout
    discriminator.add(Conv2D(filters=128, kernel_size=3, strides=2, padding="same"))  # -> 4x4
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(rate=0.25))

    # Fourth convolutional layer with Batch Normalization, Leaky RELU and Dropout
    discriminator.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same"))  # -> 4x4
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(rate=0.25))

    # Flatten and Dense layers with sigmoid activation function
    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))

    # Sets the input image shape and run the discriminator model to get probability
    img = Input(shape=(28, 28, 1))
    probability = discriminator(img)

    # Return a model that takes as input the image and outputs the probability
    return Model(inputs=img, outputs=probability)
