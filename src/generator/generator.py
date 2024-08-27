from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, Reshape, UpSampling2D

__all__ = ["build_generator"]


def build_generator() -> Model:
    """
    Build the generator model

    Returns:
        Model: The generator model
    """
    # Instantiate sequential model
    generator = Sequential()

    # Dense layer with input dimension 128 x 7 x 7
    generator.add(Dense(units=128 * 7 * 7, activation="relu", input_dim=100))

    # Reshape the image dimensions to 7 x 7 x 128
    generator.add(Reshape(target_shape=(7, 7, 128)))

    # Up sample the image to 14 x 14
    generator.add(UpSampling2D())

    # First Convolutional layer with Batch Normalization and image up sampling to 28 x 28
    generator.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())

    # Second Convolutional layer with Batch Normalization, no up sampling because the image
    # is already 28 x 28 - equal to MNIST dataset image size
    generator.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))

    # Third Convolutional layer with number of filters=1
    generator.add(Conv2D(filters=1, kernel_size=3, padding="same", activation="relu"))

    # Print the model summary
    generator.summary()

    # Input noise of length 100, 100 chosen to create a simple network
    noise = Input(shape=(100,))

    # Run the generator to create the fake image
    fake_image = generator(noise)

    # Return a model that takes as input the noise and outputs the fake image
    return Model(inputs=noise, output=fake_image)
