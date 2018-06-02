from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5,
                            subsample=(2, 2),
                            border_mode='same',
                            input_shape=(1, 28, 28)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model