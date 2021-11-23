import keras
import keras.backend as K
from keras.layers import Lambda, Conv2DTranspose, add, Concatenate
from keras.layers.core import Activation

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation=None):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding,activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x