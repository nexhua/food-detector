import tensorflow as tf
from keras import models
from keras import layers

from model import Model


def get_generators(model: Model):
    if(model.name == 'simple_v1'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=model.validation_split)
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        return data_generator, test_generator
    else:
        return None, None


def get_model(model: Model):
    if(model.name == 'simple_v1'):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(256, 256, 3)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(101, activation='softmax'))

        return model
    else:
        return model
