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
    elif(model.name == 'cnn_v1'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=model.validation_split)
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        return data_generator, test_generator
    elif(model.name == 'cnn_v2'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=model.validation_split)
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        return data_generator, test_generator
    elif(model.name == 'cnn_v3'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=model.validation_split)
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        return data_generator, test_generator
    elif(model.name == 'fine_tuned_v1'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=model.validation_split)
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        return data_generator, test_generator
    elif(model.name == 'fine_tuned_v2'):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, rotation_range=20, shear_range=0.2, zoom_range=0.2, height_shift_range=0.1,
            width_shift_range=0.1, validation_split=model.validation_split)
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
    elif(model.name == 'cnn_v1'):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(256, 256, 3)))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation="relu"))
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(101, activation='softmax'))

        return model
    elif(model.name == 'cnn_v2'):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(256, 256, 3)))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(101, activation='softmax'))

        return model
    elif(model.name == 'cnn_v3'):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(256, 256, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(101, activation='softmax'))

        return model
    elif(model.name == 'fine_tuned_v1'):
        model = models.Sequential()

        res_net_50 = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=(256, 256, 3))

        res_net_50.trainable = False
        model.add(res_net_50)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(101, activation='softmax'))

        return model
    elif(model.name == 'fine_tuned_v2'):
        model = models.Sequential()

        res_net_50 = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=(256, 256, 3))

        res_net_50.trainable = False
        model.add(res_net_50)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(101, activation='softmax'))

        return model
    else:
        return None
