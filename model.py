"""
The training was tested using Google Colab GPU notebook
with %tensorflow_version 1.x

The version compatible with this artifact saved may be as following

Keras==2.3.1
tensorflow==1.15.2

"""
import argparse
import os
import random
from math import ceil

import imageio
import numpy as np
import pandas as pd
import pendulum
import sklearn
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, ELU, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

random.seed(10)


def generator(samples, batch_size=128, img_path='./training/IMG/'):
    """
    Generator for model training
    :param samples:
    :param batch_size:
    :param img_path:
    :return:
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = imageio.imread(img_path + batch_sample[0])
                left_image = imageio.imread(img_path + batch_sample[1])
                right_image = imageio.imread(img_path + batch_sample[2])
                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2  # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # center
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -center_angle

                images.append(center_image)
                angles.append(center_angle)

                images.append(image_flipped)
                angles.append(measurement_flipped)

                # left
                image_flipped = np.fliplr(left_image)
                measurement_flipped = -left_angle

                images.append(left_image)
                angles.append(left_angle)

                images.append(image_flipped)
                angles.append(measurement_flipped)

                # right
                image_flipped = np.fliplr(right_image)
                measurement_flipped = -right_angle

                images.append(right_image)
                angles.append(right_angle)

                images.append(image_flipped)
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)
            xy = sklearn.utils.shuffle(X, y)
            yield xy[0], xy[1]


def create_model():
    """
    Function to create model neural network architecture
    The architecture is based on loosely modified Nvidia research
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    :return:
    """
    model = Sequential()
    # the input image is 320x160x3
    row, col, ch = 160, 320, 3
    model.add(
        Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch))
    )
    # Crop the top by 60x and bottom 20x
    model.add(Cropping2D(cropping=((60, 20), (0, 0))))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(units=100))
    model.add(ELU())

    model.add(Dropout(0.20))

    model.add(Dense(units=50))
    model.add(ELU())

    model.add(Dense(units=10))
    model.add(ELU())

    model.add(Dense(units=1))
    # optimizer = Adam(lr=0.0001)
    # optimizer = Adam(lr=0.001)
    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Path')
    parser.add_argument(
        'training_folder',
        type=str,
        help='Path to training folder, inside should have csv and IMG folder'
    )
    parser.add_argument(
        'log_file',
        type=str,
        nargs='?',
        default='driving_log.csv',
        help="The name of the driving log in csv formatted of 'center_img','left_img', 'right_img', "
             "'steering_angle', 'throttle', 'break', 'speed' "
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--es-patience',
        type=int,
        default=1,
        help='Early Stopping patience. This is the number of epoch to wait '
             'before stop when the loss_val no longer improving'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='Epoch parameter for the training'
    )

    parser.add_argument(
        '--artifact-dst',
        type=str,
        default="model.h5",
        help='The artifact or model saved filename, default is current folder under model.h5, can '
             'specify to mounted folder to automatically make it persistent'
    )

    args = parser.parse_args()

    device_name = tf.test.gpu_device_name()
    if device_name:
      print('Found GPU at: {}'.format(device_name))
    else:
      print("Cannot find any GPU")

    training_df = pd.read_csv(os.path.join(args.training_folder, args.log_file), header=None, names=[
        'center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed'])

    # remove the hardcoded IMG src location
    training_df['center_img'] = training_df['center_img'].str.split("/").str[-1]
    training_df['left_img'] = training_df['left_img'].str.split("/").str[-1]
    training_df['right_img'] = training_df['right_img'].str.split("/").str[-1]

    # In my data collection, 0 angle is too skewed
    # Resample it to only 500 to make it more evenly distributed
    # zero_angle = training_df.loc[training_df['steering_angle'] == 0]
    # non_zero = training_df.loc[training_df['steering_angle'] != 0]

    # training_df = non_zero
    # training_df = training_df.append(zero_angle.sample(500))
    # training_df = training_df.reset_index(drop=True)
    # del zero_angle
    # del non_zero

    # split the train and validation set
    train_df, validation_df = train_test_split(training_df, test_size=0.2, random_state=1)

    model = create_model()

    start = pendulum.now()
    batch_size = args.batch
    train_generator = generator(train_df.values, batch_size=batch_size)
    validation_generator = generator(validation_df.values, batch_size=batch_size)
    # simple early stopping
    # # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.es_patience)
    mc = ModelCheckpoint(args.artifact_dst, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=ceil(len(train_df) / batch_size),
                                  validation_data=validation_generator,
                                  validation_steps=ceil(len(validation_df) / batch_size),
                                  epochs=args.epoch, verbose=1, callbacks=[es, mc])
    end = pendulum.now()
    print("The whole training took", (end - start).in_words())
