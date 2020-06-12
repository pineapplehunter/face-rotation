from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

from PIL import Image
import numpy as np
import os
import uuid
from datetime import datetime
from pathlib import Path

from load_dataset import TrainingData, img_rows, img_cols

import argparse

sess_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

np.random.seed(0)


def create_model():
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(img_rows, img_cols, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dense(2))

    model.summary()
    return model


# with open("model.json", "w") as f:
#     model_json = model.to_json()
#     f.write(model_json)


def trig_mse(_y_true, y_pred):
    sspcs = tf.add(tf.square(y_pred[:, 0]), tf.square(y_pred[:, 1]))
    squared_difference = tf.square(1 - sspcs)
    return tf.reduce_mean(squared_difference, axis=-1)


def angle_diff(y_true, y_pred):
    true_angle = tf.atan2(y_true[:, 0], y_true[:, 1]) * 360 / (2 * np.pi)
    pred_angle = tf.atan2(y_pred[:, 0], y_pred[:, 1]) * 360 / (2 * np.pi)

    diff = tf.abs(tf.subtract(true_angle, pred_angle))
    diff = tf.minimum(diff, 360 - diff)
    return tf.reduce_mean(diff, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train dataset directory", required=True)
    parser.add_argument("--test", help="test dataset directory", required=True)
    parser.add_argument("--logdir", help="tensorboard log directory", default="logs")
    opts = parser.parse_args()

    model = create_model()
    model.compile(loss="mse", optimizer="adam", metrics=[trig_mse, angle_diff])

    checkpoint = ModelCheckpoint(
        "training/model." + sess_id + ".{epoch:02d}.hdf5",
        monitor="loss",
        verbose=1,
        save_best_only=False,
        mode="auto",
        period=5,
    )
    tensorboard = TensorBoard(log_dir=opts.logdir + "/" + sess_id)

    train_path = Path(opts.train)
    test_path = Path(opts.test)

    gen = TrainingData(train_path, batch_size=128).generator()
    vgen = TrainingData(test_path, batch_size=128).generator()

    sample_x, sample_y = gen.__next__()

    print("X shape = {}".format(sample_x.shape))
    print("Y shape = {}".format(sample_y.shape))

    model.fit(
        gen,
        steps_per_epoch=128,
        epochs=100,
        validation_data=vgen,
        validation_steps=16,
        shuffle=True,
        callbacks=[checkpoint, tensorboard],
    )

    model.save("model.h5")


if __name__ == "__main__":
    main()
