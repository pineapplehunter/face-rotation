from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

from PIL import Image
import numpy as np
import os

from datasetgen import TrainingData

img_rows, img_cols = 28, 28

np.random.seed(0)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

with open("model.json", "w") as f:
    model_json = model.to_json()
    f.write(model_json)


def angle_loss(y, y_pred):
    m = tf.abs(y - y_pred)
    return tf.minimum(m ** 2, (1 - m) ** 2)


model.compile(loss=angle_loss,
              optimizer="adam")

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.fit(TrainingData(128).generator(), steps_per_epoch=1280, epochs=10)

model.save("model.h5")
