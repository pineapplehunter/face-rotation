from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

from PIL import Image
import numpy as np
import os

import datasetgen

TrainingData = datasetgen.TrainingData
img_rows = datasetgen.img_rows
img_cols = datasetgen.img_cols

np.random.seed(0)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
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

gen = TrainingData(256,128).generator()
vgen = TrainingData(256,16,train=False).generator()

sample_x, sample_y = gen.__next__()

print("X shape = {}".format(sample_x.shape))
print("Y shape = {}".format(sample_y.shape))

model.fit(gen,steps_per_epoch=16, epochs=20 ,validation_data=vgen, validation_steps=1,shuffle=True)

model.save("model.h5")
