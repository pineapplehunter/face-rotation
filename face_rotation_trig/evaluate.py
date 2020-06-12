from tensorflow.python.keras.models import Sequential, model_from_json, load_model
import tensorflow as tf

from PIL import Image
import numpy as np
import sys
import argparse

import matplotlib.pyplot as plt

from utils.datasetgen import TrainingData, img_rows, img_cols

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="model.h5")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--data_dir")

options = parser.parse_args()

print(f"seed={options.seed}")

np.random.seed(options.seed)


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


model = load_model(options.model, custom_objects={"trig_mse": trig_mse, "angle_diff": angle_diff})

if options.data_dir:
    data = TrainingData(6 * 2, same_face_generated=1, useall=True, data_dir=options.data_dir).generator()

else:
    data = TrainingData(6 * 2, same_face_generated=1, train=False).generator()

X_test, Y_test = data.__next__()
for i in range(options.seed):
    X_test, Y_test = data.__next__()

print(X_test.shape)

y = model.predict(X_test, verbose=1)
print(f"answer     = {Y_test}")
print(f"prediction = {y}")

w = 10
h = 10
fig = plt.figure(figsize=(w, h))
columns = 6
rows = 4

# ax enables access to manipulate each of subplots
ax = []

for i in range((columns * rows) // 2):
    im = X_test[i].reshape((img_cols, img_rows))
    ax.append(fig.add_subplot(rows, columns, i + 1))

    sinv = Y_test[i][0]
    cosv = Y_test[i][1]

    angle_rad = np.arctan2(sinv, cosv)
    angle = angle_rad / (2 * np.pi) * 360

    ax[-1].set_title("label:" + str(i) + "\n" + "angle:" + str(int(angle)))  # set title
    plt.imshow(im)

for k in range((columns * rows) // 2):
    i = k + (columns * rows) // 2
    im = Image.fromarray(X_test[k].reshape((img_cols, img_rows)))

    sinv = y[k][0]
    cosv = y[k][1]

    angle_rad = np.arctan2(sinv, cosv)
    angle = angle_rad / (2 * np.pi) * 360

    im = im.rotate(-angle)
    im = np.array(im)
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("label:" + str(k) + "\n" + "angle:" + str(int(angle)))  # set title
    plt.imshow(im)

plt.savefig("output.png")
