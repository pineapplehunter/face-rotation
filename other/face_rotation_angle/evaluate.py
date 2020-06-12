from tensorflow.python.keras.models import Sequential, model_from_json, load_model
import tensorflow as tf

from PIL import Image
import numpy as np
import sys

import matplotlib.pyplot as plt

from datasetgen import TrainingData, img_rows, img_cols

try:
    seed = int(sys.argv[1])
except:
    seed = 0

print(f"seed={seed}")

np.random.seed(seed)


def angle_loss(y, y_pred):
    m = tf.abs(y - y_pred)
    return tf.minimum(m ** 2, (1 - m) ** 2)


model = load_model("model.h5", custom_objects={'angle_loss': angle_loss})

data = TrainingData(6 * 2, same_face_generated=1,train=False).generator()
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
    ax[-1].set_title("label:" + str(i) + "\n" + "angle:" + str(int(Y_test[i][0] * 360)))  # set title
    plt.imshow(im)

for k in range((columns * rows) // 2):
    i = k + (columns * rows) // 2
    im = Image.fromarray(X_test[k].reshape((img_cols, img_rows)))
    im = im.rotate(-y[k][0] * 360)
    im = np.array(im)
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("label:" + str(k) + "\n" + "angle:" + str(int(y[k][0] * 360)))  # set title
    plt.imshow(im)

plt.savefig("output.png")
