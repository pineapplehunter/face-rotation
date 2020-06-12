import numpy as np
from tensorflow.python.keras.datasets import mnist
from PIL import Image

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols))
# x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, 1))
# input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')


class TrainingData:
    def __init__(self, batch_size=128):
        self.X = []
        self.Y = []
        self.batch_size = batch_size
        self.iteration = 0

    def reset(self):
        self.X = []
        self.Y = []
        self.iteration = 0

    def generator(self):
        while True:
            for i in range(x_train.shape[0]):
                self.iteration += 1
                angle = np.random.random()
                im = Image.fromarray(x_train[i])
                im = im.rotate(angle * 360)
                x = np.array(im)
                x = x.reshape((img_rows, img_cols, 1))
                x = x / 255
                angle = np.array([np.sin(2 * angle * np.pi), np.cos(2 * angle * np.pi)])
                # print(x.shape, angle.shape)

                self.X.append(x)
                self.Y.append(angle)

                if self.iteration >= self.batch_size:
                    x = np.array(self.X)
                    y = np.array(self.Y)
                    self.reset()
                    yield x, y
