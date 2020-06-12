import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2

dataset_dir = Path(os.environ["DATASET_DIR"]) / "img_align_celeba"
casc_path = str("../haarcascade_frontalface_default.xml")
# print(casc_path)

# input image dimensions
img_rows, img_cols = 32, 32


class TrainingData:
    def __init__(self, batch_size=128,same_face_generated=20,train=True):
        self.X = []
        self.Y = []
        self.batch_size = batch_size
        self.same_face_generated = same_face_generated
        self.iteration = 0
        self.train = train

    def reset(self):
        self.X = []
        self.Y = []
        self.iteration = 0

    def generator(self):
        while True:
            for f in dataset_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name.startswith("00"):
                    if self.train:
                        continue
                else:
                    if not self.train:
                        continue

                faceCascade = cv2.CascadeClassifier(casc_path)

                image = cv2.imread(str(f))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )

                # print("Found {} faces!".format(len(faces)))

                if len(faces) == 0:
                    continue

                for (x, y, w, h) in faces:
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    im = Image.open(f)
                    im = im.convert('L')

                    margin = w / 2 * .42

                    if x - margin < 0 or y - margin < 0 or x + w + margin > im.width or y + h + margin > im.height:
                        # print("size skip")
                        continue

                    im = im.crop((x - margin, y - margin, x + w + margin, y + h + margin))

                    for _ in range(self.same_face_generated):
                        # print(self.iteration)

                        angle = np.random.random()

                        imc = im.copy()
                        imc = imc.rotate(angle * 360)
                        imc = imc.crop((margin, margin, im.width - margin, im.height - margin))
                        imc = imc.resize((img_rows, img_cols))

                        self.iteration += 1

                        x = np.array(imc, dtype=np.float32)
                        x = x.reshape((img_rows, img_cols, 1))
                        x = x / 255

                        self.X.append(x)
                        self.Y.append([angle])

                        if self.iteration >= self.batch_size:
                            x = np.array(self.X, dtype="float32")
                            y = np.array(self.Y, dtype="float32")
                            self.reset()
                            yield x, y


if __name__ == "__main__":
    np.random.seed(0)

    gen = TrainingData(10,same_face_generated=2).generator()
    X, Y = gen.__next__()

    print("X shape = {}, min = {:01.3f}, max = {:01.3f}, type = {}".format(X.shape, X.min(), X.max(), X.dtype))
    print("Y shape = {}, min = {:01.3f}, max = {:01.3f}, type = {}".format(Y.shape, Y.min(), Y.max(), Y.dtype))

    w = 10
    h = 5
    fig = plt.figure(figsize=(w, h))
    columns = 5
    rows = 2

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range((columns * rows)):
        im = X[i].reshape((img_rows, img_cols))
        ax.append(fig.add_subplot(rows, columns, i + 1))

        angle = Y[i][0] * 360

        # angle = np.arctan2(sinv, cosv) / (2 * np.pi) * 360

        ax[-1].set_title("angle:" + str(int(angle)))  # set title
        plt.imshow(im)

    plt.savefig("dataset.png")
