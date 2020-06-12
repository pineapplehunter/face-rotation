from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import sys

img_rows, img_cols = 32, 32


class TrainingData:
    def __init__(
        self, dataset_dir, batch_size=128,
    ):
        self.X = []
        self.Y = []
        self.batch_size = batch_size
        self.iteration = 0
        self.dataset_dir = Path(dataset_dir)

    def reset(self):
        self.X = []
        self.Y = []
        self.iteration = 0

    def generator(self):
        while True:
            for f in self.dataset_dir.iterdir():
                if not f.is_file():
                    continue

                im = Image.open(f)

                margin = (im.width - im.width / (np.sqrt(2))) / 2

                angle = np.random.rand()

                imc = im.copy()
                imc = imc.rotate(angle * 360)
                imc = imc.crop((margin, margin, im.width - margin, im.height - margin))
                imc = imc.resize((img_rows, img_cols))

                self.iteration += 1

                x = np.array(imc, dtype=np.float32)
                x = x.reshape((img_rows, img_cols, 1))
                x = x / 255

                self.X.append(x)
                self.Y.append([np.sin(angle * 2 * np.pi), np.cos(angle * 2 * np.pi)])

                if self.iteration >= self.batch_size:
                    x = np.array(self.X, dtype="float32")
                    y = np.array(self.Y, dtype="float32")
                    self.reset()
                    yield x, y
            # print(f"loop! {self.iteration}")


def main():
    logging.basicConfig(filename="load_dataset.log", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="dataset directory", required=True)
    parser.add_argument("--seed", "-s", help="random seed [0]", default=0, type=int)
    opts = parser.parse_args()

    np.random.seed(0)

    gen = TrainingData(opts.input, 10).generator()
    X, Y = gen.__next__()

    print(
        "X shape = {}, min = {:01.3f}, max = {:01.3f}, type = {}".format(
            X.shape, X.min(), X.max(), X.dtype
        )
    )
    print(
        "Y shape = {}, min = {:01.3f}, max = {:01.3f}, type = {}".format(
            Y.shape, Y.min(), Y.max(), Y.dtype
        )
    )

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

        sinv = Y[i][0]
        cosv = Y[i][1]

        angle = np.arctan2(sinv, cosv) / (2 * np.pi) * 360

        ax[-1].set_title("angle:" + str(int(angle)))  # set title
        plt.imshow(im)

    plt.savefig("dataset.png")


if __name__ == "__main__":
    main()
