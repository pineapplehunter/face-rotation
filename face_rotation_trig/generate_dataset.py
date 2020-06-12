from PIL import Image
from pathlib import Path
import cv2
import argparse
import logging
import sys
import random
import shutil
import numpy as np

# input image dimensions
img_rows, img_cols = 32, 32

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def trim_faces_opencv(file_path: Path, casc_path: str, min_size:int) -> [Image]:
    logging.debug("start process on %s", file_path)
    faceCascade = cv2.CascadeClassifier(casc_path)

    image = Image.open(file_path)
    image.thumbnail((500,500))
    image = pil2cv(image)
    # image = cv2.resize(image, (500,500), interpolation = cv2.INTER_LINER)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    output = []
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im = Image.open(file_path)
        im.thumbnail((500,500))
        im = im.convert("L")
        # im.thumbnail((500, 500))

        margin = w / 2 * 0.42

        if (
            x - margin < 0
            or y - margin < 0
            or x + w + margin > im.width
            or y + h + margin > im.height
        ):
            # print("size skip")
            continue

        im = im.crop((x - margin, y - margin, x + w + margin, y + h + margin))

        output.append(im)
    return output


def trim_faces_ssd_keras(picture_file: Path):
    raise NotImplementedError()


def main():
    logging.basicConfig(filename="generate_dataset.log", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="directory to load the raw data including photos",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="directory to output the generated datasets",
        required=True,
    )
    parser.add_argument(
        "--method", "-m", choices=["opencv", "ssd_keras"], default="opencv"
    )
    parser.add_argument(
        "--cascade_path", default="../haarcascade_frontalface_default.xml"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="log debug info")
    parser.add_argument(
        "--test-split",
        "-t",
        action="store_true",
        help="if you want to split data into train and test you can use this flag",
    )
    parser.add_argument(
        "--test-split-ratio",
        "-r",
        type=float,
        help="the ratio of train test split",
        default=0.1,
    )
    parser.add_argument("--min-size", "-s", type=int, help="minimum size of face", default=30)

    opts = parser.parse_args()

    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if opts.method == "ssd_keras":
        raise NotImplementedError

    input_path = Path(opts.input)
    logging.debug("input path = %s", input_path)
    output_path = Path(opts.output)
    logging.debug("output path = %s", output_path)
    assert input_path.is_dir()

    if output_path.exists() and not output_path.is_dir():
        logging.error("output path not dir")
        raise RuntimeError("output path not dir")
    if not output_path.exists():
        logging.info("creating directory at %s", output_path)
        output_path.mkdir()

    if (output_path / "train").exists() and not (output_path / "train").is_dir():
        logging.error("output/train not dir")
        raise RuntimeError("output/train not dir")
    if not (output_path / "train").exists():
        logging.info("creating directory at %s", (output_path / "train"))
        (output_path / "train").mkdir()

    try:
        logging.debug("reading index from output directory")
        with (output_path / "index.txt").open("r") as f:
            index = int(f.read())
    except IOError:
        logging.debug("using fallback index")
        index = 0
    logging.debug("index = %s", index)

    logging.debug("using method %s", opts.method)

    for picture_file in input_path.iterdir():
        if not picture_file.is_file():
            continue
        if opts.method == "opencv":
            faces = trim_faces_opencv(picture_file, casc_path=opts.cascade_path, min_size=opts.min_size)
        elif opts.method == "ssd_keras":
            faces = trim_faces_ssd_keras(picture_file)

        try:
            faces
        except NameError:
            faces = []

        for face in faces:
            face.save(output_path / "train" / f"image{index}.jpg")
            index += 1
    if opts.test_split:
        if (output_path / "test").exists() and not (output_path / "test").is_dir():
            logging.error("output/test not dir")
            raise RuntimeError("output/test not dir")
        if not (output_path / "test").exists():
            logging.info("creating directory at %s", (output_path / "test"))
            (output_path / "test").mkdir()

        train_list = [p for p in (output_path / "train").iterdir()]
        random.shuffle(train_list)
        test_list = train_list[0 : int(len(train_list) * opts.test_split_ratio)]
        for file in test_list:
            shutil.move(str(file), str(output_path / "test"))
    with (output_path / "index.txt").open("w") as f:
        f.write(f"{index}")


if __name__ == "__main__":
    main()

