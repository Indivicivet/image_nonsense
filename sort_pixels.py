import argparse

from PIL import Image
import numpy as np


def sort_pixels(img):
    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape((-1, 3))
    return Image.fromarray(np.sort(pixels, axis=0).reshape(arr.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    sort_pixels(Image.open(args.input_path)).show()
