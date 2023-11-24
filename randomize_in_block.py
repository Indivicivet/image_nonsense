import argparse
from pathlib import Path

from PIL import Image
import numpy as np


def shuffle(arr):
    pixels = arr.reshape((-1, 3))
    return np.random.default_rng().permuted(pixels, axis=0).reshape(arr.shape)


def randomize_in_block(img, n_blocks=48):
    n_blocks_y = int(img.height / img.width * n_blocks)
    arr = np.array(img.convert("RGB"))
    return Image.fromarray(np.vstack([
        np.hstack([
            shuffle(block)
            for block in np.array_split(v_blocks, n_blocks, axis=1)
        ])
        for v_blocks in np.array_split(arr, n_blocks_y, axis=0)
    ]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    result = randomize_in_block(Image.open(args.input_path))
    p = Path(args.input_path)
    result.save(p.parent / f"{p.stem}_shuffled.png")
