import argparse
from pathlib import Path

from PIL import Image
import numpy as np


def shuffle(arr):
    pixels = arr.reshape((-1, 3))
    return np.random.default_rng().permuted(pixels, axis=0).reshape(arr.shape)


def randomize_in_block(img, n_blocks=48, max_res=720):
    if img.width > max_res:
        img = img.resize((max_res, int(img.height / img.width * max_res)))
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
    p = Path(args.input_path)
    for file in (
        p.rglob("*.*")
        if p.is_dir()
        else [p]
    ):
        try:
            result = randomize_in_block(Image.open(file))
            print(f"{file.name} reading success")
        except Exception:
            print(f"{file.name} reading FAILURE")
            continue
        result.save(file.parent / f"{file.stem}_shuffled.png")
