from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def process_paths(strs):
    paths = [Path(s) for s in strs]

    def with_tag(tag):
        return paths[0].parent / f"{paths[0].stem}_{tag}.png"

    arrs = np.array([np.array(Image.open(p).convert("RGB")) for p in paths])
    Image.fromarray(np.mean(arrs, axis=0).astype(np.uint8)).save(with_tag("mean"))
    Image.fromarray(np.median(arrs, axis=0).astype(np.uint8)).save(with_tag("median"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    process_paths(args.paths)
