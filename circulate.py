import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def circulate(im, width_pix=40, cell_size=None, preview_pixellated=True):
    if isinstance(im, Image.Image):
        im = np.array(im.convert("RGB"))
    intensity = im ** 2.2
    if cell_size is None:
        cell_size = im.shape[1] // width_pix
    height_pix = (im.shape[0] * width_pix) // im.shape[1]
    downscaled_intensity = cv2.resize(
        intensity,
        (width_pix, height_pix),
        interpolation=cv2.INTER_AREA,
    )
    if preview_pixellated:
        Image.fromarray(
            cv2.resize(
                downscaled_intensity,
                im.shape[:2][::-1],
                interpolation=cv2.INTER_AREA,
            )
        ).show()
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, cell_size),
        np.linspace(-1, 1, cell_size),
    )
    rr2 = xx ** 2 + yy ** 2
    result_01 = np.full(
        shape=(height_pix * cell_size, width_pix * cell_size, 3),
        fill_value=1,
    )
    for j in range(height_pix):
        for i in range(width_pix):
            rgb_01 = downscaled_intensity[j, i] / 255
            for c_indices, r2 in zip(
                [(1, 2), (0, 2), (0, 1)],
                1 - rgb_01,
            ):
                circ = rr2 <= r2
                for c_idx in c_indices:
                    result_01[
                        j * cell_size:(j + 1) * cell_size,
                        i * cell_size:(i + 1) * cell_size,
                        c_idx
                    ] -= circ
    return Image.fromarray((result_01 * 255).clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    p = Path(args.input_path)
    result = circulate(Image.open(p))
    result.save(p.parent / f"{p.stem}_circulated.png")
