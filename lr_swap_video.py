"""
swap left and right sides of a video
you need ffmpeg (e.g. choco install ffmpeg-full)
and (if applicable for your video) an appropriate version of openh264:
https://github.com/cisco/openh264/releases?page=2
which at time of writing is 1.8.0
"""


import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def swap_left_right(input_path: Union[str, Path]) -> Path:
    """
    Swap the left and right halves of each frame in the video.
    """
    input_path = Path(input_path)
    output_path = input_path.with_stem(f"{input_path.stem}_LRswap")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")
    out = cv2.VideoWriter(
        str(output_path),
        int(cap.get(cv2.CAP_PROP_FOURCC)),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hw = frame.shape[1] // 2
        swapped = np.concatenate((frame[:, hw:], frame[:, :hw]), axis=1)
        out.write(swapped)
    cap.release()
    out.release()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swap left and right halves of a video."
    )
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to the input video file.",
    )
    args = parser.parse_args()
    result = swap_left_right(args.input_video)
    print(f"Swapped video saved to: {result}")
