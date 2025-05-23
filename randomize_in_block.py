import argparse
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm


def shuffle(arr, together):
    """
    `together` dictates if colours are kept together, or if rgb can split up
    """
    pixels = arr.reshape((-1, 3))
    shuffled = (
        pixels[np.random.default_rng().random(pixels.shape[0]).argsort()]
        if together
        else np.random.default_rng().permuted(pixels, axis=0)
    )
    return shuffled.reshape(arr.shape)


def randomize_in_block(img, n_blocks=48, max_res=720, together=True):
    if img.width > max_res:
        img = img.resize((max_res, int(img.height / img.width * max_res)))
    n_blocks_y = int(img.height / img.width * n_blocks)
    arr = np.array(img.convert("RGB"))
    return Image.fromarray(np.vstack([
        np.hstack([
            shuffle(block, together=True)
            for block in np.array_split(v_blocks, n_blocks, axis=1)
        ])
        for v_blocks in np.array_split(arr, n_blocks_y, axis=0)
    ]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    p = Path(args.input_path)
    if p.is_dir():
        out_dir = p / "shuffled"
        files = list(p.rglob("*.*"))
        for file in tqdm(files):
            if "shuffled" in str(par.resolve()):
                continue
            try:
                result = randomize_in_block(Image.open(file))
                out_path = (out_dir / file.relative_to(p)).with_suffix(".png")
                out_path.parent.mkdir(exist_ok=True, parents=True)
                assert out_path != file
            except Exception as e:
                print(f"{file.relative_to(p)} processing FAILURE ({e})")
                continue
            print(f"{file.relative_to(p)} processing success")
            result.save(out_path)
    else:
        result = randomize_in_block(Image.open(p))
        result.save(p.parent / f"{p.stem}_shuffled.png")
