"""
Uses CLIPSeg to segment regions.
"""


from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.to(self.device).eval()

    def mask_to_col(
        self,
        input_path: Path,
        prompt: str,
        threshold: float = -2,
        threshold_fade: float = 1,  # fade to not masked
        col: tuple = (0, 0, 0),
    ):
        """
        Loads an image, uses CLIPSeg to segment the regions matching `prompt`,
        thresholds the mask, paints those pixels black, and saves the result.
        """
        image = Image.open(input_path).convert("RGB")
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Note: image.height, image.width are (H, W),
        # matching PyTorch's expected order
        logits = torch.nn.functional.interpolate(
            outputs.logits.unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :].cpu().numpy()
        # map [threshold - threshold_fade, threshold] to [0, 1]
        mask = np.clip(1 + (logits - threshold) / threshold_fade, 0, 1)[..., np.newaxis]
        arr = (1 - mask) * np.array(image) + mask * col
        result = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        output_path = (
            input_path.parent
            / "clipseg"
            / prompt
            / f"{input_path.stem}_modified.png"
        )
        output_path.parent.mkdir(exist_ok=True, parents=True)
        result.save(output_path)
        print(f"Saved modified image to {output_path}")


if __name__ == "__main__":
    model = Model()
    in_path = Path(input("enter path: ").strip("\""))
    while in_path:
        pmt = input("enter prompt: ")
        if in_path.is_dir():
            for p in in_path.glob("*.*"):
                try:
                    model.mask_to_col(input_path=p, prompt=pmt)
                except Exception as e:
                    print(f"got {e} but continuing")
        else:
            model.mask_to_col(input_path=in_path, prompt=pmt)
        in_path = Path(input("enter path: "))
