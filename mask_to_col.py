"""
Uses CLIPSeg to segment regions.
"""

from pathlib import Path
import traceback

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torchvision import transforms as tv_tf
from torchvision.models.segmentation import deeplabv3_resnet101
from skimage import filters


class ModelClipSeg:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.to(self.device).eval()

    def mask_to_col(
        self,
        input_path: Path,
        prompt: str,
        threshold: float = -1.5,
        threshold_fade: float = 1.5,  # fade to not masked
        threshold_tweak_factor: float = 1,  # add ratio of max to normalize
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
        logits = (
            torch.nn.functional.interpolate(
                outputs.logits.unsqueeze(0),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0, :]
            .cpu()
            .numpy()
        )
        # map [threshold - threshold_fade, threshold] to [0, 1]
        threshold += logits.max() * threshold_tweak_factor
        mask = np.clip(1 + (logits - threshold) / threshold_fade, 0, 1)[..., np.newaxis]
        arr = (1 - mask) * np.array(image) + mask * col
        result = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        output_path = (
            input_path.parent / "clipseg" / prompt / f"{input_path.stem}_modified.png"
        )
        output_path.parent.mkdir(exist_ok=True, parents=True)
        result.save(output_path)
        print(f"Saved modified image to {output_path}")


class ModelHybrid:
    def __init__(self):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CLIPSeg for fine-grained "prompt" segmentation
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.to(self.device).eval()
        # DeepLabV3 for person-level segmentation (Pascal VOC pretrained)
        self.segmenter = deeplabv3_resnet101(pretrained=True, progress=True)
        self.segmenter.to(self.device).eval()
        # Preprocessing for segmentation
        self.seg_transform = tv_tf.Compose(
            [
                tv_tf.ToTensor(),
                tv_tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def mask_to_col(
        self,
        input_path: Path,
        prompt: str,
        threshold: float = None,
        threshold_fade: float = 1.5,
        threshold_tweak_factor: float = 3,
        col: tuple = (0, 0, 0),
    ):
        """
        Loads an image, uses an ensemble of DeepLabV3 (person-level)
        and CLIPSeg (prompt-level) segmentation to detect regions matching `prompt`,
        thresholds and refines the mask,
        then fades into `col` and saves the result.

        if `threshold` is specified, `threshold tweak factor` bases on max
        otherwise this is simply added to the otsu guess
        """
        image = Image.open(input_path).convert("RGB")
        # person segmentation DeepLabV3
        seg_input = self.seg_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Class index 15 == "person" in Pascal VOC
            person_logits = self.segmenter(seg_input)["out"][0, 15, ...]
        print(person_logits.shape)
        person_mask = (
            torch.nn.functional.interpolate(
                person_logits.unsqueeze(0).unsqueeze(0),  # interp (n, m, h, w)
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0, ...]
            .cpu()
            .numpy()
        )
        print(person_mask.shape)

        # prompt segmentation (CLIPSeg)
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            clip_out = self.model(**inputs).logits
        clip_logits = (
            torch.nn.functional.interpolate(
                clip_out.unsqueeze(0),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if threshold is None:
            flat = clip_logits.ravel().astype(np.float32)
            threshold = filters.threshold_otsu(flat) + threshold_tweak_factor
        else:
            threshold = threshold + clip_logits.max() * threshold_tweak_factor

        # combine person and prompt masks
        combined = (
            np.clip(1 + (clip_logits - threshold) / threshold_fade, 0, 1)
        ) * (person_mask > 0).astype(np.float32)

        # morphological smoothing + gaussian blur
        bin_mask = (combined > 0.1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
        mask_blur = cv2.GaussianBlur(clean.astype(np.float32), (15, 15), 0)[
            ..., np.newaxis
        ]
        col_arr = np.array(col, dtype=np.float32).reshape(1, 1, 3)
        # todo :: also move shared save stuff to some general interface
        result_arr = np.array(image) * (1 - mask_blur) + col_arr * mask_blur
        result = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8))
        output_path = (
            input_path.parent
            / "hybridseg"
            / prompt
            / f"{input_path.stem}_modified.png"
        )
        output_path.parent.mkdir(exist_ok=True, parents=True)
        result.save(output_path)
        print(f"Saved modified image to {output_path}")


if __name__ == "__main__":
    model = ModelClipSeg()  # ModelHybrid() is slower and... idk if better
    in_path = Path(input("enter path: ").strip('"'))
    while in_path:
        pmt = input("enter prompt: ")
        if in_path.is_dir():
            for p in in_path.glob("*.*"):
                try:
                    model.mask_to_col(input_path=p, prompt=pmt)
                except Exception as e:
                    print(
                        f"got {''.join(
                            traceback.TracebackException.from_exception(e).format()
                        )}"
                        f" but continuing"
                    )
        else:
            model.mask_to_col(input_path=in_path, prompt=pmt)
        in_path = Path(input("enter path: "))
