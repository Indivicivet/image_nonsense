from pathlib import Path
from PIL import Image

MARGIN_CENTER = 0.02  # total gap in the middle


def crop_white_borders(img: Image.Image, tolerance: int = 10) -> Image.Image:
    """Crops out near-white borders from the image."""
    gray = img.convert("L")
    mask = gray.point(lambda p: 0 if p > (255 - tolerance) else 255)
    bbox = mask.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def split_image_4way(img: Image.Image) -> list[Image.Image]:
    """Splits an image into 4 quadrants with margins."""
    img = crop_white_borders(img)
    w, h = img.size

    margin_center = int(w * MARGIN_CENTER)

    # Calculate boundaries
    # Quad 0: Top-Left
    q0 = (
        0,
        0,
        w // 2 - margin_center // 2,
        h // 2 - margin_center // 2,
    )
    # Quad 1: Top-Right
    q1 = (
        w // 2 + margin_center // 2,
        0,
        w,
        h // 2 - margin_center // 2,
    )
    # Quad 2: Bottom-Left
    q2 = (
        0,
        h // 2 + margin_center // 2,
        w // 2 - margin_center // 2,
        h,
    )
    # Quad 3: Bottom-Right
    q3 = (
        w // 2 + margin_center // 2,
        h // 2 + margin_center // 2,
        w,
        h,
    )

    return [img.crop(box) for box in (q0, q1, q2, q3)]


if __name__ == "__main__":
    input_dir = Path("4way_input")
    output_dir = Path("4way_output")

    if not input_dir.exists():
        print(f"Input directory '{input_dir}' does not exist.")
        exit()

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            try:
                img = Image.open(file_path)
                # Ensure image is in a mode that supports saving to all formats
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")

                quadrants = split_image_4way(img)

                for i, quad in enumerate(quadrants):
                    out_name = f"{file_path.stem}_{i}{file_path.suffix}"
                    out_path = output_dir / out_name
                    quad.save(out_path)

                print(f"Processed: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
