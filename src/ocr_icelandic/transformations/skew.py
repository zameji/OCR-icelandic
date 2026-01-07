import random
from typing import TypedDict

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


class SkewMeta(TypedDict):
    dx: float
    pad_x: int
    crop_box: tuple[int, int, int, int]
    cropped_size: tuple[int, int]
    paste_offset: tuple[int, int]
    canvas_max_side: int
    target_size: tuple[int, int]


def _skew_within_bounds(image: Image.Image, dx: float) -> tuple[Image.Image, SkewMeta]:
    width, height = image.size

    # Calculate the expanded width after skew
    max_shift = abs(dx * height)

    # Create large canvas with transparent background
    pad = int(max_shift)
    canvas_width = width + pad * 2
    canvas = Image.new("RGBA", (canvas_width, height), (0, 0, 0, 0))
    canvas.paste(image, (pad, 0), image if image.mode == "RGBA" else None)

    # Apply skew with transparent fill
    matrix = (1, dx, 0, 0, 1, 0)
    skewed = canvas.transform(
        canvas.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    # Crop to remove excess area introduced by skew
    if dx > 0:
        crop_box = (0, 0, canvas_width - int(dx * height), height)
        cropped = skewed.crop(crop_box)
    else:
        crop_box = (int(-dx * height), 0, canvas_width, height)
        cropped = skewed.crop(crop_box)

    # Paste centered on a rectangular canvas to scale back to original size
    canvas_max_side = max(cropped.width, cropped.height)
    canvas_for_resize = Image.new(
        "RGBA", (canvas_max_side, canvas_max_side), (0, 0, 0, 0)
    )
    left = canvas_max_side // 2 - cropped.width // 2
    top = canvas_max_side // 2 - cropped.height // 2
    canvas_for_resize.paste(
        cropped, (left, top), cropped if cropped.mode == "RGBA" else None
    )

    final_image = canvas_for_resize.resize((width, height), Image.Resampling.BICUBIC)

    skew_meta = {
        "dx": dx,
        "pad_x": pad,
        "crop_box": crop_box,
        "cropped_size": (cropped.width, cropped.height),
        "paste_offset": (left, top),
        "canvas_max_side": canvas_max_side,
        "target_size": (width, height),
    }

    return final_image, skew_meta


def _transform_paragraph_bboxes_for_skew(
    paragraph_bboxes: list[dict], meta: SkewMeta
) -> list[dict]:
    if not paragraph_bboxes:
        return []

    pad_x = meta["pad_x"]
    dx = meta["dx"]
    crop_left, crop_top, _, _ = meta["crop_box"]
    paste_left, paste_top = meta["paste_offset"]
    canvas_max_side = meta["canvas_max_side"]
    target_width, target_height = meta["target_size"]

    # Calculate actual resize scale factors
    resize_scale_x = target_width / canvas_max_side
    resize_scale_y = target_height / canvas_max_side

    def _map_point(x: float, y: float) -> tuple[float, float]:
        # Step 1: Add padding (image was pasted at (pad_x, 0) on canvas)
        x_padded = x + pad_x
        y_padded = y

        # Step 2: Apply skew transformation (x_new = x_old + dx * y_old)
        x_skewed = x_padded - dx * y_padded
        y_skewed = y_padded

        # Step 3: Apply crop offset
        x_cropped = x_skewed - crop_left
        y_cropped = y_skewed - crop_top

        # Step 4: Add paste offset on square canvas
        x_on_canvas = x_cropped + paste_left
        y_on_canvas = y_cropped + paste_top

        # Step 5: Apply resize to final dimensions
        x_final = x_on_canvas * resize_scale_x
        y_final = y_on_canvas * resize_scale_y

        return x_final, y_final

    transformed: list[dict] = []
    for bbox in paragraph_bboxes:
        x0, y0, x1, y1 = bbox.get("bbox", [0, 0, 0, 0])
        points = [
            _map_point(x0, y0),
            _map_point(x1, y0),
            _map_point(x1, y1),
            _map_point(x0, y1),
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        clamped_x0 = _clamp_value(min_x, 0.0, float(target_width))
        clamped_x1 = _clamp_value(max_x, 0.0, float(target_width))
        clamped_y0 = _clamp_value(min_y, 0.0, float(target_height))
        clamped_y1 = _clamp_value(max_y, 0.0, float(target_height))
        if clamped_x1 < clamped_x0:
            clamped_x1 = clamped_x0
        if clamped_y1 < clamped_y0:
            clamped_y1 = clamped_y0
        transformed.append(
            {
                **bbox,
                "bbox": _round_bbox([clamped_x0, clamped_y0, clamped_x1, clamped_y1]),
            }
        )

    return transformed


def skew(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Apply skew transformation with transparent background.

    Note: bg_color parameter is kept for API compatibility but not used.
    The transformation uses transparent fills to preserve alpha channel.
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    dx = random.uniform(-0.2, 0.2)
    skewed, skew_meta = _skew_within_bounds(image, dx)
    transformed_bboxes = _transform_paragraph_bboxes_for_skew(
        paragraph_bboxes_copy, skew_meta
    )
    return (
        skewed,
        {
            "transformation": "skew",
            "skew_factor": round(dx, 3),
        },
        transformed_bboxes,
    )
