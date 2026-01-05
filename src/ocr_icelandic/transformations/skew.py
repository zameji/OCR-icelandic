import random

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


def _skew_within_bounds(
    image: Image.Image, bg_color: str | tuple[int, int, int], dx: float
) -> tuple[Image.Image, dict]:
    width, height = image.size

    # Calculate the expanded width after skew
    max_shift = abs(dx * height)
    expanded_width = width + max_shift

    # Create large canvas
    pad_x = int(max_shift) + 40
    canvas_width = width + pad_x * 2
    # Use RGBA to preserve transparency
    if isinstance(bg_color, tuple) and len(bg_color) == 3:
        bg_rgba = bg_color + (255,)
    else:
        bg_rgba = bg_color
    canvas = Image.new("RGBA", (canvas_width, height), bg_rgba)
    canvas.paste(image, (pad_x, 0), image if image.mode == "RGBA" else None)

    # Apply skew
    matrix = (1, dx, 0, 0, 1, 0)
    skewed = canvas.transform(
        canvas.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg_rgba,
    )

    # Find center and crop expanded area
    center_x = skewed.width // 2
    crop_width = int(expanded_width)
    left = center_x - crop_width // 2

    cropped = skewed.crop((left, 0, left + crop_width, height))

    resized = cropped.resize((width, height), Image.Resampling.BICUBIC)
    skew_meta = {
        "dx": dx,
        "pad_x": pad_x,
        "expanded_width": expanded_width,
        "canvas_width": canvas_width,
        "crop_box": (left, 0, left + crop_width, height),
        "resize_scale_x": width / crop_width if crop_width else 1.0,
        "resize_scale_y": 1.0,
        "target_size": (width, height),
    }

    return resized, skew_meta


def _transform_paragraph_bboxes_for_skew(
    paragraph_bboxes: list[dict], meta: dict
) -> list[dict]:
    if not paragraph_bboxes:
        return []

    pad_x = meta["pad_x"]
    dx = meta["dx"]
    crop_left, crop_top, _, _ = meta["crop_box"]
    scale_x = meta["resize_scale_x"]
    scale_y = meta["resize_scale_y"]
    target_width, target_height = meta["target_size"]

    def _map_point(x: float, y: float) -> tuple[float, float]:
        x_with_pad = x + pad_x
        y_with_pad = y
        skewed_x = x_with_pad - dx * y_with_pad
        cropped_x = skewed_x - crop_left
        cropped_y = y_with_pad - crop_top
        return cropped_x * scale_x, cropped_y * scale_y

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
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    dx = random.uniform(-0.2, 0.2)
    skewed, skew_meta = _skew_within_bounds(image, bg_color, dx)
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
