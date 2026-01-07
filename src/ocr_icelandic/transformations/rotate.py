import math
import random

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


def _rotate_within_bounds(image: Image.Image, angle: float) -> tuple[Image.Image, dict]:
    width, height = image.size

    # Calculate how much the corners can expand when rotated
    angle_rad = math.radians(abs(angle))
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))

    # Maximum dimensions after rotation
    max_width = int(width * cos_a + height * sin_a)
    max_height = int(width * sin_a + height * cos_a)

    # Create canvas large enough for rotation with transparent background
    pad = max(max_width - width, max_height - height) // 2 + 20
    canvas_width = width + pad * 2
    canvas_height = height + pad * 2
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    canvas.paste(image, (pad, pad), image if image.mode == "RGBA" else None)

    # Rotate with transparent fill
    rotated = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )

    # Crop from center
    center_x = rotated.width / 2
    center_y = rotated.height / 2

    # If rotated content is larger than target, scale it down
    scale = min(width / max_width, height / max_height, 1.0)

    crop_width = int(width / scale)
    crop_height = int(height / scale)

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2

    cropped = rotated.crop((left, top, left + crop_width, top + crop_height))

    # Resize back to original dimensions if we scaled
    if scale < 1.0:
        cropped = cropped.resize((width, height), Image.Resampling.BICUBIC)

    rotation_meta = {
        "pad": pad,
        "canvas_center": (canvas_width / 2, canvas_height / 2),
        "rotated_center": (rotated.width / 2, rotated.height / 2),
        "angle": angle,
        "crop_box": (left, top, left + crop_width, top + crop_height),
        "resize_scale": (
            width / crop_width if scale < 1.0 else 1.0,
            height / crop_height if scale < 1.0 else 1.0,
        ),
        "target_size": (width, height),
    }

    return cropped, rotation_meta


def _transform_paragraph_bboxes_for_rotation(
    paragraph_bboxes: list[dict], meta: dict
) -> list[dict]:
    if not paragraph_bboxes:
        return []

    pad = meta["pad"]
    canvas_center_x, canvas_center_y = meta["canvas_center"]
    rotated_center_x, rotated_center_y = meta["rotated_center"]
    angle_rad = math.radians(meta["angle"])
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    crop_left, crop_top, _, _ = meta["crop_box"]
    scale_x, scale_y = meta["resize_scale"]
    target_width, target_height = meta["target_size"]

    transformed: list[dict] = []

    def _map_point(x: float, y: float) -> tuple[float, float]:
        # Step 1: Add padding (image was pasted at (pad, pad) on canvas)
        canvas_x = x + pad
        canvas_y = y + pad

        # Step 2: Convert to relative coordinates from canvas center
        rel_x = canvas_x - canvas_center_x
        rel_y = canvas_y - canvas_center_y

        # Step 3: Apply rotation matrix (negated angle to match PIL's rotation)
        # PIL rotates counter-clockwise for positive angles, so we use -angle
        rotated_rel_x = cos_theta * rel_x + sin_theta * rel_y
        rotated_rel_y = -sin_theta * rel_x + cos_theta * rel_y

        # Step 4: Convert back to absolute coordinates in rotated image space
        rotated_x = rotated_rel_x + rotated_center_x
        rotated_y = rotated_rel_y + rotated_center_y

        # Step 5: Apply crop offset
        cropped_x = rotated_x - crop_left
        cropped_y = rotated_y - crop_top

        # Step 6: Apply resize scaling to final dimensions
        final_x = cropped_x * scale_x
        final_y = cropped_y * scale_y

        return final_x, final_y

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


def rotate(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Apply rotation transformation with transparent background.

    Note: bg_color parameter is kept for API compatibility but not used.
    The transformation uses transparent fills to preserve alpha channel.
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    angle = random.uniform(-5, 5)
    rotated, rotate_meta = _rotate_within_bounds(image, angle)
    transformed_bboxes = _transform_paragraph_bboxes_for_rotation(
        paragraph_bboxes_copy, rotate_meta
    )
    return (
        rotated,
        {
            "transformation": "rotate",
            "angle": round(angle, 2),
        },
        transformed_bboxes,
    )
