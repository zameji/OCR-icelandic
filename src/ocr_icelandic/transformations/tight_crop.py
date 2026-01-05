import random

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


def tight_crop(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Crop the image tightly around the text content when there's little text.
    This simulates documents where the paper is cut close to the content.

    Args:
        image: The input image
        bg_color: Background color for filling if needed
        paragraph_bboxes: List of paragraph bounding boxes

    Returns:
        Tuple of (transformed image, metadata dict, transformed bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # If no bounding boxes, return original
    if not paragraph_bboxes_copy:
        return (
            image,
            {"transformation": "tight_crop", "applied": False},
            paragraph_bboxes_copy,
        )

    # Calculate the union of all text bounding boxes
    all_x0 = []
    all_y0 = []
    all_x1 = []
    all_y1 = []

    for bbox_data in paragraph_bboxes_copy:
        bbox = bbox_data.get("bbox", [])
        if len(bbox) == 4:
            all_x0.append(bbox[0])
            all_y0.append(bbox[1])
            all_x1.append(bbox[2])
            all_y1.append(bbox[3])

    # If no valid bboxes, return original
    if not all_x0:
        return (
            image,
            {"transformation": "tight_crop", "applied": False},
            paragraph_bboxes_copy,
        )

    # Get overall text bounds
    text_x0 = min(all_x0)
    text_y0 = min(all_y0)
    text_x1 = max(all_x1)
    text_y1 = max(all_y1)

    text_width = text_x1 - text_x0
    text_height = text_y1 - text_y0

    # Calculate what percentage of the image is covered by text
    image_width, image_height = image.size
    text_area = text_width * text_height
    image_area = image_width * image_height
    coverage = text_area / image_area if image_area > 0 else 1.0

    # Only apply tight crop if text coverage is less than 50%
    # This means there's significant empty space
    if coverage >= 0.5:
        return (
            image,
            {
                "transformation": "tight_crop",
                "applied": False,
                "reason": "text_coverage_too_high",
                "coverage": round(coverage, 3),
            },
            paragraph_bboxes_copy,
        )

    # Add random padding around the text (5% to 15% of text dimensions)
    pad_ratio = random.uniform(0.05, 0.15)
    pad_x = int(text_width * pad_ratio)
    pad_y = int(text_height * pad_ratio)

    # Calculate crop box with padding
    crop_x0 = max(0, text_x0 - pad_x)
    crop_y0 = max(0, text_y0 - pad_y)
    crop_x1 = min(image_width, text_x1 + pad_x)
    crop_y1 = min(image_height, text_y1 + pad_y)

    # Ensure crop box is valid
    crop_width = crop_x1 - crop_x0
    crop_height = crop_y1 - crop_y0

    if crop_width <= 0 or crop_height <= 0:
        return (
            image,
            {
                "transformation": "tight_crop",
                "applied": False,
                "reason": "invalid_crop",
            },
            paragraph_bboxes_copy,
        )

    # Crop the image
    cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    # Create a transparent canvas of original size
    result = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

    # Calculate position to center the cropped content
    paste_x = (image_width - crop_width) // 2
    paste_y = (image_height - crop_height) // 2

    # Paste the cropped image onto the transparent canvas
    result.paste(cropped, (paste_x, paste_y))

    # Transform all bounding boxes (only translation, no scaling)
    transformed_bboxes = []
    for bbox_data in paragraph_bboxes_copy:
        bbox = bbox_data.get("bbox", [])
        if len(bbox) == 4:
            # Translate by crop offset, then by paste position
            new_x0 = bbox[0] - crop_x0 + paste_x
            new_y0 = bbox[1] - crop_y0 + paste_y
            new_x1 = bbox[2] - crop_x0 + paste_x
            new_y1 = bbox[3] - crop_y0 + paste_y

            # Clamp to image bounds
            new_x0 = _clamp_value(new_x0, 0.0, float(image_width))
            new_y0 = _clamp_value(new_y0, 0.0, float(image_height))
            new_x1 = _clamp_value(new_x1, 0.0, float(image_width))
            new_y1 = _clamp_value(new_y1, 0.0, float(image_height))

            # Ensure valid bbox
            if new_x1 < new_x0:
                new_x1 = new_x0
            if new_y1 < new_y0:
                new_y1 = new_y0

            transformed_bboxes.append(
                {**bbox_data, "bbox": _round_bbox([new_x0, new_y0, new_x1, new_y1])}
            )
        else:
            # Keep invalid bboxes as-is
            transformed_bboxes.append(bbox_data)

    return (
        result,
        {
            "transformation": "tight_crop",
            "applied": True,
            "coverage": round(coverage, 3),
            "crop_box": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
            "paste_position": [paste_x, paste_y],
            "pad_ratio": round(pad_ratio, 3),
        },
        transformed_bboxes,
    )
