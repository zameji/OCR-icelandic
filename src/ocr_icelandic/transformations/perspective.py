import random

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


def _apply_perspective_distortion(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    distortion_type: str = "book_curve",
) -> tuple[Image.Image, dict]:
    """
    Apply perspective distortion to simulate book curvature or camera angles.

    Args:
        image: Input image
        bg_color: Background color for filling
        distortion_type: Type of distortion ("book_curve", "camera_angle", or "combined")

    Returns:
        Tuple of (transformed image, metadata dictionary)
    """
    width, height = image.size

    # Create a much larger canvas to accommodate the transformation
    # This ensures content doesn't get cut off
    pad = max(width, height) // 2  # Dynamic padding based on image size
    canvas_width = width + pad * 2
    canvas_height = height + pad * 2
    # Use RGBA to preserve transparency
    if isinstance(bg_color, tuple) and len(bg_color) == 3:
        bg_rgba = bg_color + (255,)
    else:
        bg_rgba = bg_color
    canvas = Image.new("RGBA", (canvas_width, canvas_height), bg_rgba)
    canvas.paste(image, (pad, pad), image if image.mode == "RGBA" else None)

    # Define the four corners of the original image on the canvas
    # Top-left, top-right, bottom-right, bottom-left
    src_points = [
        (pad, pad),
        (pad + width, pad),
        (pad + width, pad + height),
        (pad, pad + height),
    ]

    # Initialize destination points (will be modified based on distortion type)
    dst_points = list(src_points)

    metadata = {
        "pad": pad,
        "canvas_size": (canvas_width, canvas_height),
        "src_points": src_points,
    }

    if distortion_type == "book_curve":
        # Simulate book spine curvature - push center inward, pull edges outward
        # Reduced intensity to keep content within bounds
        curve_intensity = random.uniform(0.02, 0.08)
        vertical_offset = int(height * curve_intensity)
        horizontal_inset = int(width * curve_intensity * 0.3)

        # Adjust corners to create curve effect
        # Keep well within the padded canvas
        dst_points = [
            (
                pad + horizontal_inset,
                pad + vertical_offset // 2,
            ),  # top-left (reduced vertical)
            (pad + width - horizontal_inset, pad + vertical_offset // 2),  # top-right
            (
                pad + width - horizontal_inset,
                pad + height - vertical_offset // 2,
            ),  # bottom-right
            (
                pad + horizontal_inset,
                pad + height - vertical_offset // 2,
            ),  # bottom-left
        ]

        metadata.update(
            {
                "curve_intensity": round(curve_intensity, 3),
                "vertical_offset": vertical_offset,
                "horizontal_inset": horizontal_inset,
            }
        )

    elif distortion_type == "camera_angle":
        # Simulate viewing document from an angle (trapezoidal perspective)
        angle_type = random.choice(["top", "bottom", "left", "right"])
        # Reduced strength to prevent content from going out of bounds
        perspective_strength = random.uniform(0.05, 0.15)

        if angle_type == "top":
            # Camera above, looking down - top appears smaller
            shrink = int(width * perspective_strength)
            dst_points = [
                (pad + shrink, pad),
                (pad + width - shrink, pad),
                (pad + width, pad + height),
                (pad, pad + height),
            ]
        elif angle_type == "bottom":
            # Camera below, looking up - bottom appears smaller
            shrink = int(width * perspective_strength)
            dst_points = [
                (pad, pad),
                (pad + width, pad),
                (pad + width - shrink, pad + height),
                (pad + shrink, pad + height),
            ]
        elif angle_type == "left":
            # Camera to the left - left side appears smaller
            shrink = int(height * perspective_strength)
            dst_points = [
                (pad, pad + shrink),
                (pad + width, pad),
                (pad + width, pad + height),
                (pad, pad + height - shrink),
            ]
        else:  # right
            # Camera to the right - right side appears smaller
            shrink = int(height * perspective_strength)
            dst_points = [
                (pad, pad),
                (pad + width, pad + shrink),
                (pad + width, pad + height - shrink),
                (pad, pad + height),
            ]

        metadata.update(
            {
                "angle_type": angle_type,
                "perspective_strength": round(perspective_strength, 3),
            }
        )

    else:  # combined
        # Combine both book curve and camera angle
        # Very conservative values for combined effect
        curve = random.uniform(0.02, 0.05)
        angle = random.uniform(0.03, 0.08)

        v_offset = int(height * curve)
        h_inset = int(width * curve * 0.3)
        shrink = int(width * angle * 0.5)

        # Create a combined effect - keep within bounds
        dst_points = [
            (pad + h_inset + shrink, pad + v_offset // 2),
            (pad + width - h_inset - shrink, pad + v_offset // 2),
            (pad + width - h_inset, pad + height - v_offset // 2),
            (pad + h_inset, pad + height - v_offset // 2),
        ]

        metadata.update(
            {
                "curve_intensity": round(curve, 3),
                "perspective_strength": round(angle, 3),
            }
        )

    metadata["dst_points"] = dst_points

    # Calculate perspective transform coefficients
    coeffs = _find_perspective_coefficients(src_points, dst_points)

    # Apply perspective transformation
    transformed = canvas.transform(
        canvas.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg_rgba,
    )

    # Find bounding box of the transformed content
    bbox = _find_content_bbox(transformed, bg_color)

    # Crop and resize back to original dimensions
    if bbox:
        x0, y0, x1, y1 = bbox
        # Add safety margin to ensure we capture all content
        margin = 10
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(canvas_width, x1 + margin)
        y1 = min(canvas_height, y1 + margin)

        cropped = transformed.crop((x0, y0, x1, y1))
        metadata["crop_box"] = (x0, y0, x1, y1)
        metadata["crop_size"] = (x1 - x0, y1 - y0)
    else:
        # If no content found, use the original image area with padding
        x0, y0 = pad // 2, pad // 2
        x1, y1 = pad + width + pad // 2, pad + height + pad // 2
        cropped = transformed.crop((x0, y0, x1, y1))
        metadata["crop_box"] = (x0, y0, x1, y1)
        metadata["crop_size"] = (x1 - x0, y1 - y0)

    # Resize back to original size
    final = cropped.resize((width, height), Image.Resampling.BICUBIC)

    metadata["coeffs"] = coeffs
    metadata["target_size"] = (width, height)

    return final, metadata


def _find_perspective_coefficients(
    src_points: list[tuple], dst_points: list[tuple]
) -> tuple:
    """
    Calculate perspective transform coefficients from source to destination points.
    Uses the 8-parameter perspective transform matrix.
    """
    # Extract coordinates
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = src_points
    (X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3) = dst_points

    # Solve the linear system for the 8 coefficients
    # This is a simplified approach; for more accuracy, use matrix operations
    matrix = []
    matrix.append([x0, y0, 1, 0, 0, 0, -X0 * x0, -X0 * y0])
    matrix.append([0, 0, 0, x0, y0, 1, -Y0 * x0, -Y0 * y0])
    matrix.append([x1, y1, 1, 0, 0, 0, -X1 * x1, -X1 * y1])
    matrix.append([0, 0, 0, x1, y1, 1, -Y1 * x1, -Y1 * y1])
    matrix.append([x2, y2, 1, 0, 0, 0, -X2 * x2, -X2 * y2])
    matrix.append([0, 0, 0, x2, y2, 1, -Y2 * x2, -Y2 * y2])
    matrix.append([x3, y3, 1, 0, 0, 0, -X3 * x3, -X3 * y3])
    matrix.append([0, 0, 0, x3, y3, 1, -Y3 * x3, -Y3 * y3])

    b = [X0, Y0, X1, Y1, X2, Y2, X3, Y3]

    # Simple Gaussian elimination for 8x8 system
    coeffs = _solve_linear_system(matrix, b)

    return tuple(coeffs)


def _solve_linear_system(matrix: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system using Gaussian elimination."""
    n = len(matrix)
    # Create augmented matrix
    for i in range(n):
        matrix[i].append(b[i])

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if matrix[i][i] == 0:
                continue
            factor = matrix[k][i] / matrix[i][i]
            for j in range(i, n + 1):
                matrix[k][j] -= factor * matrix[i][j]

    # Back substitution
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        if matrix[i][i] == 0:
            solution[i] = 0
            continue
        solution[i] = matrix[i][n]
        for j in range(i + 1, n):
            solution[i] -= matrix[i][j] * solution[j]
        solution[i] /= matrix[i][i]

    return solution


def _find_content_bbox(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    threshold: int = 20,
) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of non-background content in the image.
    """
    # Convert bg_color to RGB tuple if it's a string
    if isinstance(bg_color, str):
        from PIL import ImageColor

        bg_rgb = ImageColor.getrgb(bg_color)
    else:
        bg_rgb = bg_color

    # Convert to numpy-like processing
    pixels = image.load()
    if pixels is None:
        return None
    width, height = image.size

    min_x, min_y = width, height
    max_x, max_y = 0, 0

    found_content = False

    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            if not isinstance(pixel, tuple) or len(pixel) < 3:
                continue
            r, g, b = pixel[0], pixel[1], pixel[2]
            # Check if pixel is significantly different from background
            diff = abs(r - bg_rgb[0]) + abs(g - bg_rgb[1]) + abs(b - bg_rgb[2])
            if diff > threshold:
                found_content = True
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if not found_content:
        return None

    # Add small padding
    pad = 5
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(width, max_x + pad)
    max_y = min(height, max_y + pad)

    return (min_x, min_y, max_x, max_y)


def _transform_paragraph_bboxes_for_perspective(
    paragraph_bboxes: list[dict], meta: dict
) -> list[dict]:
    """
    Transform paragraph bounding boxes through perspective transformation.
    This is an approximation since the exact inverse transform is complex.
    """
    if not paragraph_bboxes:
        return []

    # For perspective transforms, the bounding boxes will be approximate
    # since we're doing multiple transformations (pad, perspective, crop, resize)
    # We'll use a simplified approach that scales the boxes appropriately

    pad = meta["pad"]
    canvas_width, canvas_height = meta["canvas_size"]
    crop_box = meta.get("crop_box", (0, 0, canvas_width, canvas_height))
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    target_width, target_height = meta["target_size"]

    scale_x = target_width / crop_width if crop_width > 0 else 1.0
    scale_y = target_height / crop_height if crop_height > 0 else 1.0

    transformed: list[dict] = []

    for bbox in paragraph_bboxes:
        x0, y0, x1, y1 = bbox.get("bbox", [0, 0, 0, 0])

        # Translate to canvas coordinates
        canvas_x0 = x0 + pad
        canvas_y0 = y0 + pad
        canvas_x1 = x1 + pad
        canvas_y1 = y1 + pad

        # After perspective transform, estimate the box position
        # (This is simplified - exact transformation would require applying the inverse)
        # For now, we just adjust for the crop and resize
        cropped_x0 = canvas_x0 - crop_left
        cropped_y0 = canvas_y0 - crop_top
        cropped_x1 = canvas_x1 - crop_left
        cropped_y1 = canvas_y1 - crop_top

        # Scale to final size
        final_x0 = cropped_x0 * scale_x
        final_y0 = cropped_y0 * scale_y
        final_x1 = cropped_x1 * scale_x
        final_y1 = cropped_y1 * scale_y

        # Clamp to image bounds
        clamped_x0 = _clamp_value(final_x0, 0.0, float(target_width))
        clamped_y0 = _clamp_value(final_y0, 0.0, float(target_height))
        clamped_x1 = _clamp_value(final_x1, 0.0, float(target_width))
        clamped_y1 = _clamp_value(final_y1, 0.0, float(target_height))

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


def perspective(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    perspective_type = random.choice(["book_curve", "camera_angle", "combined"])
    perspective_img, perspective_meta = _apply_perspective_distortion(
        image, bg_color, perspective_type
    )
    transformed_bboxes = _transform_paragraph_bboxes_for_perspective(
        paragraph_bboxes_copy, perspective_meta
    )
    return (
        perspective_img,
        {
            "transformation": "perspective",
            "type": perspective_type,
            **perspective_meta,
        },
        transformed_bboxes,
    )
