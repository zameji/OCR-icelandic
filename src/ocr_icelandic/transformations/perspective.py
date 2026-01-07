import random

from PIL import Image

from ocr_icelandic.transformations.shared import (
    _clamp_value,
    _copy_paragraph_bboxes,
    _round_bbox,
)


def _apply_perspective_distortion(
    image: Image.Image,
    distortion_type: str = "book_curve",
) -> tuple[Image.Image, dict]:
    """
    Apply perspective distortion to simulate book curvature or camera angles.

    Args:
        image: Input image (should be RGBA)
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
    # Use transparent background to preserve alpha channel
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
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

    # Calculate inverse perspective transform coefficients for PIL
    # PIL expects coefficients that map from destination to source (inverse mapping)
    inverse_coeffs = _find_perspective_coefficients(dst_points, src_points)

    # Apply perspective transformation with transparent fill
    transformed = canvas.transform(
        canvas.size,
        Image.Transform.PERSPECTIVE,
        inverse_coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    # Find bounding box of the transformed content (using alpha channel)
    bbox = _find_content_bbox(transformed)

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

    metadata["inverse_coeffs"] = inverse_coeffs
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
    alpha_threshold: int = 10,
) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of non-transparent content in the image.

    Args:
        image: RGBA image
        alpha_threshold: Minimum alpha value to consider as content (0-255)

    Returns:
        Bounding box (min_x, min_y, max_x, max_y) or None if no content found
    """
    # Ensure image is RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")

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
            if not isinstance(pixel, tuple) or len(pixel) < 4:
                continue
            # Check alpha channel - if pixel is not transparent, it's content
            alpha = pixel[3]
            if alpha > alpha_threshold:
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
    """
    if not paragraph_bboxes:
        return []

    pad = meta["pad"]
    src_points = meta["src_points"]
    dst_points = meta["dst_points"]
    crop_box = meta.get("crop_box", (0, 0, 0, 0))
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    target_width, target_height = meta["target_size"]

    scale_x = target_width / crop_width if crop_width > 0 else 1.0
    scale_y = target_height / crop_height if crop_height > 0 else 1.0

    # Get forward transformation coefficients (from source to destination)
    # PIL uses inverse coefficients, so we need to compute the forward transform
    forward_coeffs = _find_perspective_coefficients(src_points, dst_points)

    def _map_point(x: float, y: float) -> tuple[float, float]:
        """Transform a point through the full pipeline."""
        # Step 1: Add padding to get canvas coordinates
        canvas_x = x + pad
        canvas_y = y + pad

        # Step 2: Apply forward perspective transformation
        # The forward transform maps source to destination
        w = forward_coeffs[6] * canvas_x + forward_coeffs[7] * canvas_y + 1.0
        if abs(w) < 1e-10:  # Avoid division by zero
            w = 1e-10
        transformed_x = (
            forward_coeffs[0] * canvas_x
            + forward_coeffs[1] * canvas_y
            + forward_coeffs[2]
        ) / w
        transformed_y = (
            forward_coeffs[3] * canvas_x
            + forward_coeffs[4] * canvas_y
            + forward_coeffs[5]
        ) / w

        # Step 3: Subtract crop offset
        cropped_x = transformed_x - crop_left
        cropped_y = transformed_y - crop_top

        # Step 4: Apply resize scale
        final_x = cropped_x * scale_x
        final_y = cropped_y * scale_y

        return final_x, final_y

    transformed: list[dict] = []

    for bbox in paragraph_bboxes:
        x0, y0, x1, y1 = bbox.get("bbox", [0, 0, 0, 0])

        # Transform all 4 corners of the bounding box
        points = [
            _map_point(x0, y0),  # top-left
            _map_point(x1, y0),  # top-right
            _map_point(x1, y1),  # bottom-right
            _map_point(x0, y1),  # bottom-left
        ]

        # Calculate new axis-aligned bounding box from transformed corners
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Clamp to image bounds
        clamped_x0 = _clamp_value(min_x, 0.0, float(target_width))
        clamped_y0 = _clamp_value(min_y, 0.0, float(target_height))
        clamped_x1 = _clamp_value(max_x, 0.0, float(target_width))
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


def perspective(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Apply perspective transformation with transparent background.

    Note: bg_color parameter is kept for API compatibility but not used.
    The transformation uses transparent fills to preserve alpha channel.
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    perspective_type = random.choice(["book_curve", "camera_angle", "combined"])
    perspective_img, perspective_meta = _apply_perspective_distortion(
        image, perspective_type
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
