from pathlib import Path
import random
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFilter

from ocr_icelandic.transformations.perspective import perspective
from ocr_icelandic.transformations.rotate import rotate
from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes
from ocr_icelandic.transformations.skew import skew
from ocr_icelandic.transformations.tight_crop import tight_crop


def blur(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    radius = random.uniform(0.1, 0.5)
    return (
        image.filter(ImageFilter.GaussianBlur(radius)),
        {
            "transformation": "blur",
            "radius": round(radius, 2),
        },
        paragraph_bboxes_copy,
    )


def ink_splashes(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    splashes = random.randint(3, 6)
    for _ in range(splashes):
        radius = random.randint(10, 30)
        cx = random.randint(0, image.width)
        cy = random.randint(0, image.height)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        color = (0, 0, 0, random.randint(80, 150))

        # Create temporary image for single splash with blur
        splash = Image.new("RGBA", image.size, (0, 0, 0, 0))
        splash_draw = ImageDraw.Draw(splash)
        splash_draw.ellipse(bbox, fill=color)
        splash = splash.filter(ImageFilter.GaussianBlur(radius=2))

        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, splash)
    # Ensure image is RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)
    return (
        combined,
        {
            "transformation": "ink_splashes",
            "splashes": splashes,
        },
        paragraph_bboxes_copy,
    )


stain_textures = list(Path("assets/stains").glob("*.png"))


def textured_stains(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, any]] | None = None,
) -> tuple[Image.Image, dict[str, any], list[dict[str, any]]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    texture = random.choice(stain_textures)
    stain = Image.open(texture).convert("RGBA")
    # Adjust scale factor to ensure stain fits within image
    max_scale = min(image.width / stain.width, image.height / stain.height) * 0.8
    scale_factor = random.uniform(0.5, min(1.5, max_scale))
    new_size = (int(stain.width * scale_factor), int(stain.height * scale_factor))
    stain = stain.resize(new_size, Image.Resampling.LANCZOS)

    # Reduce opacity to 80%
    alpha = stain.split()[3]
    alpha = alpha.point(lambda p: int(p * 0.8))
    stain.putalpha(alpha)

    # Allow stain to be positioned partially outside image bounds
    pos_x = random.randint(-stain.width // 2, image.width - stain.width // 2)
    pos_y = random.randint(-stain.height // 2, image.height - stain.height // 2)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste(stain, (pos_x, pos_y), stain)

    # Ensure image is RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)
    return (
        combined,
        {
            "transformation": "coffee_stains",
            "position": (pos_x, pos_y),
            "scale_factor": round(scale_factor, 2),
        },
        paragraph_bboxes_copy,
    )


def dusty_paper(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """create grainy overlay to simulate dusty paper
    varies in grain size and intensity"""
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    grain_size = random.randint(1, 3)
    intensity = random.uniform(0.05, 0.15)
    noise = Image.effect_noise(image.size, grain_size * 10)
    grainy_overlay = noise.convert("RGBA" if image.mode == "RGBA" else "RGB")
    dusty_image = Image.blend(image, grainy_overlay, intensity)
    return (
        dusty_image,
        {
            "transformation": "dusty-paper",
            "grain_size": grain_size,
            "intensity": round(intensity, 3),
        },
        paragraph_bboxes_copy,
    )


def light_reflection(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """Simulate light reflection on the image."""
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # Get image dimensions
    width, height = image.size

    # Get position for the reflection center
    center_x = random.randint(int(width * 0.2), int(width * 0.8))
    center_y = random.randint(int(height * 0.2), int(height * 0.8))

    # Get ellipse size
    ellipse_width = random.randint(width // 8, width // 4)
    ellipse_height = random.randint(height // 8, height // 6)

    # Create a mask for the reflection
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)

    # Draw ellipse on mask
    left = center_x - ellipse_width // 2
    top = center_y - ellipse_height // 2
    right = center_x + ellipse_width // 2
    bottom = center_y + ellipse_height // 2

    mask_draw.ellipse([left, top, right, bottom], fill=255)

    # Apply blur for softer edges
    mask = mask.filter(
        ImageFilter.GaussianBlur(radius=(ellipse_width + ellipse_height) // 4)
    )

    # Create overlay
    light_color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255),
        200,
    )
    reflection = Image.new("RGBA", (width, height), light_color)
    reflection.putalpha(mask)

    # Overlay over image
    result = Image.alpha_composite(image.convert("RGBA"), reflection)

    return (
        result.convert(image.mode),
        {
            "transformation": "light_reflection",
            "center_x": center_x,
            "center_y": center_y,
            "ellipse_width": ellipse_width,
            "ellipse_height": ellipse_height,
        },
        paragraph_bboxes_copy,
    )


def reverse_bleed_through(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """Simulate bleed-through effect on the image."""
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    intensity = np.random.uniform(
        0.01, 0.04
    )  # Adjust intensity of the bleed-through effect

    # Store original alpha channel if present
    has_alpha = image.mode == "RGBA"
    if has_alpha:
        alpha_channel = image.split()[3]

    # Convert PIL image to numpy array (RGB only for processing)
    if has_alpha:
        img_rgb = image.convert("RGB")
        img_array = np.array(img_rgb)
    else:
        img_array = np.array(image)

    # Flip the image horizontally
    flipped = cv2.flip(img_array, 1)

    # Apply random shift
    # Calculate minimum shift based on image size (10% of width/height)
    min_shift_x = max(3, int(img_array.shape[1] * 0.1))
    min_shift_y = max(3, int(img_array.shape[0] * 0.1))

    # Generate random shift
    shift_x = np.random.choice([-1, 1]) * np.random.randint(
        min_shift_x, min_shift_x + 10
    )
    shift_y = np.random.choice([-1, 1]) * np.random.randint(
        min_shift_y, min_shift_y + 10
    )

    # Create transformation matrix to shift the flipped image (bleed-through)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(
        flipped,
        M,
        (img_array.shape[1], img_array.shape[0]),
        borderValue=(255, 255, 255),
    )

    # Create mask for dark colors (low intensity values - light colors should not bleed through)
    gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    dark_mask = gray_shifted < 128

    # Apply where original image is light and shifted image is dark
    gray_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    light_mask = gray_original > 200

    # Combine masks
    bleed_mask = dark_mask & light_mask

    # Apply the effect
    result = img_array.copy().astype(np.float32)
    for i in range(3):  # Apply to each color channel
        result[:, :, i] = np.where(
            bleed_mask,
            img_array[:, :, i] * (1 - intensity) + shifted[:, :, i] * intensity,
            img_array[:, :, i],
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(result)

    # Restore alpha channel if original had it
    if has_alpha:
        result_image = result_image.convert("RGBA")
        result_image.putalpha(alpha_channel)

    return (
        result_image,
        {
            "transformation": "reverse_bleed_through",
            "intensity": round(intensity, 3),
            "shift_x": shift_x,
            "shift_y": shift_y,
        },
        paragraph_bboxes_copy,
    )


def shadow_overlay(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """Cast a random uneven shadow from one edge with fuzzy borders."""
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Shadow layer
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    # Random edge selection (0=top, 1=right, 2=bottom, 3=left)
    edge = random.randint(0, 3)

    # Random shadow parameters - how far the shadow extends and its opacity
    max_depth = random.uniform(0.15, 0.5) * min(image.width, image.height)
    opacity = random.randint(20, 120)

    # Create uneven shadow polygon
    points = []
    polygons_points = 3
    if edge == 0:  # Top edge
        points = [(0, 0), (image.width, 0)]
        for i in range(polygons_points):
            x = (i + 1) * image.width / 6
            y = random.uniform(max_depth * 0.3, max_depth)
            points.append((x, y))
        points.extend([(0, random.uniform(max_depth * 0.3, max_depth))])
    elif edge == 1:  # Right edge
        points = [(image.width, 0), (image.width, image.height)]
        for i in range(polygons_points):
            x = image.width - random.uniform(max_depth * 0.3, max_depth)
            y = (i + 1) * image.height / 6
            points.append((x, y))
        points.extend([(image.width - random.uniform(max_depth * 0.3, max_depth), 0)])
    elif edge == 2:  # Bottom edge
        points = [(0, image.height), (image.width, image.height)]
        for i in range(polygons_points):
            x = (i + 1) * image.width / 6
            y = image.height - random.uniform(max_depth * 0.3, max_depth)
            points.append((x, y))
        points.extend([(0, image.height - random.uniform(max_depth * 0.3, max_depth))])
    else:  # Left edge
        points = [(0, 0), (0, image.height)]
        for i in range(polygons_points):
            x = random.uniform(max_depth * 0.3, max_depth)
            y = (i + 1) * image.height / 6
            points.append((x, y))
        points.extend([(random.uniform(max_depth * 0.3, max_depth), 0)])

    # Draw shadow polygon
    shadow_draw.polygon(points, fill=(0, 0, 0, opacity))

    # Apply blur for fuzzy edges
    blur_radius = random.uniform(10, 30)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Composite with original image
    image = Image.alpha_composite(image, shadow)

    return (
        image,
        {
            "transformation": "shadow_overlay",
            "edge": edge,
            "max_depth": round(max_depth, 2),
            "opacity": opacity,
            "blur_radius": round(blur_radius, 2),
        },
        paragraph_bboxes_copy,
    )


CONTENT_TRANSFORMATIONS = [
    blur,
    ink_splashes,
    dusty_paper,
    reverse_bleed_through,
    textured_stains,
]
PERSPECTIVE_TRANSFORMATIONS = [
    rotate,
    skew,
    perspective,
    tight_crop,
]
POSTPROCESSING_TRANSFORMATIONS = [
    light_reflection,
    shadow_overlay,
]


def _get_random_subset(transformations: list) -> list:
    k = random.randint(0, len(transformations))
    return random.sample(transformations, k)


def apply_random_transformation(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, list[dict], list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # Convert to RGBA at the start of the pipeline
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Apply content and perspective transformations with RGBA
    pre_composite_transformations = [
        *_get_random_subset(CONTENT_TRANSFORMATIONS),
        *_get_random_subset(PERSPECTIVE_TRANSFORMATIONS),
    ]

    transformation_meta: list[dict] = []
    for transform in pre_composite_transformations:
        image, meta, paragraph_bboxes_copy = transform(
            image, bg_color, paragraph_bboxes_copy
        )

        transformation_meta.append(meta)

    # Composite RGBA onto background color
    background = Image.new("RGB", image.size, bg_color)
    if image.mode == "RGBA":
        background.paste(image, (0, 0), image)
        image = background
    else:
        # Fallback if transformation returned non-RGBA
        image = image.convert("RGB")

    # Apply postprocessing transformations after background composite
    postprocessing_transformations = _get_random_subset(POSTPROCESSING_TRANSFORMATIONS)
    for transform in postprocessing_transformations:
        image, meta, paragraph_bboxes_copy = transform(
            image, bg_color, paragraph_bboxes_copy
        )

        transformation_meta.append(meta)

    return image, transformation_meta, paragraph_bboxes_copy
