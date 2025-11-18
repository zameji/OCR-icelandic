import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def load_font(
    font_path: str = "Arial.ttf",
    font_size: int = 20,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a TrueType font or default if not found.
    Args:
        font: Path to the .ttf font file
        font_size: Size of the font
    Returns:
        ImageFont.FreeTypeFont object
    """
    # Load a font
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        return ImageFont.load_default()


@dataclass
class WrappedParagraph:
    lines: list[str]
    text: str
    has_text: bool


@dataclass
class LinePlacement:
    text: str
    paragraph_index: int | None
    column_index: int
    line_index: int
    is_blank: bool


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    tab_width: int = 4,
) -> list[WrappedParagraph]:
    """Wrap each paragraph to fit within the given width."""

    paragraphs = text.split("\n")
    wrapped_paragraphs: list[WrappedParagraph] = []

    for paragraph in paragraphs:
        stripped_paragraph = paragraph.strip()
        if not stripped_paragraph:
            wrapped_paragraphs.append(
                WrappedParagraph(lines=[], text="", has_text=False)
            )
            continue

        leading_whitespace = ""
        left_stripped = paragraph.lstrip()
        if len(paragraph) > len(left_stripped):
            leading_whitespace = paragraph[: len(paragraph) - len(left_stripped)]
            leading_whitespace = leading_whitespace.replace("\t", " " * tab_width)

        left_stripped = left_stripped.replace("\t", " " * tab_width)
        words = left_stripped.split()
        paragraph_lines: list[str] = []
        current_line: list[str] = []
        is_first_line = True

        for word in words:
            test_line_base = " ".join(current_line + [word])
            test_line = (
                leading_whitespace + test_line_base if is_first_line else test_line_base
            )
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    paragraph_lines.append(
                        (leading_whitespace if is_first_line else "")
                        + " ".join(current_line)
                    )
                    is_first_line = False
                current_line = [word]
                test_line_base = " ".join(current_line)
                test_line = (
                    leading_whitespace + test_line_base
                    if is_first_line
                    else test_line_base
                )
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] > max_width:
                    paragraph_lines.append(
                        (leading_whitespace if is_first_line else "") + word
                    )
                    is_first_line = False
                    current_line = []

        if current_line:
            paragraph_lines.append(
                (leading_whitespace if is_first_line else "") + " ".join(current_line)
            )

        wrapped_paragraphs.append(
            WrappedParagraph(
                lines=paragraph_lines, text=stripped_paragraph, has_text=True
            )
        )

    return wrapped_paragraphs


def arrange_lines_in_columns(
    paragraphs: list[WrappedParagraph],
    max_lines_per_column: int,
    num_columns: int,
) -> tuple[list[LinePlacement], list[int]]:
    placements: list[LinePlacement] = []
    column_counts = [0] * num_columns
    current_column = 0

    def advance_column() -> None:
        nonlocal current_column
        while (
            current_column < num_columns
            and column_counts[current_column] >= max_lines_per_column
        ):
            current_column += 1

    def add_line(text: str, paragraph_index: int | None, is_blank: bool) -> bool:
        nonlocal current_column
        advance_column()
        if current_column >= num_columns:
            return False
        placements.append(
            LinePlacement(
                text=text,
                paragraph_index=paragraph_index,
                column_index=current_column,
                line_index=column_counts[current_column],
                is_blank=is_blank,
            )
        )
        column_counts[current_column] += 1
        return True

    for idx, paragraph in enumerate(paragraphs):
        if paragraph.has_text:
            for line in paragraph.lines:
                if not add_line(line, idx, is_blank=False):
                    return placements, column_counts
            if idx < len(paragraphs) - 1:
                if not add_line("", None, is_blank=True):
                    return placements, column_counts
        else:
            if not add_line("", None, is_blank=True):
                return placements, column_counts

    return placements, column_counts


def create_image_with_text(
    text: str,
    image_size: tuple[int, int] = (400, 100),
    font_path: str = "Arial.ttf",
    font_size: int = 20,
    font_color: str | tuple[int, int, int] = "black",
    bg_color: str | tuple[int, int, int] = "white",
    max_width_ratio: float = 0.9,
    tab_width: int = 4,
    alignment: str = "center",
    vertical_alignment: str = "center",
    dpi: int = 72,
    num_columns: int = 1,
    column_gap: int = 20,
    column_width: int | None = None,
) -> tuple[Image.Image, str, list[dict]]:
    """
    Create an image with text for OCR training and return paragraph bounding boxes.

    Args:
        text: Text to render
        image_size: Tuple of (width, height) in pixels at default DPI (72)
        font_path: Path to the .ttf font file
        font_size: Size of the font in points at default DPI (72)
        font_color: Color of the font
        bg_color: Background color of the image
        max_width_ratio: Ratio of image width to use for text (0.0-1.0)
        tab_width: Number of spaces to replace tabs with
        alignment: Text alignment - 'center', 'left', or 'right'
        vertical_alignment: Vertical text alignment - 'top', 'center', or 'bottom'
        dpi: Dots per inch for the image
        num_columns: Number of columns to use when laying out text
        column_gap: Gap in pixels between columns
        column_width: Fixed pixel width for each column (None to auto-size)

    Returns:
        tuple: (PIL Image object, string of text that actually fits in the image, paragraph bounding boxes)
    """
    scale_factor = dpi / 72.0
    scaled_image_size = (
        int(image_size[0] * scale_factor),
        int(image_size[1] * scale_factor),
    )
    scaled_font_size = int(font_size * scale_factor)

    image = Image.new("RGB", scaled_image_size, color=bg_color)
    image.info["dpi"] = (dpi, dpi)
    draw = ImageDraw.Draw(image)

    # add gaussian noice to the background to make it more realistic and less uniform
    noise = Image.effect_noise(scaled_image_size, 10)
    image = Image.blend(image, noise.convert("RGB"), 0.1)
    draw = ImageDraw.Draw(image)

    # add "dirt" texture to the background
    dirt_texture = Image.effect_noise(scaled_image_size, 5)
    image = Image.blend(image, dirt_texture.convert("RGB"), 0.05)
    draw = ImageDraw.Draw(image)

    font = load_font(font_path=font_path, font_size=scaled_font_size)

    usable_width = max(1, int(scaled_image_size[0] * max_width_ratio))
    num_columns = max(1, num_columns)
    column_gap = max(0, column_gap)
    total_gap = column_gap * (num_columns - 1)
    if usable_width - total_gap <= 0:
        num_columns = 1
        column_gap = 0
        total_gap = 0

    max_available_width = max(1, usable_width - total_gap)
    if max_available_width < num_columns:
        num_columns = 1
        column_gap = 0
        total_gap = 0
        max_available_width = max(1, usable_width)
    if column_width is not None:
        requested_width = max(1, column_width)
        resolved_column_width = min(requested_width, max_available_width)
        if resolved_column_width * num_columns > max_available_width:
            resolved_column_width = max(1, max_available_width // num_columns)
    else:
        resolved_column_width = max(1, max_available_width // num_columns)

    resolved_column_width = max(1, resolved_column_width)
    column_width = resolved_column_width
    block_width = column_width * num_columns + total_gap
    margin_x = max(0, (scaled_image_size[0] - block_width) // 2)

    wrapped_paragraphs = wrap_text(draw, text, font, column_width, tab_width)

    line_height = (
        draw.textbbox((0, 0), "Ag", font=font)[3]
        - draw.textbbox((0, 0), "Ag", font=font)[1]
    )
    line_spacing = int(line_height * 0.2)
    effective_line_height = line_height + line_spacing
    max_lines_per_column = int(
        max(1, (scaled_image_size[1] - line_height) // effective_line_height + 1)
    )

    placements, column_counts = arrange_lines_in_columns(
        wrapped_paragraphs, max_lines_per_column, num_columns
    )
    max_lines_used = max(column_counts) if column_counts else 0

    if max_lines_used > 0:
        block_height = max_lines_used * effective_line_height - line_spacing
    else:
        block_height = 0

    if vertical_alignment == "top" or not block_height:
        start_y = 0
    elif vertical_alignment == "bottom":
        start_y = max(0, scaled_image_size[1] - block_height)
    else:
        start_y = max(0, (scaled_image_size[1] - block_height) // 2)

    column_positions = [
        margin_x + c * (column_width + column_gap) for c in range(num_columns)
    ]

    paragraph_bboxes_map: dict[int, dict] = {}
    actual_text_lines: list[str] = []

    for placement in placements:
        actual_text_lines.append(placement.text)
        if not placement.text or placement.is_blank:
            continue

        column_x = column_positions[placement.column_index]
        y_position = start_y + placement.line_index * effective_line_height
        bbox = draw.textbbox((0, 0), placement.text, font=font)
        line_width = bbox[2] - bbox[0]
        if alignment == "left":
            x_position = column_x
        elif alignment == "right":
            x_position = column_x + max(0, column_width - line_width)
        else:
            x_position = column_x + (max(0, column_width - line_width) // 2)

        x_position_int = int(x_position)
        y_position_int = int(y_position)
        draw.text(
            (x_position_int, y_position_int),
            placement.text,
            fill=font_color,
            font=font,
        )

        paragraph_index = placement.paragraph_index
        if paragraph_index is None:
            continue

        current_bbox = paragraph_bboxes_map.get(paragraph_index)
        line_bbox = [
            x_position_int,
            y_position_int,
            x_position_int + line_width,
            y_position_int + line_height,
        ]
        if current_bbox:
            x0 = min(current_bbox["bbox"][0], line_bbox[0])
            y0 = min(current_bbox["bbox"][1], line_bbox[1])
            x1 = max(current_bbox["bbox"][2], line_bbox[2])
            y1 = max(current_bbox["bbox"][3], line_bbox[3])
            current_bbox["bbox"] = [x0, y0, x1, y1]
        else:
            paragraph_bboxes_map[paragraph_index] = {
                "paragraph_text": wrapped_paragraphs[paragraph_index].text,
                "column": placement.column_index,
                "bbox": line_bbox,
            }

    while actual_text_lines and not actual_text_lines[-1].strip():
        actual_text_lines.pop()

    actual_text = "\n".join(actual_text_lines)

    paragraph_bboxes = [
        {"paragraph_index": idx, **data}
        for idx, data in sorted(paragraph_bboxes_map.items())
    ]

    return image, actual_text, paragraph_bboxes


def apply_random_transformation(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    transformation = random.choice(["blur", "rotate", "ink_splashes", "skew"])

    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    if transformation == "blur":
        radius = random.uniform(0.1, 0.5)
        return (
            image.filter(ImageFilter.GaussianBlur(radius)),
            {
                "transformation": "blur",
                "radius": round(radius, 2),
            },
            paragraph_bboxes_copy,
        )

    if transformation == "rotate":
        angle = random.uniform(-5, 5)
        rotated, rotate_meta = _rotate_within_bounds(image, bg_color, angle)
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

    if transformation == "ink_splashes":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        splashes = random.randint(3, 6)
        for _ in range(splashes):
            radius = random.randint(10, 30)
            cx = random.randint(0, image.width)
            cy = random.randint(0, image.height)
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            color = (0, 0, 0, random.randint(80, 150))
            overlay_draw.ellipse(bbox, fill=color)
        combined = Image.alpha_composite(image.convert("RGBA"), overlay)
        return (
            combined.convert("RGB"),
            {
                "transformation": "ink_splashes",
                "splashes": splashes,
            },
            paragraph_bboxes_copy,
        )

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


def _rotate_within_bounds(
    image: Image.Image, bg_color: str | tuple[int, int, int], angle: float
) -> tuple[Image.Image, dict]:
    width, height = image.size

    # Calculate how much the corners can expand when rotated
    angle_rad = math.radians(abs(angle))
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))

    # Maximum dimensions after rotation
    max_width = int(width * cos_a + height * sin_a)
    max_height = int(width * sin_a + height * cos_a)

    # Create canvas large enough for rotation
    pad = max(max_width - width, max_height - height) // 2 + 20
    canvas_width = width + pad * 2
    canvas_height = height + pad * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    canvas.paste(image, (pad, pad))

    # Rotate
    rotated = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=bg_color,
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
        "canvas_size": (canvas_width, canvas_height),
        "rotation_center": (canvas_width / 2, canvas_height / 2),
        "rotated_size": (rotated.width, rotated.height),
        "rotation_offset": (
            rotated.width / 2 - canvas_width / 2,
            rotated.height / 2 - canvas_height / 2,
        ),
        "angle": angle,
        "crop_box": (left, top, left + crop_width, top + crop_height),
        "resize_scale": (
            width / crop_width if scale < 1.0 else 1.0,
            height / crop_height if scale < 1.0 else 1.0,
        ),
        "target_size": (width, height),
    }

    return cropped, rotation_meta


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
    canvas = Image.new("RGB", (canvas_width, height), bg_color)
    canvas.paste(image, (pad_x, 0))

    # Apply skew
    matrix = (1, dx, 0, 0, 1, 0)
    skewed = canvas.transform(
        canvas.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg_color,
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


def _copy_paragraph_bboxes(paragraph_bboxes: list[dict] | None) -> list[dict]:
    if not paragraph_bboxes:
        return []
    return [{**bbox, "bbox": list(bbox.get("bbox", []))} for bbox in paragraph_bboxes]


def _transform_paragraph_bboxes_for_rotation(
    paragraph_bboxes: list[dict], meta: dict
) -> list[dict]:
    if not paragraph_bboxes:
        return []

    pad = meta["pad"]
    center_x, center_y = meta["rotation_center"]
    offset_x, offset_y = meta["rotation_offset"]
    angle_rad = math.radians(meta["angle"])
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    crop_left, crop_top, _, _ = meta["crop_box"]
    scale_x, scale_y = meta["resize_scale"]
    target_width, target_height = meta["target_size"]

    transformed: list[dict] = []

    def _map_point(x: float, y: float) -> tuple[float, float]:
        canvas_x = x + pad
        canvas_y = y + pad
        rel_x = canvas_x - center_x
        rel_y = canvas_y - center_y
        rotated_x = cos_theta * rel_x - sin_theta * rel_y + center_x
        rotated_y = sin_theta * rel_x + cos_theta * rel_y + center_y
        rotated_x += offset_x
        rotated_y += offset_y
        cropped_x = rotated_x - crop_left
        cropped_y = rotated_y - crop_top
        return cropped_x * scale_x, cropped_y * scale_y

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


def _clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _round_bbox(coords: list[float]) -> list[int]:
    return [int(round(value)) for value in coords]


def dummy_text_with_line_breaks(num_sentences=5):
    sentences = [
        "Icelandic characters: ð, þ, æ, ö, á, é, í, ó, ú.",
        # "This is a sample sentence for OCR training.",
        # "Pillow makes it easy to create images with text.",
        # "Line breaks should be handled properly.",
        # "Tabs and spaces can affect text alignment.",
        # "This is the last sentence in this example.",
        # "Additional text to test overflow handling.",
        # "More text that might get cut off.",
        # "Even more text for testing purposes.",
        # "This line might not fit in smaller images.",
        # "Final line that definitely won't fit in tiny images.",
        "„Megi hann fara og vera en ég vona svo sannarlega að hann komi aldrei aftur til Íslands,“ segir Helgi Magnús Gunnarsson fyrrverandi vararíkssaksóknari um nýjustu vendingar í máli Mohamads Kourani. Helgi, sem sætti líflátshótunum",
    ]
    selected_sentences = random.choices(sentences, k=num_sentences)
    return "\n".join(selected_sentences)
