import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


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
    except OSError:
        return ImageFont.load_default()


@dataclass
class WrappedParagraph:
    lines: list[str]
    text: str
    has_text: bool


@dataclass
class WrapResult:
    paragraphs: list[WrappedParagraph]
    has_overflow: bool


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
) -> WrapResult:
    """Wrap each paragraph to fit within the given width.

    Returns:
        WrapResult containing wrapped paragraphs and overflow flag
    """

    paragraphs = text.split("\n")
    wrapped_paragraphs: list[WrappedParagraph] = []
    has_overflow = False

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
                    # Word is too long to fit on a line - mark as overflow
                    has_overflow = True
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

    return WrapResult(paragraphs=wrapped_paragraphs, has_overflow=has_overflow)


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

    # Retry loop: reduce columns if words don't fit
    wrapped_paragraphs = None
    has_overflow = True
    while has_overflow and num_columns >= 1:
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
        current_column_width = resolved_column_width

        # Try wrapping with current column configuration
        wrap_result = wrap_text(draw, text, font, current_column_width, tab_width)
        wrapped_paragraphs = wrap_result.paragraphs
        has_overflow = wrap_result.has_overflow

        # If overflow detected and we can reduce columns, try again
        if has_overflow and num_columns > 1:
            num_columns -= 1
        else:
            # Either no overflow or we're at minimum columns (1)
            break

    # Final column configuration after retry loop
    column_width = current_column_width
    total_gap = column_gap * (num_columns - 1)
    block_width = column_width * num_columns + total_gap
    margin_x = max(0, (scaled_image_size[0] - block_width) // 2)

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


def _visualise_bboxes(
    image: Image.Image,
    paragraph_bboxes: list[dict],
    line_width: int = 2,
    show_labels: bool = True,
    max_label_chars: int = 20,
) -> Image.Image:
    """
    Draw bounding boxes on an image to visualize paragraph locations.

    Args:
        image: PIL Image object to draw on
        paragraph_bboxes: List of bbox dictionaries with format:
            [{"paragraph_index": int, "paragraph_text": str, "column": int, "bbox": [x1, y1, x2, y2]}]
        line_width: Width of the rectangle border in pixels
        show_labels: Whether to show paragraph text preview labels
        max_label_chars: Maximum number of characters to show in label preview

    Returns:
        PIL Image object with bounding boxes drawn
    """
    # Create a copy to avoid modifying the original
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image)

    # Define color palette for sequential cycling
    color_palette = [
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    # Load a small font for labels
    try:
        label_font = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        label_font = ImageFont.load_default()

    # Draw each bbox
    for idx, bbox_data in enumerate(paragraph_bboxes):
        # Get bbox coordinates
        bbox = bbox_data.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Select color from palette (cycle sequentially)
        color = color_palette[idx % len(color_palette)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label if enabled
        if show_labels:
            paragraph_text = bbox_data.get("paragraph_text", "")
            if paragraph_text:
                # Truncate text to max_label_chars
                label_text = paragraph_text[:max_label_chars]
                if len(paragraph_text) > max_label_chars:
                    label_text += "..."

                # Calculate label background size
                label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]

                # Position label at top-left of bbox with padding
                label_x = x1
                label_y = y1 - label_height - 4  # 4px padding

                # If label would go above image, place it inside the bbox
                if label_y < 0:
                    label_y = y1 + 2

                # Draw semi-transparent background for label
                background_padding = 2
                draw.rectangle(
                    [
                        label_x - background_padding,
                        label_y - background_padding,
                        label_x + label_width + background_padding,
                        label_y + label_height + background_padding,
                    ],
                    fill=(0, 0, 0, 200),  # Black with some transparency
                )

                # Draw label text
                draw.text(
                    (label_x, label_y),
                    label_text,
                    fill=(255, 255, 255),
                    font=label_font,
                )

    return visualized_image
