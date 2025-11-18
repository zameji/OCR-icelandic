"""
Script to prepare a dataset with images generated from text data.
Handles text overflow by creating multiple images if necessary.
Saves the new dataset to disk and optionally pushes it to the Hugging Face Hub.
"""

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, Image, load_dataset
from fontTools.ttLib import TTFont
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import apply_random_transformation, create_image_with_text

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset creation"""

    dataset_path: str = "arnastofnun/IGC-2024"  # Hugging Face dataset path
    text_column: str = "document"  # Column in dataset containing text
    data_directory: str = "parla"  # Subdirectory or config name in the dataset
    split: str = "train"  # Which split to use from the dataset
    max_length: int = 512
    max_entries: int = 400
    show_sample: bool = False  # Whether to show a sample image after creation
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 72
    img_background_color: str = "white"
    font_path: str = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_size: int = 12
    font_color: str = "black"
    use_random_font_colors: bool = True  # Whether to use random font colors
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 0  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )
    use_random_fonts: bool = True  # Whether to use random fonts
    use_random_backgrounds: bool = True  # Whether to use random background colors
    max_text_length: int = 2000  # Maximum characters per text before splitting
    column_gap: int = 20  # Horizontal gap in pixels between columns
    num_columns: int | None = (
        None  # Number of columns when rendering text (None => random)
    )
    min_num_columns: int = 1  # Minimum number of columns when randomizing
    max_num_columns: int = 5  # Maximum number of columns when randomizing
    column_width: int | None = None  # Fixed column width in pixels (None => random)
    min_column_width: int = 100  # Minimum column width when randomizing
    max_column_width: int = 512  # Maximum column width when randomizing


def get_random_background_color():
    """Generate a random paper-like background color."""
    # Choose paper type
    paper_type = random.choice(["white", "cream", "aged"])

    if paper_type == "white":
        base = random.randint(245, 252)
        r = base + random.randint(-3, 3)
        g = base + random.randint(-5, 0)
        b = base + random.randint(-8, 0)
    elif paper_type == "cream":
        base = random.randint(235, 245)
        r = base + random.randint(0, 8)
        g = base + random.randint(-5, 3)
        b = base + random.randint(-12, -3)
    else:  # aged
        base = random.randint(220, 235)
        r = base + random.randint(5, 15)
        g = base + random.randint(0, 10)
        b = base + random.randint(-15, -5)

    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)


def get_random_font_color(bg_color, contrast_threshold=100):
    """Generate a random font color that contrasts with the background color."""

    def luminance(color):
        r, g, b = color
        return 0.299 * r + 0.587 * g + 0.114 * b

    bg_lum = luminance(bg_color)

    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        font_color = (r, g, b)
        font_lum = luminance(font_color)
        if abs(bg_lum - font_lum) >= contrast_threshold:
            return font_color


def split_long_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks at sentence boundaries to avoid mid-sentence splits.

    Args:
        text: The text to split
        max_length: Maximum length for each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    # Split on sentence boundaries
    sentences = (
        text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    )

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max_length, save current chunk and start new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def check_font_supports_char(fontpath, unicode_char):
    font = TTFont(fontpath)  # specify the path to the font in question

    for cmap in font["cmap"].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


def get_icelandic_compatible_fonts():
    # load fonts from font directory

    random.seed(42)  # For reproducibility

    # Check common font directories based on OS
    current_os = sys.platform

    font_dirs = []

    # macos
    if current_os.startswith("darwin"):
        font_dirs = [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
        ]
    # linux
    if current_os.startswith("linux"):
        font_dirs += [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ]
    # windows
    if current_os.startswith("win"):
        font_dirs += [
            str(Path.home() / "AppData/Local/Microsoft/Windows/Fonts"),
            str(Path.home() / "AppData/Roaming/Microsoft/Windows/Fonts"),
            "C:/Windows/Fonts",
        ]

    logger.info(f"Searching for fonts in directories: {font_dirs}")

    available_fonts = []
    characters_to_check = "ÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö"
    for font_dir in tqdm(font_dirs, desc="Scanning font directories"):
        font_path = Path(font_dir)
        if font_path.exists() and font_path.is_dir():
            for font_file in font_path.rglob("*.[tT][tT][fF]"):
                for char in characters_to_check:
                    if check_font_supports_char(font_file, char):
                        available_fonts.append(str(font_file))
                        break  # No need to check other characters for this font

    logger.info(f"Found {len(available_fonts)} Icelandic-compatible fonts.")

    return available_fonts


def _normalize_range(
    min_value: int, max_value: int, minimum: int = 1
) -> tuple[int, int]:
    """Ensure provided min/max values form a valid range."""

    min_value = max(minimum, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def generate_image_dataset(texts: list[str], cfg: DataConfig) -> Dataset:
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    Args:
        texts (list of str): List of text entries to convert to images
        cfg (DataConfig): Configuration for image generation
    Returns:
        Dataset: A Hugging Face Dataset with 'text' and 'image' columns
    """
    new_data: dict[str, list] = {
        "text": [],
        "image": [],
        "font_path": [],
        "bg_color": [],
        "font_color": [],
        "font_size": [],
        "image_width": [],
        "image_height": [],
        "image_dpi": [],
        "text_vertical_alignment": [],
        "text_horizontal_alignment": [],
        "paragraph_bboxes": [],
        "transformation": [],
    }

    # Get settings from config
    width = cfg.image_width
    height = cfg.image_height
    dpi = cfg.image_dpi
    font_size = cfg.font_size
    alignment = cfg.text_horizontal_alignment
    font_path = cfg.font_path
    bg_color = cfg.img_background_color
    font_color = cfg.font_color
    vertical_alignment = cfg.text_vertical_alignment

    available_fonts = None
    if cfg.use_random_fonts:
        available_fonts = get_icelandic_compatible_fonts()

    column_range = _normalize_range(cfg.min_num_columns, cfg.max_num_columns, minimum=1)
    column_width_range = _normalize_range(
        cfg.min_column_width, cfg.max_column_width, minimum=1
    )

    # fix number of examples to generate if specified
    if cfg.num_examples > 0:
        texts = texts[: cfg.num_examples]

    logger.info("Generating images from text...")
    total_splits = 0
    for text in tqdm(texts, desc="Processing text", unit="text"):
        # Split long texts first
        text_chunks = split_long_text(text.strip(), cfg.max_text_length)
        if len(text_chunks) > 1:
            total_splits += len(text_chunks) - 1

        for chunk in tqdm(text_chunks, desc="Processing chunk", leave=False):
            remaining_text = chunk
            while remaining_text:
                # Select random font if enabled
                current_font_path = font_path
                if cfg.use_random_fonts and available_fonts:
                    current_font_path = random.choice(available_fonts)

                # Select random background color if enabled
                current_bg_color = bg_color
                if cfg.use_random_backgrounds:
                    current_bg_color = get_random_background_color()

                # Select random font color if enabled
                if cfg.use_random_font_colors:
                    font_color = get_random_font_color(current_bg_color)

                if cfg.num_columns is not None and cfg.num_columns > 0:
                    num_columns = cfg.num_columns
                else:
                    num_columns = random.randint(*column_range)

                if cfg.column_width is not None and cfg.column_width > 0:
                    column_width = cfg.column_width
                else:
                    column_width = random.randint(*column_width_range)

                image, fitted_text, paragraph_bboxes = create_image_with_text(
                    remaining_text,
                    image_size=(width, height),
                    alignment=alignment,
                    font_size=font_size,
                    font_path=current_font_path,
                    bg_color=current_bg_color,
                    font_color=font_color,
                    vertical_alignment=vertical_alignment,
                    dpi=dpi,
                    num_columns=num_columns,
                    column_gap=cfg.column_gap,
                    column_width=column_width,
                )

                transformed_image, transformation_meta, transformed_paragraph_bboxes = (
                    apply_random_transformation(
                        image,
                        current_bg_color,
                        paragraph_bboxes=paragraph_bboxes,
                    )
                )

                if not fitted_text:
                    # No text could be fitted, break to avoid infinite loop
                    break

                new_data[cfg.text_column].append(fitted_text)
                new_data["image"].append(transformed_image)
                new_data["font_path"].append(current_font_path)
                new_data["bg_color"].append(current_bg_color)
                new_data["font_color"].append(font_color)
                new_data["font_size"].append(font_size)
                new_data["image_width"].append(width)
                new_data["image_height"].append(height)
                new_data["image_dpi"].append(dpi)
                new_data["text_vertical_alignment"].append(vertical_alignment)
                new_data["text_horizontal_alignment"].append(alignment)
                new_data["paragraph_bboxes"].append(transformed_paragraph_bboxes)
                new_data["transformation"].append(transformation_meta)

                # Update remaining text
                # This assumes create_image_with_text preserves original whitespace
                # and returns a prefix of the input text.
                remaining_text = remaining_text[len(fitted_text) :].lstrip()

    logger.info(f"Split {total_splits} long texts into multiple chunks")

    # Create a new Hugging Face Dataset
    image_dataset = Dataset.from_dict(new_data).cast_column("image", Image())
    return image_dataset


def display_sample(dataset: dict) -> None:
    logger.info("\nShowing first generated image...")
    if len(dataset["train"]) > 0:
        logger.info("Text for first image:")
        logger.info(f"'{dataset['train'][0]['text']}'")
        dataset["train"][0]["image"].show()


def create_image_dataset(cfg: DataConfig) -> None:
    """
    Create a dataset with images generated from text data.
    Args:
        cfg (DataConfig): Configuration for dataset creation
    """
    # load dataset
    dataset = cast(
        Dataset,
        load_dataset(
            cfg.dataset_path,
            cfg.data_directory if hasattr(cfg, "data_directory") else None,
            split=cfg.split,
        ),
    )

    # select number of entries if specified
    if cfg.max_entries > 0:
        dataset = dataset.select(range(cfg.max_entries))

    texts = list(dataset[cfg.text_column])

    # rename text column to 'text' if necessary
    if cfg.text_column != "text":
        logger.info(f"Renaming text column '{cfg.text_column}' to 'text'")
        dataset = dataset.rename_column(cfg.text_column, "text")
        cfg.text_column = "text"

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    logger.info(f"\nOriginal dataset size: {len(texts)}")
    logger.info(f"New image dataset size: {len(image_dataset)}")

    # Create a train/test/validation split (80/10/10)
    split_dataset = image_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    final_dataset = {
        "train": split_dataset["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }

    # Save the new dataset
    output_path = cfg.output_path
    # Use DatasetDict for saving splits

    dataset_dict = DatasetDict(list(final_dataset.items()))
    dataset_dict.save_to_disk(output_path)
    logger.info(f"Image dataset saved to {output_path}")

    # Display the first image as an example
    if cfg.show_sample:
        display_sample(final_dataset)

    # upload to huggingface dataset hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing dataset to the hub at {cfg.hub_repo_id}...")
        dataset_dict.push_to_hub(cfg.hub_repo_id)
        logger.info("Dataset pushed to the hub successfully.")


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = DataConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    create_image_dataset(cfg)


if __name__ == "__main__":
    main()
