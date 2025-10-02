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

from datasets import Dataset, DatasetDict, Image, load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import create_image_with_text

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset creation"""

    dataset_path: str = "mideind/is_prototyping_corpus"
    text_column: str = "text"  # Column in dataset containing text
    data_directory: str = "igc"  # Subdirectory or config name in the dataset
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
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 1000  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )
    use_random_fonts: bool = True  # Whether to use random fonts
    use_random_backgrounds: bool = True  # Whether to use random background colors
    max_text_length: int = 2000  # Maximum characters per text before splitting


def get_icelandic_compatible_fonts():
    """Get a curated list of fonts known to support Icelandic characters."""
    # Curated list of fonts that support Latin Extended-A (includes Icelandic)
    font_names = [
        # macOS system fonts
        "Arial.ttf",
        "Arial Bold.ttf",
        "Arial Italic.ttf",
        "Helvetica.ttc",
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Courier New.ttf",
        "Courier New Bold.ttf",
        "Georgia.ttf",
        "Georgia Bold.ttf",
        "Verdana.ttf",
        "Verdana Bold.ttf",
        "Trebuchet MS.ttf",
        "Trebuchet MS Bold.ttf",
        "Comic Sans MS.ttf",
        "Impact.ttf",
        "Tahoma.ttf",
        "Tahoma Bold.ttf",
        # Apple system fonts
        "SF-Pro-Display-Regular.otf",
        "SF-Pro-Display-Bold.otf",
        "SF-Pro-Text-Regular.otf",
        "SF-Pro-Text-Bold.otf",
        # Additional macOS fonts
        "Palatino.ttc",
        "Futura.ttc",
        "Optima.ttc",
        "Baskerville.ttc",
        "Didot.ttc",
        "Avenir.ttc",
        "Avenir Next.ttc",
        # Ubuntu/Linux fonts
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSerif-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "FreeSans.ttf",
        "FreeSansBold.ttf",
        "FreeSerif.ttf",
        "FreeSerifBold.ttf",
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "NotoSerif-Regular.ttf",
        "NotoSerif-Bold.ttf",
    ]

    # Platform-specific font directories
    font_directories = [
        # macOS paths
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        # Ubuntu/Linux paths
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype/liberation2",
        "/usr/share/fonts/truetype/freefont",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/opentype/noto",
        # User fonts (both platforms)
        str(Path.home() / ".fonts"),
        str(Path.home() / ".local/share/fonts"),
    ]

    available_fonts = []

    for font_name in font_names:
        for font_dir in font_directories:
            font_path = Path(font_dir) / font_name
            if font_path.exists():
                available_fonts.append(str(font_path))
                break  # Found the font, no need to check other directories

    if not available_fonts:
        logger.warning("No Icelandic-compatible fonts found in standard locations")
        logger.info(f"Searched directories: {font_directories}")
    else:
        logger.info(f"Found {len(available_fonts)} Icelandic-compatible fonts")
        logger.debug(f"Available fonts: {available_fonts[:5]}...")  # Log first 5

    return available_fonts


def get_random_background_color():
    """Generate a random brown/beige/white background color."""
    color_palettes = [
        # Whites
        [(250, 250, 250), (255, 255, 255), (248, 248, 248), (245, 245, 245)],
        # Beiges
        [(245, 245, 220), (255, 248, 220), (250, 235, 215), (255, 239, 213)],
        # Light browns
        [(222, 184, 135), (210, 180, 140), (188, 143, 143), (205, 175, 149)],
        # Creams
        [(255, 253, 208), (255, 250, 205), (253, 245, 230), (250, 240, 230)],
    ]

    palette = random.choice(color_palettes)
    return random.choice(palette)


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
    new_data: dict[str, list] = {"text": [], "image": []}

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

    # Get available fonts if random fonts are enabled
    available_fonts = None
    if cfg.use_random_fonts:
        available_fonts = get_icelandic_compatible_fonts()
        if not available_fonts:
            logger.warning("No Icelandic-compatible fonts found. Using default font.")
            available_fonts = None
        else:
            logger.info(f"Found {len(available_fonts)} Icelandic-compatible fonts")

    # fix number of examples to generate if specified
    if cfg.num_examples:
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

                image, fitted_text = create_image_with_text(
                    remaining_text,
                    image_size=(width, height),
                    alignment=alignment,
                    font_size=font_size,
                    font_path=current_font_path,
                    bg_color=current_bg_color,
                    font_color=font_color,
                    vertical_alignment=vertical_alignment,
                    dpi=dpi,
                )

                if not fitted_text:
                    # No text could be fitted, break to avoid infinite loop
                    break

                new_data[cfg.text_column].append(fitted_text)
                new_data["image"].append(image)

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
    dataset = load_dataset(
        cfg.dataset_path,
        cfg.data_directory if hasattr(cfg, "data_directory") else None,
        split=cfg.split,
    )

    # select number of entries if specified
    if cfg.max_entries > 0:
        dataset = dataset.select(range(cfg.max_entries))

    texts = dataset[cfg.text_column]

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

    dataset_dict = DatasetDict(final_dataset)
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
