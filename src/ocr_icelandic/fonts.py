"""
Font management utilities for OCR Icelandic.
Handles font discovery, Google Fonts synchronization, and Icelandic character support checking.
"""

import logging
import random
import sys
from pathlib import Path
from typing import TypedDict

import requests
from fontTools.ttLib import TTFont
import tenacity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class GoogleFont(TypedDict):
    family: str
    variants: list[str]
    subsets: list[str]
    version: str
    lastModified: str
    files: dict[str, str]
    category: str
    kind: str


class GoogleFontsResponse(TypedDict):
    kind: str
    items: list[GoogleFont]


def check_font_supports_char(fontpath, unicode_char):
    """
    Check if a font file supports a specific Unicode character.

    Args:
        fontpath: Path to the font file
        unicode_char: Unicode character to check

    Returns:
        True if the font supports the character, False otherwise
    """
    font = TTFont(fontpath)  # specify the path to the font in question

    for cmap in font["cmap"].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


def fetch_google_fonts_list(api_key: str) -> list[GoogleFont] | None:
    """
    Fetch list of all Google Fonts from the API.

    Args:
        api_key: Google Fonts API key

    Returns:
        List of font metadata dictionaries, or None on error
    """
    try:
        url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Google Fonts list: {e}")
        return None


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
)
def download_font_file(font_url: str, output_path: Path) -> None:
    """
    Download a single font file from URL.

    Args:
        font_url: URL to the font file
        output_path: Path where to save the font file

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(font_url, timeout=30)
        response.raise_for_status()

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write font file
        with open(output_path, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download font from {font_url}: {e}")
        raise e
    except OSError as e:
        logger.warning(f"Failed to write font to {output_path}: {e}")
        raise e


def download_font_task(font: GoogleFont, fonts_path: Path) -> tuple[int, int, int]:
    """Download all variants of a single font family."""
    local_downloaded = 0
    local_skipped = 0
    local_failed = 0

    family = font.get("family", "Unknown")
    files = font.get("files", {})

    if not files:
        return local_downloaded, local_skipped, local_failed

    # Download all variants (regular, bold, italic, etc.)
    for variant, url in files.items():
        # Create safe filename
        safe_family = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in family
        )
        safe_variant = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in variant
        )
        filename = f"{safe_family}-{safe_variant}.ttf"
        output_path = fonts_path / filename

        # Skip if already exists
        if output_path.exists():
            local_skipped += 1
            continue

        # Download the font
        try:
            download_font_file(url, output_path)
            local_downloaded += 1
        except Exception as e:
            logger.warning(f"Final failure: Failed to download font from {url}: {e}")
            local_failed += 1

    return local_downloaded, local_skipped, local_failed


def sync_google_fonts(api_key: str, google_fonts_dir: str) -> bool:
    """
    Synchronize Google Fonts to local directory.
    Downloads only fonts that are not already present.

    Args:
        api_key: Google Fonts API key
        google_fonts_dir: Directory to store downloaded fonts

    Returns:
        True if successful (at least some fonts available), False if completely failed
    """
    fonts_path = Path(google_fonts_dir)

    # Try to use cached fonts if API fails
    fonts_list = fetch_google_fonts_list(api_key)

    if fonts_list is None:
        # API failed, check if we have cached fonts
        if fonts_path.exists() and any(fonts_path.glob("**/*.[tT][tT][fF]")):
            logger.warning(
                "Google Fonts API request failed, but using cached fonts from "
                f"{google_fonts_dir}"
            )
            return True
        else:
            logger.warning(
                "Google Fonts API request failed and no cached fonts found in "
                f"{google_fonts_dir}"
            )
            return False

    logger.info(f"Found {len(fonts_list)} Google Fonts to sync")

    # Create directory if needed
    fonts_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    # Use ThreadPoolExecutor to download fonts in parallel
    max_workers = min(10, len(fonts_list))  # Limit concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_font_task, font, fonts_path): font
            for font in fonts_list
        }

        with tqdm(total=len(fonts_list), desc="Syncing Google Fonts") as pbar:
            for future in as_completed(futures):
                d, s, f = future.result()
                downloaded += d
                skipped += s
                failed += f
                pbar.update(1)

    logger.info(
        f"Google Fonts sync complete: {downloaded} downloaded, {skipped} skipped, "
        f"{failed} failed"
    )

    return True


def get_icelandic_compatible_fonts(google_fonts_directory: str | None = None):
    """
    Scan system and Google Fonts directories for fonts that support Icelandic characters.

    Args:
        google_fonts_directory: Optional path to Google Fonts directory to include in scan

    Returns:
        List of absolute paths to compatible TTF font files
    """
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

    # Add Google Fonts directory if provided and exists
    if google_fonts_directory:
        google_fonts_path = Path(google_fonts_directory)
        if google_fonts_path.exists() and google_fonts_path.is_dir():
            font_dirs.append(str(google_fonts_path))
            logger.info(f"Including Google Fonts directory: {google_fonts_directory}")

    logger.info(f"Searching for fonts in directories: {font_dirs}")

    available_fonts: list[str] = []
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
