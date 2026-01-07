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


def get_compatible_fonts(
    language_code: str,
    use_cache: bool = True,
    cache_dir: str = ".fontcache",
    google_fonts_directory: str | None = None,
) -> list[str]:
    """
    Scan system and Google Fonts directories for fonts compatible with a language.

    This function supports multiple languages via ISO 639-1 codes and uses SQLite
    caching to avoid re-scanning fonts on subsequent runs. Cache entries are
    automatically invalidated when font files are modified.

    Args:
        language_code: ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code
                      (e.g., "is" for Icelandic, "de" for German)
        use_cache: Whether to use caching (default: True)
        cache_dir: Directory to store cache database (default: ".fontcache")
        google_fonts_directory: Optional path to Google Fonts directory to include

    Returns:
        List of absolute paths to compatible TTF font files

    Raises:
        NotImplementedError: If the language code is not supported

    Example:
        >>> # Get Icelandic-compatible fonts with caching
        >>> fonts = get_compatible_fonts("is")
        >>> len(fonts)
        42

        >>> # Get German fonts without caching
        >>> fonts = get_compatible_fonts("de", use_cache=False)

        >>> # Add custom language and get fonts
        >>> from ocr_icelandic.language_support import LanguageRegistry, LanguageCharacterSet
        >>> LanguageRegistry.add_custom_language(LanguageCharacterSet(
        ...     iso_639_1="ja",
        ...     iso_639_3="jpn",
        ...     name_english="Japanese",
        ...     name_native="日本語",
        ...     special_characters="あいうえお"
        ... ))
        >>> fonts = get_compatible_fonts("ja")
    """
    import time
    from ocr_icelandic.font_cache import FontCompatibilityCache
    from ocr_icelandic.language_support import LanguageRegistry

    random.seed(42)  # For reproducibility

    # Get language character set (raises NotImplementedError if not supported)
    language = LanguageRegistry.get_language(language_code)
    characters_to_check = language.special_characters

    logger.info(
        f"Scanning for {language.name_english} ({language_code}) compatible fonts"
    )
    logger.info(f"Character set: {characters_to_check}")

    # Initialize cache if enabled
    cache = FontCompatibilityCache(cache_dir) if use_cache else None

    # Determine font directories based on OS
    current_os = sys.platform
    font_dirs = []

    # macOS
    if current_os.startswith("darwin"):
        font_dirs = [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
        ]
    # Linux
    if current_os.startswith("linux"):
        font_dirs += [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ]
    # Windows
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

    start_time = time.time()
    available_fonts: list[str] = []
    cache_hits = 0
    cache_misses = 0
    fonts_found = 0

    for font_dir in tqdm(font_dirs, desc=f"Scanning font directories ({language_code})"):
        font_path = Path(font_dir)
        if not font_path.exists() or not font_path.is_dir():
            continue

        for font_file in font_path.rglob("*.[tT][tT][fF]"):
            fonts_found += 1
            font_file_str = str(font_file)

            # Check cache first if enabled
            if cache:
                cached_result = cache.get_cached_compatibility(
                    font_file_str, language_code
                )
                if cached_result is not None:
                    cache_hits += 1
                    if cached_result:
                        available_fonts.append(font_file_str)
                    continue

            # Cache miss - need to check the font
            cache_misses += 1
            is_compatible = False

            # Check if font supports at least one character from the set
            for char in characters_to_check:
                if check_font_supports_char(font_file, char):
                    is_compatible = True
                    available_fonts.append(font_file_str)
                    break

            # Store result in cache
            if cache:
                cache.store_compatibility(
                    font_file_str, language_code, characters_to_check, is_compatible
                )

    duration = time.time() - start_time

    # Record scan statistics
    if cache:
        cache.record_scan(
            scan_directory=", ".join(font_dirs),
            language_code=language_code,
            fonts_found=fonts_found,
            compatible_fonts=len(available_fonts),
            duration_seconds=duration,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    logger.info(
        f"Found {len(available_fonts)} {language.name_english}-compatible fonts "
        f"in {duration:.2f}s"
    )
    if cache:
        logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")

    return available_fonts


def get_icelandic_compatible_fonts(google_fonts_directory: str | None = None):
    """
    Scan system and Google Fonts directories for fonts that support Icelandic characters.

    .. deprecated:: 0.2.0
        Use :func:`get_compatible_fonts` with language_code="is" instead.
        This function is maintained for backward compatibility but will be removed
        in a future version.

    Args:
        google_fonts_directory: Optional path to Google Fonts directory to include in scan

    Returns:
        List of absolute paths to compatible TTF font files

    Example:
        >>> # Old way (deprecated)
        >>> fonts = get_icelandic_compatible_fonts()

        >>> # New way (recommended)
        >>> fonts = get_compatible_fonts("is")
    """
    import warnings

    warnings.warn(
        "get_icelandic_compatible_fonts() is deprecated. "
        "Use get_compatible_fonts(language_code='is') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return get_compatible_fonts(
        language_code="is",
        use_cache=True,
        cache_dir=".fontcache",
        google_fonts_directory=google_fonts_directory,
    )
