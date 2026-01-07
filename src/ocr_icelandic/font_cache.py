"""SQLite-based caching for font compatibility results.

This module provides caching functionality to avoid re-scanning font files on every run.
Cache entries are automatically invalidated when font files are modified.
"""

import hashlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class FontCompatibilityCache:
    """SQLite-based cache for font compatibility results.

    The cache stores font file metadata (hash, size, mtime) and compatibility
    results per language. Cache entries are automatically invalidated when
    font files change.

    Example:
        >>> cache = FontCompatibilityCache()
        >>> # Check if result is cached
        >>> result = cache.get_cached_compatibility("/path/to/font.ttf", "is")
        >>> if result is None:
        ...     # Not cached, check font and store result
        ...     is_compatible = check_font("/path/to/font.ttf", "is")
        ...     cache.store_compatibility("/path/to/font.ttf", "is", "ÁáÐð", is_compatible)
    """

    SCHEMA_VERSION = 1

    def __init__(self, cache_dir: str = ".fontcache"):
        """Initialize cache with database path.

        Args:
            cache_dir: Directory to store the SQLite database file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "font_compatibility.db"
        self._init_database()

    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Schema version tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Font file metadata
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS font_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime TEXT NOT NULL,
                    last_checked TEXT NOT NULL
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_path ON font_files(file_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_hash ON font_files(file_hash)"
            )

            # Language compatibility results
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS language_compatibility (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    font_file_id INTEGER NOT NULL,
                    language_code TEXT NOT NULL,
                    character_set TEXT NOT NULL,
                    is_compatible INTEGER NOT NULL,
                    checked_at TEXT NOT NULL,
                    FOREIGN KEY (font_file_id) REFERENCES font_files(id) ON DELETE CASCADE,
                    UNIQUE(font_file_id, language_code)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_language ON language_compatibility(language_code)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_compatible ON language_compatibility(is_compatible)"
            )

            # Scan history for performance monitoring
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_directory TEXT NOT NULL,
                    language_code TEXT NOT NULL,
                    fonts_found INTEGER NOT NULL,
                    compatible_fonts INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    cache_misses INTEGER NOT NULL DEFAULT 0,
                    scan_duration_seconds REAL NOT NULL,
                    scanned_at TEXT NOT NULL
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scan_dir ON scan_history(scan_directory)"
            )

            # Insert schema version if not exists
            conn.execute(
                """
                INSERT OR IGNORE INTO schema_version (version, created_at)
                VALUES (?, ?)
            """,
                (self.SCHEMA_VERSION, datetime.now().isoformat()),
            )

            conn.commit()

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file (first 8KB only for speed).

        Args:
            file_path: Path to the font file

        Returns:
            Hexadecimal SHA256 hash string
        """
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read(8192)).hexdigest()

    def get_cached_compatibility(
        self, font_path: str, language_code: str
    ) -> Optional[bool]:
        """Check if font compatibility is cached and still valid.

        Validates cache entry by comparing file hash, size, and modification time.
        Returns None if not cached or if the file has changed.

        Args:
            font_path: Absolute path to the font file
            language_code: ISO 639-1 language code

        Returns:
            True if compatible, False if incompatible, None if not cached/stale

        Example:
            >>> cache = FontCompatibilityCache()
            >>> result = cache.get_cached_compatibility("/usr/share/fonts/font.ttf", "is")
            >>> if result is not None:
            ...     print(f"Cache hit: {result}")
        """
        font_path_obj = Path(font_path)
        if not font_path_obj.exists():
            return None

        try:
            # Get current file metadata
            stat = font_path_obj.stat()
            current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            current_size = stat.st_size
            current_hash = self._compute_file_hash(font_path_obj)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Check if we have a cached result
                result = conn.execute(
                    """
                    SELECT
                        ff.file_hash,
                        ff.file_size,
                        ff.file_mtime,
                        lc.is_compatible
                    FROM font_files ff
                    JOIN language_compatibility lc ON ff.id = lc.font_file_id
                    WHERE ff.file_path = ? AND lc.language_code = ?
                """,
                    (str(font_path), language_code),
                ).fetchone()

                if result:
                    # Verify file hasn't changed
                    if (
                        result["file_hash"] == current_hash
                        and result["file_size"] == current_size
                        and result["file_mtime"] == current_mtime
                    ):
                        return bool(result["is_compatible"])

            return None

        except (OSError, sqlite3.Error):
            # If any error occurs, treat as cache miss
            return None

    def store_compatibility(
        self,
        font_path: str,
        language_code: str,
        character_set: str,
        is_compatible: bool,
    ) -> None:
        """Store font compatibility result in cache.

        Args:
            font_path: Absolute path to the font file
            language_code: ISO 639-1 language code
            character_set: String of characters that were checked
            is_compatible: Whether the font supports the character set

        Example:
            >>> cache = FontCompatibilityCache()
            >>> cache.store_compatibility(
            ...     "/usr/share/fonts/font.ttf",
            ...     "is",
            ...     "ÁáÐðÉé",
            ...     True
            ... )
        """
        font_path_obj = Path(font_path)
        if not font_path_obj.exists():
            return

        try:
            stat = font_path_obj.stat()
            file_hash = self._compute_file_hash(font_path_obj)
            file_size = stat.st_size
            file_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            now = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                # Insert or update font file record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO font_files
                    (file_path, file_hash, file_size, file_mtime, last_checked)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (str(font_path), file_hash, file_size, file_mtime, now),
                )

                # Get font file ID
                font_file_id = conn.execute(
                    "SELECT id FROM font_files WHERE file_path = ?", (str(font_path),)
                ).fetchone()[0]

                # Insert or update compatibility record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO language_compatibility
                    (font_file_id, language_code, character_set, is_compatible, checked_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        font_file_id,
                        language_code,
                        character_set,
                        int(is_compatible),
                        now,
                    ),
                )

                conn.commit()

        except (OSError, sqlite3.Error):
            # Silently fail on cache storage errors
            pass

    def get_compatible_fonts(self, language_code: str) -> list[str]:
        """Get all cached compatible fonts for a language.

        Args:
            language_code: ISO 639-1 language code

        Returns:
            List of absolute font file paths that are compatible

        Example:
            >>> cache = FontCompatibilityCache()
            >>> fonts = cache.get_compatible_fonts("is")
            >>> len(fonts)
            42
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute(
                    """
                    SELECT ff.file_path
                    FROM font_files ff
                    JOIN language_compatibility lc ON ff.id = lc.font_file_id
                    WHERE lc.language_code = ? AND lc.is_compatible = 1
                    ORDER BY ff.file_path
                """,
                    (language_code,),
                ).fetchall()

                return [row[0] for row in results]

        except sqlite3.Error:
            return []

    def record_scan(
        self,
        scan_directory: str,
        language_code: str,
        fonts_found: int,
        compatible_fonts: int,
        duration_seconds: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ) -> None:
        """Record scan statistics for performance monitoring.

        Args:
            scan_directory: Directory that was scanned
            language_code: ISO 639-1 language code
            fonts_found: Total number of fonts found
            compatible_fonts: Number of compatible fonts
            duration_seconds: Time taken for the scan
            cache_hits: Number of fonts found in cache
            cache_misses: Number of fonts that needed checking

        Example:
            >>> cache = FontCompatibilityCache()
            >>> cache.record_scan(
            ...     "/usr/share/fonts",
            ...     "is",
            ...     fonts_found=100,
            ...     compatible_fonts=42,
            ...     duration_seconds=0.5,
            ...     cache_hits=95,
            ...     cache_misses=5
            ... )
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO scan_history
                    (scan_directory, language_code, fonts_found, compatible_fonts,
                     cache_hits, cache_misses, scan_duration_seconds, scanned_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        scan_directory,
                        language_code,
                        fonts_found,
                        compatible_fonts,
                        cache_hits,
                        cache_misses,
                        duration_seconds,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()

        except sqlite3.Error:
            # Silently fail on recording errors
            pass

    def clear_cache(self, language_code: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            language_code: If specified, clear only entries for this language.
                          If None, clear all cache entries.

        Example:
            >>> cache = FontCompatibilityCache()
            >>> cache.clear_cache("is")  # Clear only Icelandic
            >>> cache.clear_cache()      # Clear all
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if language_code:
                    # Clear only specific language
                    conn.execute(
                        """
                        DELETE FROM language_compatibility
                        WHERE language_code = ?
                    """,
                        (language_code,),
                    )
                    # Clean up orphaned font_files entries
                    conn.execute(
                        """
                        DELETE FROM font_files
                        WHERE id NOT IN (
                            SELECT DISTINCT font_file_id FROM language_compatibility
                        )
                    """
                    )
                else:
                    # Clear everything
                    conn.execute("DELETE FROM language_compatibility")
                    conn.execute("DELETE FROM font_files")
                    conn.execute("DELETE FROM scan_history")

                conn.commit()

        except sqlite3.Error:
            pass

    def get_cache_stats(self) -> dict:
        """Get cache statistics for debugging and monitoring.

        Returns:
            Dictionary with cache statistics including:
            - total_fonts: Number of font files in cache
            - total_checks: Total compatibility checks cached
            - languages_cached: Number of different languages cached
            - database_size_kb: Size of the database file in KB
            - recent_scans: List of 5 most recent scan records

        Example:
            >>> cache = FontCompatibilityCache()
            >>> stats = cache.get_cache_stats()
            >>> print(f"Fonts cached: {stats['total_fonts']}")
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                total_fonts = conn.execute(
                    "SELECT COUNT(*) as count FROM font_files"
                ).fetchone()["count"]

                total_checks = conn.execute(
                    "SELECT COUNT(*) as count FROM language_compatibility"
                ).fetchone()["count"]

                languages_cached = conn.execute(
                    "SELECT COUNT(DISTINCT language_code) as count FROM language_compatibility"
                ).fetchone()["count"]

                # Get recent scan history
                recent_scans = conn.execute(
                    """
                    SELECT
                        scan_directory,
                        language_code,
                        fonts_found,
                        compatible_fonts,
                        cache_hits,
                        cache_misses,
                        scan_duration_seconds,
                        scanned_at
                    FROM scan_history
                    ORDER BY scanned_at DESC
                    LIMIT 5
                """
                ).fetchall()

                recent_scans_list = [
                    {
                        "directory": row["scan_directory"],
                        "language": row["language_code"],
                        "fonts_found": row["fonts_found"],
                        "compatible": row["compatible_fonts"],
                        "cache_hits": row["cache_hits"],
                        "cache_misses": row["cache_misses"],
                        "duration_seconds": row["scan_duration_seconds"],
                        "scanned_at": row["scanned_at"],
                    }
                    for row in recent_scans
                ]

                db_size_kb = self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0

                return {
                    "total_fonts": total_fonts,
                    "total_checks": total_checks,
                    "languages_cached": languages_cached,
                    "database_size_kb": round(db_size_kb, 2),
                    "recent_scans": recent_scans_list,
                }

        except (sqlite3.Error, OSError):
            return {
                "total_fonts": 0,
                "total_checks": 0,
                "languages_cached": 0,
                "database_size_kb": 0,
                "recent_scans": [],
            }

    def vacuum(self) -> None:
        """Optimize database by running VACUUM command.

        This reclaims unused space and optimizes the database file.
        Should be run periodically if many entries are deleted.

        Example:
            >>> cache = FontCompatibilityCache()
            >>> cache.vacuum()
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()
        except sqlite3.Error:
            pass
