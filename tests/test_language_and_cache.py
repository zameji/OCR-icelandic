#!/usr/bin/env python3
"""Test suite for language support and font caching functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from src.ocr_icelandic.font_cache import FontCompatibilityCache
from src.ocr_icelandic.language_support import (
    LanguageCharacterSet,
    LanguageRegistry,
)


# ============================================================================
# Language Support Tests
# ============================================================================


class TestLanguageCharacterSet:
    """Tests for LanguageCharacterSet dataclass."""

    def test_create_valid_language_set(self):
        """Test creating a valid language character set."""
        lang = LanguageCharacterSet(
            iso_639_1="is",
            iso_639_3="isl",
            name_english="Icelandic",
            name_native="Íslenska",
            special_characters="ÁáÐð",
            description="Test description",
        )

        assert lang.iso_639_1 == "is"
        assert lang.iso_639_3 == "isl"
        assert lang.name_english == "Icelandic"
        assert lang.special_characters == "ÁáÐð"

    def test_invalid_iso_639_1_code(self):
        """Test that invalid ISO 639-1 code raises error."""
        with pytest.raises(ValueError, match="ISO 639-1 code must be 2 characters"):
            LanguageCharacterSet(
                iso_639_1="isl",  # Too long
                iso_639_3="isl",
                name_english="Icelandic",
                name_native="Íslenska",
                special_characters="ÁáÐð",
            )

    def test_invalid_iso_639_3_code(self):
        """Test that invalid ISO 639-3 code raises error."""
        with pytest.raises(ValueError, match="ISO 639-3 code must be 3 characters"):
            LanguageCharacterSet(
                iso_639_1="is",
                iso_639_3="is",  # Too short
                name_english="Icelandic",
                name_native="Íslenska",
                special_characters="ÁáÐð",
            )

    def test_empty_special_characters(self):
        """Test that empty special characters raises error."""
        with pytest.raises(ValueError, match="special_characters cannot be empty"):
            LanguageCharacterSet(
                iso_639_1="is",
                iso_639_3="isl",
                name_english="Icelandic",
                name_native="Íslenska",
                special_characters="",
            )


class TestLanguageRegistry:
    """Tests for LanguageRegistry class."""

    def test_get_icelandic_by_iso_639_1(self):
        """Test getting Icelandic language by ISO 639-1 code."""
        lang = LanguageRegistry.get_language("is")

        assert lang.iso_639_1 == "is"
        assert lang.iso_639_3 == "isl"
        assert lang.name_english == "Icelandic"
        assert "Ð" in lang.special_characters
        assert "Þ" in lang.special_characters

    def test_get_icelandic_by_iso_639_3(self):
        """Test getting Icelandic language by ISO 639-3 code."""
        lang = LanguageRegistry.get_language("isl")

        assert lang.iso_639_1 == "is"
        assert lang.iso_639_3 == "isl"

    def test_get_german(self):
        """Test getting German language."""
        lang = LanguageRegistry.get_language("de")

        assert lang.iso_639_1 == "de"
        assert lang.name_english == "German"
        assert "ä" in lang.special_characters
        assert "ß" in lang.special_characters

    def test_get_unsupported_language(self):
        """Test that unsupported language raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Language 'zz' is not supported"):
            LanguageRegistry.get_language("zz")

    def test_list_supported_languages(self):
        """Test listing all supported languages."""
        languages = LanguageRegistry.list_supported_languages()

        assert isinstance(languages, list)
        assert "is" in languages
        assert "de" in languages
        assert "fr" in languages
        assert len(languages) >= 10  # We have at least 10 pre-configured languages

    def test_get_language_info(self):
        """Test getting detailed language information."""
        info = LanguageRegistry.get_language_info()

        assert isinstance(info, list)
        assert len(info) > 0

        # Check first language entry structure
        first = info[0]
        assert "iso_639_1" in first
        assert "iso_639_3" in first
        assert "name_english" in first
        assert "name_native" in first
        assert "description" in first

    def test_add_custom_language(self):
        """Test adding a custom language at runtime."""
        custom_lang = LanguageCharacterSet(
            iso_639_1="ja",
            iso_639_3="jpn",
            name_english="Japanese",
            name_native="日本語",
            special_characters="あいうえお",
            description="Japanese hiragana",
        )

        LanguageRegistry.add_custom_language(custom_lang)

        # Verify language was added
        lang = LanguageRegistry.get_language("ja")
        assert lang.name_english == "Japanese"
        assert "あ" in lang.special_characters

        # Clean up
        LanguageRegistry.remove_language("ja")

    def test_add_duplicate_language(self):
        """Test that adding duplicate language raises error."""
        with pytest.raises(ValueError, match="already exists"):
            LanguageRegistry.add_custom_language(
                LanguageCharacterSet(
                    iso_639_1="is",  # Already exists
                    iso_639_3="xxx",
                    name_english="Test",
                    name_native="Test",
                    special_characters="abc",
                )
            )

    def test_remove_language(self):
        """Test removing a language from registry."""
        # Add a custom language
        custom_lang = LanguageCharacterSet(
            iso_639_1="zz",
            iso_639_3="zzz",
            name_english="Test Language",
            name_native="Test",
            special_characters="abc",
        )
        LanguageRegistry.add_custom_language(custom_lang)

        # Verify it was added
        assert "zz" in LanguageRegistry.list_supported_languages()

        # Remove it
        LanguageRegistry.remove_language("zz")

        # Verify it was removed
        assert "zz" not in LanguageRegistry.list_supported_languages()

    def test_remove_nonexistent_language(self):
        """Test that removing nonexistent language raises error."""
        with pytest.raises(KeyError, match="not found"):
            LanguageRegistry.remove_language("nonexistent")


# ============================================================================
# Font Cache Tests
# ============================================================================


class TestFontCompatibilityCache:
    """Tests for FontCompatibilityCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance with temporary directory."""
        return FontCompatibilityCache(cache_dir=temp_cache_dir)

    @pytest.fixture
    def temp_font_file(self):
        """Create a temporary font file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
            tmp.write(b"fake font data for testing")
            tmp.flush()
            yield tmp.name
        # Cleanup
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization creates database."""
        cache = FontCompatibilityCache(cache_dir=temp_cache_dir)

        db_path = Path(temp_cache_dir) / "font_compatibility.db"
        assert db_path.exists()

    def test_store_and_retrieve_compatibility(self, cache, temp_font_file):
        """Test storing and retrieving font compatibility result."""
        # Store a compatibility result
        cache.store_compatibility(
            font_path=temp_font_file,
            language_code="is",
            character_set="ÁáÐð",
            is_compatible=True,
        )

        # Retrieve it
        result = cache.get_cached_compatibility(temp_font_file, "is")

        assert result is True

    def test_cache_miss_for_new_font(self, cache, temp_font_file):
        """Test that uncached font returns None."""
        result = cache.get_cached_compatibility(temp_font_file, "is")

        assert result is None

    def test_cache_miss_for_different_language(self, cache, temp_font_file):
        """Test that different language is not cached."""
        # Store for Icelandic
        cache.store_compatibility(
            font_path=temp_font_file,
            language_code="is",
            character_set="ÁáÐð",
            is_compatible=True,
        )

        # Try to retrieve for German
        result = cache.get_cached_compatibility(temp_font_file, "de")

        assert result is None

    def test_store_incompatible_font(self, cache, temp_font_file):
        """Test storing incompatible font result."""
        cache.store_compatibility(
            font_path=temp_font_file,
            language_code="is",
            character_set="ÁáÐð",
            is_compatible=False,
        )

        result = cache.get_cached_compatibility(temp_font_file, "is")

        assert result is False

    def test_get_compatible_fonts(self, cache, temp_font_file):
        """Test retrieving all compatible fonts for a language."""
        # Store some fonts
        cache.store_compatibility(
            font_path=temp_font_file,
            language_code="is",
            character_set="ÁáÐð",
            is_compatible=True,
        )

        # Get all compatible fonts
        fonts = cache.get_compatible_fonts("is")

        assert temp_font_file in fonts
        assert len(fonts) == 1

    def test_record_scan_statistics(self, cache):
        """Test recording scan statistics."""
        cache.record_scan(
            scan_directory="/usr/share/fonts",
            language_code="is",
            fonts_found=100,
            compatible_fonts=42,
            duration_seconds=5.5,
            cache_hits=90,
            cache_misses=10,
        )

        # Verify statistics were recorded
        stats = cache.get_cache_stats()
        assert len(stats["recent_scans"]) == 1
        assert stats["recent_scans"][0]["fonts_found"] == 100
        assert stats["recent_scans"][0]["compatible"] == 42

    def test_clear_cache_all(self, cache, temp_font_file):
        """Test clearing all cache entries."""
        # Store some data
        cache.store_compatibility(temp_font_file, "is", "ÁáÐð", True)

        # Clear cache
        cache.clear_cache()

        # Verify it's gone
        result = cache.get_cached_compatibility(temp_font_file, "is")
        assert result is None

    def test_clear_cache_for_language(self, cache, temp_font_file):
        """Test clearing cache for specific language only."""
        # Store for two languages
        cache.store_compatibility(temp_font_file, "is", "ÁáÐð", True)
        cache.store_compatibility(temp_font_file, "de", "äöü", True)

        # Clear only Icelandic
        cache.clear_cache("is")

        # Icelandic should be gone
        assert cache.get_cached_compatibility(temp_font_file, "is") is None

        # German should still exist
        assert cache.get_cached_compatibility(temp_font_file, "de") is True

    def test_get_cache_stats(self, cache, temp_font_file):
        """Test getting cache statistics."""
        # Add some data
        cache.store_compatibility(temp_font_file, "is", "ÁáÐð", True)
        cache.store_compatibility(temp_font_file, "de", "äöü", False)

        stats = cache.get_cache_stats()

        assert stats["total_fonts"] == 1
        assert stats["total_checks"] == 2
        assert stats["languages_cached"] == 2
        assert stats["database_size_kb"] > 0

    def test_vacuum_database(self, cache, temp_font_file):
        """Test database vacuum operation."""
        # Add and remove data
        cache.store_compatibility(temp_font_file, "is", "ÁáÐð", True)
        cache.clear_cache()

        # Vacuum should not raise error
        cache.vacuum()

    def test_cache_invalidation_on_file_modification(self, cache):
        """Test that cache is invalidated when file is modified."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
            tmp.write(b"original content")
            tmp_name = tmp.name

        try:
            # Store compatibility
            cache.store_compatibility(tmp_name, "is", "ÁáÐð", True)

            # Verify it's cached
            assert cache.get_cached_compatibility(tmp_name, "is") is True

            # Modify the file
            import time

            time.sleep(0.1)  # Ensure mtime changes
            with open(tmp_name, "wb") as f:
                f.write(b"modified content")

            # Cache should be invalidated (returns None)
            result = cache.get_cached_compatibility(tmp_name, "is")
            assert result is None

        finally:
            os.unlink(tmp_name)

    def test_cache_with_nonexistent_file(self, cache):
        """Test cache operations with nonexistent file."""
        fake_path = "/nonexistent/font.ttf"

        # Should not crash
        cache.store_compatibility(fake_path, "is", "ÁáÐð", True)

        # Should return None
        result = cache.get_cached_compatibility(fake_path, "is")
        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestLanguageAndCacheIntegration:
    """Integration tests combining language support and caching."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cache_multiple_languages(self, temp_cache_dir):
        """Test caching compatibility for multiple languages."""
        cache = FontCompatibilityCache(cache_dir=temp_cache_dir)

        # Create temporary font file
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
            tmp.write(b"fake font")
            tmp_name = tmp.name

        try:
            # Get character sets from language registry
            icelandic = LanguageRegistry.get_language("is")
            german = LanguageRegistry.get_language("de")

            # Store compatibility for multiple languages
            cache.store_compatibility(
                tmp_name, "is", icelandic.special_characters, True
            )
            cache.store_compatibility(tmp_name, "de", german.special_characters, False)

            # Verify both are cached correctly
            assert cache.get_cached_compatibility(tmp_name, "is") is True
            assert cache.get_cached_compatibility(tmp_name, "de") is False

        finally:
            os.unlink(tmp_name)

    def test_language_not_implemented_error(self):
        """Test that unsupported language raises NotImplementedError as requested."""
        with pytest.raises(NotImplementedError, match="Language.*is not supported"):
            LanguageRegistry.get_language("unsupported_code")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
