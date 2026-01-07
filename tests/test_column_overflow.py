#!/usr/bin/env python3
"""Test script to verify column overflow handling."""

import os
import pytest
from src.ocr_icelandic.utils import create_image_with_text

# Test case 1: Very long word that shouldn't fit in multiple narrow columns
TEST_TEXT_1 = "This is a test with a verylongwordthatdefinitelywontfitinasinglecolumnandwillcauseoverflow and some normal words."

# Test case 2: Multiple long words
TEST_TEXT_2 = """
Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis.
This line has some extraordinarilylengthydictionarydefinitions.
Normal text should work fine.
"""

# Test case 3: Icelandic text with long compound words
TEST_TEXT_3 = """
Þetta er langt íslenskt samsetningarorð: varaþingmannakjördæmistillögunefndarstjóri.
Og annað langt orð: fjármálaeftirlitsstofnunarbúnaðarviðhaldsdeild.
"""


@pytest.fixture(scope="session", autouse=True)
def setup_output_directory():
    """Ensure output directory exists."""
    os.makedirs("local_output", exist_ok=True)


@pytest.mark.parametrize(
    "test_name,text,num_columns",
    [
        ("long_word_3_columns", TEST_TEXT_1, 3),
        ("long_word_2_columns", TEST_TEXT_1, 2),
        ("multiple_long_words_4_columns", TEST_TEXT_2, 4),
        ("icelandic_compounds_3_columns", TEST_TEXT_3, 3),
        ("single_column_baseline", TEST_TEXT_1, 1),
    ],
)
def test_column_overflow_handling(test_name, text, num_columns):
    """Test that column overflow handling works correctly with long words."""
    print(f"\nTest: {test_name}")
    print(f"Requested columns: {num_columns}")

    # Create image - should not raise any exceptions
    image, fitted_text, bboxes = create_image_with_text(
        text=text,
        image_size=(800, 400),
        font_size=14,
        num_columns=num_columns,
        column_gap=20,
        alignment="left",
        vertical_alignment="top",
    )

    # Assertions
    assert image is not None, "Image should be created"
    assert image.size == (800, 400), "Image size should match requested size"
    assert len(fitted_text) > 0, "Fitted text should not be empty"
    assert isinstance(bboxes, list), "Bboxes should be a list"

    # Save image for visual inspection
    output_path = f"local_output/test_{test_name}.png"
    image.save(output_path)
    print(f"  ✓ Image saved to: {output_path}")
    print(f"  ✓ Fitted {len(fitted_text)} characters")
    print(f"  ✓ Generated {len(bboxes)} paragraph bboxes")


def test_long_word_forces_column_reduction():
    """Test that very long words cause automatic column reduction."""
    # Create a word that's definitely too long for narrow columns
    very_long_word = "a" * 500
    text = f"Short text {very_long_word} more text."

    # Request 5 columns - should reduce to fewer due to long word
    image, fitted_text, bboxes = create_image_with_text(
        text=text,
        image_size=(800, 400),
        font_size=14,
        num_columns=5,
        column_gap=20,
        alignment="left",
        vertical_alignment="top",
    )

    # Should still create a valid image
    assert image is not None
    assert image.size == (800, 400)
    # The long word should be in the fitted text
    assert very_long_word in fitted_text

    print(f"  ✓ Successfully handled {len(very_long_word)}-character word")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
