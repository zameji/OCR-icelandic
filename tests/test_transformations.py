#!/usr/bin/env python3
"""Test suite for individual image transformations."""

import os
import pytest
from PIL import Image
from ocr_icelandic.utils import create_image_with_text
from ocr_icelandic.transformations.transformations import (
    blur,
    ink_splashes,
    dusty_paper,
    reverse_bleed_through,
    textured_stains,
    light_reflection,
    shadow_overlay,
    apply_random_transformation,
)
from ocr_icelandic.transformations.rotate import rotate
from ocr_icelandic.transformations.skew import skew
from ocr_icelandic.transformations.perspective import perspective


# Test fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_output_directory():
    """Ensure output directory exists for test images."""
    os.makedirs("local_output/transformations", exist_ok=True)


@pytest.fixture
def sample_image_with_text():
    """Create a sample image with text and bboxes for testing."""
    text = "This is a test.\nMultiple lines.\nFor transformation testing."
    image, fitted_text, bboxes = create_image_with_text(
        text=text,
        image_size=(400, 300),
        font_size=16,
        num_columns=1,
        alignment="left",
        vertical_alignment="top",
        bg_color="white",
        font_color="black",
    )
    return image, fitted_text, bboxes


@pytest.fixture
def sample_image_no_text():
    """Create a blank sample image without text."""
    return Image.new("RGB", (400, 300), color="white")


# Helper function to validate transformation output
def validate_transformation_output(
    result: tuple, original_image: Image.Image, has_bboxes: bool = True
):
    """Validate that transformation returns proper structure."""
    assert isinstance(result, tuple), "Transformation should return a tuple"
    assert len(result) == 3, "Transformation should return 3 elements"

    image, metadata, bboxes = result

    # Validate image
    assert isinstance(image, Image.Image), "First element should be PIL Image"
    assert image.size == original_image.size, "Image size should be preserved"
    assert image.mode in ["RGB", "RGBA"], (
        f"Image mode should be RGB or RGBA, got {image.mode}"
    )

    # Validate metadata
    assert isinstance(metadata, dict), "Second element should be metadata dict"
    assert "transformation" in metadata, "Metadata should contain transformation name"

    # Validate bboxes
    if has_bboxes:
        assert isinstance(bboxes, list), "Third element should be list of bboxes"
    else:
        assert bboxes is None or isinstance(bboxes, list), (
            "Third element should be None or list"
        )

    return True


# ============================================================================
# CONTENT TRANSFORMATIONS TESTS
# ============================================================================


class TestContentTransformations:
    """Test suite for content-based transformations."""

    def test_blur_with_text(self, sample_image_with_text):
        """Test blur transformation with text image."""
        image, _, bboxes = sample_image_with_text
        result = blur(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "blur"
        assert "radius" in metadata
        assert 0.1 <= metadata["radius"] <= 0.5

        # Save for visual inspection
        result[0].save("local_output/transformations/test_blur.png")

    def test_blur_no_text(self, sample_image_no_text):
        """Test blur transformation without text."""
        result = blur(sample_image_no_text, bg_color="white", paragraph_bboxes=None)
        validate_transformation_output(result, sample_image_no_text, has_bboxes=False)

    def test_ink_splashes_with_text(self, sample_image_with_text):
        """Test ink splashes transformation."""
        image, _, bboxes = sample_image_with_text
        result = ink_splashes(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "ink_splashes"
        assert "splashes" in metadata
        assert 3 <= metadata["splashes"] <= 6

        result[0].save("local_output/transformations/test_ink_splashes.png")

    def test_dusty_paper_with_text(self, sample_image_with_text):
        """Test dusty paper transformation."""
        image, _, bboxes = sample_image_with_text
        result = dusty_paper(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "dusty-paper"
        assert "grain_size" in metadata
        assert "intensity" in metadata
        assert 1 <= metadata["grain_size"] <= 3
        assert 0.05 <= metadata["intensity"] <= 0.15

        result[0].save("local_output/transformations/test_dusty_paper.png")

    def test_reverse_bleed_through_with_text(self, sample_image_with_text):
        """Test reverse bleed through transformation."""
        image, _, bboxes = sample_image_with_text
        result = reverse_bleed_through(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "reverse_bleed_through"
        assert "intensity" in metadata
        assert "shift_x" in metadata
        assert "shift_y" in metadata

        result[0].save("local_output/transformations/test_reverse_bleed.png")

    @pytest.mark.skipif(
        not os.path.exists("assets/stains"),
        reason="Stain textures directory not found",
    )
    def test_textured_stains_with_text(self, sample_image_with_text):
        """Test textured stains transformation."""
        image, _, bboxes = sample_image_with_text
        result = textured_stains(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "coffee_stains"
        assert "position" in metadata
        assert "scale_factor" in metadata

        result[0].save("local_output/transformations/test_textured_stains.png")


# ============================================================================
# PERSPECTIVE TRANSFORMATIONS TESTS
# ============================================================================


class TestPerspectiveTransformations:
    """Test suite for geometric/perspective transformations."""

    def test_rotate_with_text(self, sample_image_with_text):
        """Test rotation transformation."""
        image, _, bboxes = sample_image_with_text
        result = rotate(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        transformed_img, metadata, transformed_bboxes = result

        assert metadata["transformation"] == "rotate"
        assert "angle" in metadata
        assert -5 <= metadata["angle"] <= 5

        # Verify bboxes were transformed
        assert len(transformed_bboxes) == len(bboxes)

        result[0].save("local_output/transformations/test_rotate.png")

    def test_rotate_no_text(self, sample_image_no_text):
        """Test rotation without text."""
        result = rotate(sample_image_no_text, bg_color="white", paragraph_bboxes=None)
        validate_transformation_output(result, sample_image_no_text, has_bboxes=False)

    def test_skew_with_text(self, sample_image_with_text):
        """Test skew transformation."""
        image, _, bboxes = sample_image_with_text
        result = skew(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        transformed_img, metadata, transformed_bboxes = result

        assert metadata["transformation"] == "skew"
        assert "skew_factor" in metadata
        assert -0.2 <= metadata["skew_factor"] <= 0.2

        # Verify bboxes were transformed
        assert len(transformed_bboxes) == len(bboxes)

        result[0].save("local_output/transformations/test_skew.png")

    def test_perspective_with_text(self, sample_image_with_text):
        """Test perspective transformation."""
        image, _, bboxes = sample_image_with_text
        result = perspective(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        transformed_img, metadata, transformed_bboxes = result

        assert metadata["transformation"] == "perspective"
        assert "type" in metadata
        assert metadata["type"] in ["book_curve", "camera_angle", "combined"]

        # Verify bboxes were transformed
        assert len(transformed_bboxes) == len(bboxes)

        result[0].save("local_output/transformations/test_perspective.png")


# ============================================================================
# POSTPROCESSING TRANSFORMATIONS TESTS
# ============================================================================


class TestPostprocessingTransformations:
    """Test suite for postprocessing transformations."""

    def test_light_reflection_with_text(self, sample_image_with_text):
        """Test light reflection transformation."""
        image, _, bboxes = sample_image_with_text
        result = light_reflection(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "light_reflection"
        assert "center_x" in metadata
        assert "center_y" in metadata
        assert "ellipse_width" in metadata
        assert "ellipse_height" in metadata

        result[0].save("local_output/transformations/test_light_reflection.png")

    def test_shadow_overlay_with_text(self, sample_image_with_text):
        """Test shadow overlay transformation."""
        image, _, bboxes = sample_image_with_text
        result = shadow_overlay(image, bg_color="white", paragraph_bboxes=bboxes)

        validate_transformation_output(result, image)
        _, metadata, _ = result

        assert metadata["transformation"] == "shadow_overlay"
        assert "edge" in metadata
        assert "max_depth" in metadata
        assert "opacity" in metadata
        assert "blur_radius" in metadata
        assert 0 <= metadata["edge"] <= 3

        result[0].save("local_output/transformations/test_shadow_overlay.png")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestApplyRandomTransformation:
    """Test suite for the main apply_random_transformation function."""

    def test_apply_random_transformation_with_text(self, sample_image_with_text):
        """Test apply_random_transformation with text image."""
        image, _, bboxes = sample_image_with_text
        result = apply_random_transformation(
            image, bg_color="white", paragraph_bboxes=bboxes
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        transformed_img, transformation_metadata, transformed_bboxes = result

        # Validate image
        assert isinstance(transformed_img, Image.Image)
        assert transformed_img.size == image.size

        # Validate metadata list
        assert isinstance(transformation_metadata, list)
        assert len(transformation_metadata) > 0

        # Each metadata entry should have transformation name
        for meta in transformation_metadata:
            assert isinstance(meta, dict)
            assert "transformation" in meta

        # Validate bboxes
        assert isinstance(transformed_bboxes, list)
        assert len(transformed_bboxes) == len(bboxes)

        result[0].save("local_output/transformations/test_random_transformation.png")

    def test_apply_random_transformation_no_text(self, sample_image_no_text):
        """Test apply_random_transformation without text."""
        result = apply_random_transformation(
            sample_image_no_text, bg_color="white", paragraph_bboxes=None
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        transformed_img, transformation_metadata, transformed_bboxes = result

        assert isinstance(transformed_img, Image.Image)
        assert isinstance(transformation_metadata, list)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_image(self):
        """Test transformations on very small image."""
        small_image = Image.new("RGB", (50, 50), color="white")

        # Test a few transformations on small image
        blur_result = blur(small_image, bg_color="white", paragraph_bboxes=None)
        assert blur_result[0].size == (50, 50)

        rotate_result = rotate(small_image, bg_color="white", paragraph_bboxes=None)
        assert rotate_result[0].size == (50, 50)

    def test_large_image(self):
        """Test transformations on larger image."""
        large_image = Image.new("RGB", (2000, 1500), color="white")

        blur_result = blur(large_image, bg_color="white", paragraph_bboxes=None)
        assert blur_result[0].size == (2000, 1500)

    def test_with_rgba_image(self):
        """Test transformations handle RGBA images."""
        rgba_image = Image.new("RGBA", (400, 300), color=(255, 255, 255, 255))

        result = blur(rgba_image, bg_color="white", paragraph_bboxes=None)
        # Should convert to RGB
        assert result[0].mode in ["RGB", "RGBA"]

    def test_empty_bboxes_list(self):
        """Test transformations with empty bboxes list."""
        image = Image.new("RGB", (400, 300), color="white")
        empty_bboxes = []

        result = rotate(image, bg_color="white", paragraph_bboxes=empty_bboxes)
        assert result[2] == []


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================


class TestConsistency:
    """Test consistency of transformation outputs."""

    def test_multiple_applications_produce_different_results(
        self, sample_image_with_text
    ):
        """Test that random transformations produce different results."""
        image, _, bboxes = sample_image_with_text

        # Apply same transformation multiple times
        results = [
            blur(image, bg_color="white", paragraph_bboxes=bboxes) for _ in range(5)
        ]

        # Metadata should vary (random parameters)
        radii = [result[1]["radius"] for result in results]
        # At least some should be different
        assert len(set(radii)) > 1, "Random parameters should vary"

    def test_bbox_preservation_in_content_transforms(self, sample_image_with_text):
        """Test that content transformations preserve bbox coordinates."""
        image, _, original_bboxes = sample_image_with_text

        # Content transformations should not alter bboxes
        content_transforms = [blur, ink_splashes, dusty_paper]

        for transform in content_transforms:
            _, _, result_bboxes = transform(
                image, bg_color="white", paragraph_bboxes=original_bboxes
            )
            assert len(result_bboxes) == len(original_bboxes)
            # Coordinates should be identical
            for orig, res in zip(original_bboxes, result_bboxes):
                assert orig["bbox"] == res["bbox"], (
                    f"{transform.__name__} should preserve bboxes"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
