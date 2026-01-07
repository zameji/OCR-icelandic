#!/usr/bin/env python3
"""Snapshot tests for image transformations to prevent regressions.

These tests use syrupy to capture and compare transformation outputs.
Run with: pytest tests/test_transformation_snapshots.py
Update snapshots: pytest tests/test_transformation_snapshots.py --snapshot-update
"""

import random
import io
from typing import Any
from PIL import Image
import pytest
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.amber import AmberSnapshotExtension
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

from src.ocr_icelandic.utils import create_image_with_text


# ============================================================================
# Custom Syrupy Extension for PIL Images
# ============================================================================


class PNGImageSnapshotExtension(SingleFileSnapshotExtension):
    """Extension to save PIL Images as PNG files in snapshots."""

    _write_mode = WriteMode.BINARY
    _file_extension = "png"

    def serialize(
        self,
        data: Any,
        *,
        exclude: Any = None,
        include: Any = None,
        matcher: Any = None,
    ) -> bytes:
        """Serialize PIL Image to PNG bytes."""
        if isinstance(data, Image.Image):
            buffer = io.BytesIO()
            data.save(buffer, format="PNG")
            return buffer.getvalue()
        raise TypeError(f"Cannot serialize type {type(data)}")


@pytest.fixture
def snapshot_png(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Fixture for PNG image snapshots."""
    return snapshot.use_extension(PNGImageSnapshotExtension)


@pytest.fixture
def snapshot_json(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Fixture for JSON data snapshots (metadata and bboxes)."""
    return snapshot.use_extension(AmberSnapshotExtension)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def base_test_image():
    """Create a standard test image with text for transformation testing."""
    text = """This is a sample document with multiple paragraphs.

Each paragraph will have its own bounding box that needs to be tracked through transformations.

This ensures that the OCR model can learn proper text localization even after geometric distortions."""

    image, fitted_text, bboxes = create_image_with_text(
        text=text,
        image_size=(512, 512),
        font_size=14,
        num_columns=1,
        alignment="left",
        vertical_alignment="center",
        bg_color="white",
        font_color="black",
    )
    return image, fitted_text, bboxes


@pytest.fixture
def multicolumn_test_image():
    """Create a multi-column test image."""
    text = """Column layouts are common in documents like newspapers and academic papers. Each column should be handled properly.

The transformation must preserve the relative positions of text across all columns."""

    image, fitted_text, bboxes = create_image_with_text(
        text=text,
        image_size=(600, 400),
        font_size=12,
        num_columns=2,
        column_gap=30,
        alignment="left",
        vertical_alignment="top",
        bg_color="white",
        font_color="black",
    )
    return image, fitted_text, bboxes


# ============================================================================
# Helper Functions
# ============================================================================


def normalize_bboxes_for_snapshot(bboxes: list[dict]) -> list[dict]:
    """
    Normalize bounding boxes for snapshot comparison.

    Rounds coordinates to reduce floating point comparison issues.
    """
    normalized = []
    for bbox in bboxes:
        bbox_copy = bbox.copy()
        if "bbox" in bbox_copy:
            # Round to 2 decimal places to avoid floating point noise
            bbox_copy["bbox"] = [round(coord, 2) for coord in bbox_copy["bbox"]]
        normalized.append(bbox_copy)
    return normalized


def normalize_metadata_for_snapshot(metadata: dict) -> dict:
    """
    Normalize metadata for snapshot comparison.

    Removes non-deterministic or overly precise values.
    """
    normalized = metadata.copy()

    # Round floating point values to reduce noise
    for key, value in normalized.items():
        if isinstance(value, float):
            normalized[key] = round(value, 3)
        elif isinstance(value, tuple) and all(isinstance(v, float) for v in value):
            normalized[key] = tuple(round(v, 3) for v in value)
        elif isinstance(value, list) and all(isinstance(v, float) for v in value):
            normalized[key] = [round(v, 3) for v in value]

    return normalized


# ============================================================================
# Rotate Transformation Snapshot Tests
# ============================================================================


class TestRotateSnapshots:
    """Snapshot tests for rotation transformation."""

    def test_rotate_small_angle_positive(
        self, base_test_image, snapshot_png, snapshot_json
    ):
        """Test rotation with small positive angle."""
        image, _, bboxes = base_test_image

        # Set seed for deterministic transformation
        random.seed(42)

        # Override random angle with fixed value
        angle = 3.5
        from src.ocr_icelandic.transformations.rotate import _rotate_within_bounds

        rotated_img, meta = _rotate_within_bounds(image, "white", angle)

        # Import transform function to get transformed bboxes
        from src.ocr_icelandic.transformations.rotate import (
            _transform_paragraph_bboxes_for_rotation,
        )

        transformed_bboxes = _transform_paragraph_bboxes_for_rotation(bboxes, meta)

        # Snapshot the image
        assert snapshot_png == rotated_img

        # Snapshot the bounding boxes
        normalized_bboxes = normalize_bboxes_for_snapshot(transformed_bboxes)
        assert snapshot_json == normalized_bboxes

    def test_rotate_small_angle_negative(
        self, base_test_image, snapshot_png, snapshot_json
    ):
        """Test rotation with small negative angle."""
        image, _, bboxes = base_test_image

        random.seed(42)
        angle = -2.8

        from src.ocr_icelandic.transformations.rotate import (
            _rotate_within_bounds,
            _transform_paragraph_bboxes_for_rotation,
        )

        rotated_img, meta = _rotate_within_bounds(image, "white", angle)
        transformed_bboxes = _transform_paragraph_bboxes_for_rotation(bboxes, meta)

        assert snapshot_png == rotated_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_rotate_multicolumn(
        self, multicolumn_test_image, snapshot_png, snapshot_json
    ):
        """Test rotation with multi-column layout."""
        image, _, bboxes = multicolumn_test_image

        random.seed(100)
        angle = 4.2

        from src.ocr_icelandic.transformations.rotate import (
            _rotate_within_bounds,
            _transform_paragraph_bboxes_for_rotation,
        )

        rotated_img, meta = _rotate_within_bounds(image, "white", angle)
        transformed_bboxes = _transform_paragraph_bboxes_for_rotation(bboxes, meta)

        assert snapshot_png == rotated_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)


# ============================================================================
# Skew Transformation Snapshot Tests
# ============================================================================


class TestSkewSnapshots:
    """Snapshot tests for skew transformation."""

    def test_skew_positive_dx(self, base_test_image, snapshot_png, snapshot_json):
        """Test skew with positive horizontal displacement."""
        image, _, bboxes = base_test_image

        random.seed(42)
        dx = 0.15

        from src.ocr_icelandic.transformations.skew import (
            _skew_within_bounds,
            _transform_paragraph_bboxes_for_skew,
        )

        skewed_img, meta = _skew_within_bounds(image, "white", dx)
        transformed_bboxes = _transform_paragraph_bboxes_for_skew(bboxes, meta)

        assert snapshot_png == skewed_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_skew_negative_dx(self, base_test_image, snapshot_png, snapshot_json):
        """Test skew with negative horizontal displacement."""
        image, _, bboxes = base_test_image

        random.seed(42)
        dx = -0.12

        from src.ocr_icelandic.transformations.skew import (
            _skew_within_bounds,
            _transform_paragraph_bboxes_for_skew,
        )

        skewed_img, meta = _skew_within_bounds(image, "white", dx)
        transformed_bboxes = _transform_paragraph_bboxes_for_skew(bboxes, meta)

        assert snapshot_png == skewed_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_skew_small_dx(self, base_test_image, snapshot_png, snapshot_json):
        """Test skew with very small displacement."""
        image, _, bboxes = base_test_image

        random.seed(42)
        dx = 0.05

        from src.ocr_icelandic.transformations.skew import (
            _skew_within_bounds,
            _transform_paragraph_bboxes_for_skew,
        )

        skewed_img, meta = _skew_within_bounds(image, "white", dx)
        transformed_bboxes = _transform_paragraph_bboxes_for_skew(bboxes, meta)

        assert snapshot_png == skewed_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_skew_multicolumn(
        self, multicolumn_test_image, snapshot_png, snapshot_json
    ):
        """Test skew with multi-column layout."""
        image, _, bboxes = multicolumn_test_image

        random.seed(100)
        dx = -0.18

        from src.ocr_icelandic.transformations.skew import (
            _skew_within_bounds,
            _transform_paragraph_bboxes_for_skew,
        )

        skewed_img, meta = _skew_within_bounds(image, "white", dx)
        transformed_bboxes = _transform_paragraph_bboxes_for_skew(bboxes, meta)

        assert snapshot_png == skewed_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)


# ============================================================================
# Perspective Transformation Snapshot Tests
# ============================================================================


class TestPerspectiveSnapshots:
    """Snapshot tests for perspective transformation."""

    def test_perspective_book_curve(self, base_test_image, snapshot_png, snapshot_json):
        """Test perspective transformation with book curve effect."""
        image, _, bboxes = base_test_image

        random.seed(42)

        from src.ocr_icelandic.transformations.perspective import (
            _apply_perspective_distortion,
            _transform_paragraph_bboxes_for_perspective,
        )

        perspective_img, meta = _apply_perspective_distortion(
            image, "white", distortion_type="book_curve"
        )
        transformed_bboxes = _transform_paragraph_bboxes_for_perspective(bboxes, meta)

        assert snapshot_png == perspective_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_perspective_camera_angle_top(
        self, base_test_image, snapshot_png, snapshot_json
    ):
        """Test perspective transformation with camera angle from top."""
        image, _, bboxes = base_test_image

        # Seed to force "top" angle type
        random.seed(10)

        from src.ocr_icelandic.transformations.perspective import (
            _apply_perspective_distortion,
            _transform_paragraph_bboxes_for_perspective,
        )

        perspective_img, meta = _apply_perspective_distortion(
            image, "white", distortion_type="camera_angle"
        )
        transformed_bboxes = _transform_paragraph_bboxes_for_perspective(bboxes, meta)

        assert snapshot_png == perspective_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_perspective_camera_angle_left(
        self, base_test_image, snapshot_png, snapshot_json
    ):
        """Test perspective transformation with camera angle from left."""
        image, _, bboxes = base_test_image

        # Seed to force "left" angle type
        random.seed(15)

        from src.ocr_icelandic.transformations.perspective import (
            _apply_perspective_distortion,
            _transform_paragraph_bboxes_for_perspective,
        )

        perspective_img, meta = _apply_perspective_distortion(
            image, "white", distortion_type="camera_angle"
        )
        transformed_bboxes = _transform_paragraph_bboxes_for_perspective(bboxes, meta)

        assert snapshot_png == perspective_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_perspective_combined(self, base_test_image, snapshot_png, snapshot_json):
        """Test perspective transformation with combined effects."""
        image, _, bboxes = base_test_image

        random.seed(50)

        from src.ocr_icelandic.transformations.perspective import (
            _apply_perspective_distortion,
            _transform_paragraph_bboxes_for_perspective,
        )

        perspective_img, meta = _apply_perspective_distortion(
            image, "white", distortion_type="combined"
        )
        transformed_bboxes = _transform_paragraph_bboxes_for_perspective(bboxes, meta)

        assert snapshot_png == perspective_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_perspective_multicolumn(
        self, multicolumn_test_image, snapshot_png, snapshot_json
    ):
        """Test perspective transformation with multi-column layout."""
        image, _, bboxes = multicolumn_test_image

        random.seed(100)

        from src.ocr_icelandic.transformations.perspective import (
            _apply_perspective_distortion,
            _transform_paragraph_bboxes_for_perspective,
        )

        perspective_img, meta = _apply_perspective_distortion(
            image, "white", distortion_type="book_curve"
        )
        transformed_bboxes = _transform_paragraph_bboxes_for_perspective(bboxes, meta)

        assert snapshot_png == perspective_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCaseSnapshots:
    """Snapshot tests for edge cases."""

    def test_rotate_near_zero(self, base_test_image, snapshot_png, snapshot_json):
        """Test rotation with very small angle (near identity transform)."""
        image, _, bboxes = base_test_image

        angle = 0.1

        from src.ocr_icelandic.transformations.rotate import (
            _rotate_within_bounds,
            _transform_paragraph_bboxes_for_rotation,
        )

        rotated_img, meta = _rotate_within_bounds(image, "white", angle)
        transformed_bboxes = _transform_paragraph_bboxes_for_rotation(bboxes, meta)

        assert snapshot_png == rotated_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_skew_near_zero(self, base_test_image, snapshot_png, snapshot_json):
        """Test skew with very small displacement (near identity transform)."""
        image, _, bboxes = base_test_image

        dx = 0.01

        from src.ocr_icelandic.transformations.skew import (
            _skew_within_bounds,
            _transform_paragraph_bboxes_for_skew,
        )

        skewed_img, meta = _skew_within_bounds(image, "white", dx)
        transformed_bboxes = _transform_paragraph_bboxes_for_skew(bboxes, meta)

        assert snapshot_png == skewed_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)

    def test_small_image_rotate(self, snapshot_png, snapshot_json):
        """Test rotation on small image."""
        small_image = Image.new("RGB", (100, 100), color="white")
        bboxes = [{"bbox": [10, 10, 90, 90]}]

        angle = 5.0

        from src.ocr_icelandic.transformations.rotate import (
            _rotate_within_bounds,
            _transform_paragraph_bboxes_for_rotation,
        )

        rotated_img, meta = _rotate_within_bounds(small_image, "white", angle)
        transformed_bboxes = _transform_paragraph_bboxes_for_rotation(bboxes, meta)

        assert snapshot_png == rotated_img
        assert snapshot_json == normalize_bboxes_for_snapshot(transformed_bboxes)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--snapshot-update"])
