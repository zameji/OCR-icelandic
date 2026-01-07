# Test Suite Documentation

This directory contains the test suite for the OCR-icelandic project.

## Test Files

### `test_transformations.py`
Comprehensive unit tests for all image transformation functions. Tests validate:
- Transformation output structure (image, metadata, bounding boxes)
- Metadata correctness (angles, factors, parameters)
- Image size preservation
- Bounding box transformation accuracy
- Edge cases (small/large images, empty bboxes, RGBA handling)

**Run these tests:**
```bash
pytest tests/test_transformations.py -v
```

### `test_column_overflow.py`
Tests for column layout overflow handling with long words and compound words.

**Run these tests:**
```bash
pytest tests/test_column_overflow.py -v
```

### `test_transformation_snapshots.py` ⭐ NEW
**Snapshot tests** for catching visual regressions in transformation outputs.

## Snapshot Testing

Snapshot tests prevent regressions by comparing current transformation outputs against known-good reference outputs (snapshots). When transformations are modified, these tests will fail if the output changes, helping you catch unintended side effects.

### What Are Snapshot Tests?

Snapshot tests capture the output of your code and save it as a "snapshot" file. On subsequent test runs, the output is compared against the saved snapshot. If they don't match, the test fails.

**Benefits:**
- **Catches visual regressions** - Detects unexpected changes in image transformations
- **Detects bbox transformation bugs** - Ensures bounding boxes transform correctly
- **Low maintenance** - No need to manually specify expected outputs
- **Easy to review changes** - Visual diffs show exactly what changed

### Running Snapshot Tests

#### First Time Setup

Install test dependencies (if not already installed):
```bash
uv sync --group dev
# or
pip install -e ".[dev]"
```

#### Run Snapshot Tests

```bash
# Run all snapshot tests
pytest tests/test_transformation_snapshots.py -v

# Run specific test class
pytest tests/test_transformation_snapshots.py::TestRotateSnapshots -v

# Run specific test
pytest tests/test_transformation_snapshots.py::TestRotateSnapshots::test_rotate_small_angle_positive -v
```

#### Generate Initial Snapshots

The first time you run the tests, you need to generate the initial snapshots:

```bash
pytest tests/test_transformation_snapshots.py --snapshot-update
```

This creates snapshot files in `tests/__snapshots__/` directory.

#### Update Snapshots After Code Changes

If you intentionally change transformation behavior, update the snapshots:

```bash
# Update all snapshots
pytest tests/test_transformation_snapshots.py --snapshot-update

# Update specific test snapshots
pytest tests/test_transformation_snapshots.py::TestRotateSnapshots --snapshot-update
```

**⚠️ Warning:** Only update snapshots after carefully reviewing the changes to ensure they're intentional!

### Understanding Snapshot Test Output

#### Test Passes ✅
```
tests/test_transformation_snapshots.py::TestRotateSnapshots::test_rotate_small_angle_positive PASSED
```
The transformation output matches the saved snapshot.

#### Test Fails ❌
```
tests/test_transformation_snapshots.py::TestRotateSnapshots::test_rotate_small_angle_positive FAILED

AssertionError: Snapshot does not match
```
The transformation output has changed. This could indicate:
1. **Bug introduced** - Unintended change in transformation logic
2. **Intentional change** - You modified the transformation algorithm
3. **Platform differences** - Different OS/library versions producing slightly different output

**Next Steps:**
1. Review the diff in the test output
2. Check if the change was intentional
3. If intentional: `pytest tests/test_transformation_snapshots.py --snapshot-update`
4. If unintended: Fix the bug in the transformation code

### Snapshot File Structure

Snapshots are stored in `tests/__snapshots__/` with the following structure:

```
tests/__snapshots__/
├── test_transformation_snapshots/
│   ├── TestRotateSnapshots.test_rotate_small_angle_positive.png
│   ├── TestRotateSnapshots.test_rotate_small_angle_positive.1.amber
│   ├── TestRotateSnapshots.test_rotate_small_angle_negative.png
│   ├── TestRotateSnapshots.test_rotate_small_angle_negative.1.amber
│   └── ... (more snapshots)
```

- **`.png` files** - Visual snapshots of transformed images
- **`.amber` files** - JSON snapshots of bounding box coordinates

### What Tests Are Included?

#### Rotate Transformation Tests
- Small positive angle rotation (+3.5°)
- Small negative angle rotation (-2.8°)
- Multi-column layout rotation (+4.2°)
- Near-zero rotation (edge case)

#### Skew Transformation Tests
- Positive horizontal skew (+0.15)
- Negative horizontal skew (-0.12)
- Small skew (+0.05)
- Multi-column layout skew
- Near-zero skew (edge case)

#### Perspective Transformation Tests
- Book curve effect
- Camera angle from top
- Camera angle from left
- Combined effects
- Multi-column layout perspective

#### Edge Cases
- Very small transformations (near identity)
- Small images (100×100)
- Empty bounding boxes

### Deterministic Testing

All snapshot tests use **fixed random seeds** to ensure deterministic behavior:

```python
random.seed(42)  # Results are now reproducible
```

This is critical because transformations like `rotate()`, `skew()`, and `perspective()` use random parameters by default. Fixed seeds ensure consistent snapshots across test runs.

### Adding New Snapshot Tests

To add a new snapshot test:

1. **Create test function:**
```python
def test_my_new_transformation(self, base_test_image, snapshot_png, snapshot_json):
    """Test description."""
    image, _, bboxes = base_test_image

    # Set seed for determinism
    random.seed(42)

    # Apply transformation with fixed parameters
    result_img, meta = my_transformation(image, "white", param=0.5)
    result_bboxes = transform_bboxes(bboxes, meta)

    # Snapshot assertions
    assert result_img == snapshot_png
    assert normalize_bboxes_for_snapshot(result_bboxes) == snapshot_json
```

2. **Generate snapshot:**
```bash
pytest tests/test_transformation_snapshots.py::TestMyClass::test_my_new_transformation --snapshot-update
```

3. **Verify snapshot:**
- Check `tests/__snapshots__/` for generated files
- Visually inspect the PNG snapshot
- Verify the JSON bbox data looks correct

4. **Commit snapshots:**
```bash
git add tests/__snapshots__/
git commit -m "Add snapshot tests for my_transformation"
```

### Best Practices

✅ **DO:**
- Always use fixed random seeds in snapshot tests
- Use meaningful test names that describe what's being tested
- Update snapshots intentionally after reviewing changes
- Commit snapshot files to version control
- Run snapshot tests in CI/CD pipelines

❌ **DON'T:**
- Update snapshots without reviewing changes
- Use random parameters without fixing the seed
- Delete snapshot files manually (use `--snapshot-update` instead)
- Ignore failing snapshot tests
- Skip snapshot tests in CI

### Troubleshooting

#### Issue: Snapshots differ across platforms

**Problem:** PIL/Pillow may produce slightly different outputs on different OS or library versions.

**Solution:**
- Normalize floating-point values (already done in `normalize_bboxes_for_snapshot()`)
- Use consistent environment (Docker, uv lock file)
- Accept minor pixel differences by using image similarity metrics instead of exact matching

#### Issue: Too many snapshot files

**Problem:** Every test creates 2 files (PNG + JSON), can clutter the repo.

**Solution:**
- Combine related tests into parameterized tests
- Test only critical transformation parameters
- Use `.gitignore` patterns if needed (not recommended for snapshots)

#### Issue: Snapshot tests are slow

**Problem:** Generating and comparing images takes time.

**Solution:**
- Run snapshot tests separately from unit tests
- Use pytest markers: `@pytest.mark.snapshot` and run with `-m snapshot`
- Run in parallel with `pytest-xdist`: `pytest -n auto`

### CI/CD Integration

Add snapshot tests to your CI pipeline:

```yaml
# .github/workflows/test.yml (example)
- name: Run snapshot tests
  run: |
    pytest tests/test_transformation_snapshots.py -v
```

**Important:** Do NOT use `--snapshot-update` in CI. Tests should fail if snapshots don't match, prompting you to review changes locally.

## Running All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ocr_icelandic --cov-report=html

# Run specific test markers
pytest tests/ -m "not slow" -v
```

## Test Output

- **Unit test images**: Saved to `local_output/transformations/`
- **Snapshot files**: Saved to `tests/__snapshots__/`

Both directories should be added to `.gitignore` (except snapshots which should be committed).

## Contributing

When contributing new transformations or modifying existing ones:

1. Run existing tests: `pytest tests/test_transformations.py -v`
2. Run snapshot tests: `pytest tests/test_transformation_snapshots.py -v`
3. If transformation logic changed intentionally:
   - Review the differences carefully
   - Update snapshots: `pytest tests/test_transformation_snapshots.py --snapshot-update`
   - Commit updated snapshots with clear commit message explaining why
4. Add new tests for new functionality
5. Ensure all tests pass before submitting PR

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Syrupy (Snapshot Testing)](https://github.com/tophat/syrupy)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)
