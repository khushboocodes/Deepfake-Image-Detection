"""
Unit and Integration Tests for Ingestion Security and Memory Protection Module.
"""

import io
import pytest
import PIL.Image
from src.security import (
    validate_magic_bytes,
    validate_file_size,
    validate_image_dimensions,
    sanitize_filename,
    safe_ingest_image,
    MalformedFileError,
    FileSizeOutOfBoundsError,
    DimensionOutOfBoundsError
)


def test_validate_magic_bytes_jpeg():
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    assert validate_magic_bytes(jpeg_header) == "JPEG"


def test_validate_magic_bytes_png():
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert validate_magic_bytes(png_header) == "PNG"


def test_validate_magic_bytes_invalid():
    invalid_bytes = b"MZ\x90\x00\x03\x00\x00\x00"  # PE binary header
    with pytest.raises(MalformedFileError):
        validate_magic_bytes(invalid_bytes)


def test_validate_file_size_bounds():
    with pytest.raises(FileSizeOutOfBoundsError):
        validate_file_size(b"too small", min_bytes=1024)

    valid_payload = b"x" * 2048
    assert validate_file_size(valid_payload, min_bytes=1024, max_bytes=10000) == 2048

    with pytest.raises(FileSizeOutOfBoundsError):
        validate_file_size(valid_payload, min_bytes=1024, max_bytes=1500)


def test_validate_image_dimensions():
    # Valid image
    buf = io.BytesIO()
    PIL.Image.new("RGB", (200, 150)).save(buf, format="JPEG")
    buf.seek(0)
    w, h = validate_image_dimensions(buf, min_dim=64, max_dim=4096)
    assert w == 200 and h == 150

    # Dimension too small (<64px)
    buf_small = io.BytesIO()
    PIL.Image.new("RGB", (32, 32)).save(buf_small, format="JPEG")
    buf_small.seek(0)
    with pytest.raises(DimensionOutOfBoundsError):
        validate_image_dimensions(buf_small, min_dim=64)


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert sanitize_filename("..\\..\\windows\\system32.png") == "system32.png"
    assert sanitize_filename("hello@#$ world.jpg") == "hello_world.jpg"
    assert sanitize_filename("") == "unnamed_image.jpg"


def test_safe_ingest_image_success():
    buf = io.BytesIO()
    PIL.Image.new("RGB", (128, 128), (200, 100, 50)).save(buf, format="JPEG")
    raw = buf.getvalue()
    
    is_valid, img, tele, err = safe_ingest_image(raw, "test_pic.jpg")
    assert is_valid is True
    assert img is not None
    assert tele["dimensions"] == [128, 128]
    assert tele["format"] == "JPEG"
    assert err is None


def test_safe_ingest_image_corrupt():
    corrupt_bytes = b"NOT_AN_IMAGE_PAYLOAD_TESTING"
    is_valid, img, tele, err = safe_ingest_image(corrupt_bytes, "malicious.jpg")
    assert is_valid is False
    assert img is None
    assert err is not None
