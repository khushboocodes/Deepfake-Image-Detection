"""
Security and Ingestion Module
Enforces magic byte validation, decompression bomb boundaries, filename sanitization,
and memory protection for the deepfake detection pipeline.
"""

import io
import re
import hashlib
from typing import Tuple, Optional
import PIL.Image

# Global defense against decompression bombs (16 megapixels max)
PIL.Image.MAX_IMAGE_PIXELS = 16_777_216


class SecurityValidationError(Exception):
    """Base exception for security and ingestion validation failures."""
    pass


class MalformedFileError(SecurityValidationError):
    """Raised when file magic bytes do not match supported image formats."""
    pass


class FileSizeOutOfBoundsError(SecurityValidationError):
    """Raised when payload size is under 1KB or over 15MB."""
    pass


class DimensionOutOfBoundsError(SecurityValidationError):
    """Raised when image pixel dimensions fall outside [64, 4096]."""
    pass


def validate_magic_bytes(file_bytes: bytes) -> str:
    """
    Inspects the initial bytes of a buffer to verify JPEG or PNG magic signatures.
    
    Args:
        file_bytes: Binary payload of image file.
        
    Returns:
        str: 'JPEG' or 'PNG'.
        
    Raises:
        MalformedFileError: If format is unsupported or bytes truncated.
    """
    if len(file_bytes) < 8:
        raise MalformedFileError("File payload is truncated or empty (<8 bytes).")
    
    # JPEG magic numbers: FF D8 FF
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    
    # PNG magic numbers: 89 50 4E 47 0D 0A 1A 0A
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    
    raise MalformedFileError(
        "File signature does not match permitted image formats (JPEG, PNG). "
        "Executable, vector, or corrupted headers are blocked."
    )


def validate_file_size(file_bytes: bytes, min_bytes: int = 1024, max_bytes: int = 15_728_640) -> int:
    """
    Enforces minimum and maximum allowable byte payload limits.
    
    Args:
        file_bytes: Binary payload of image.
        min_bytes: Minimum bytes allowed (default: 1 KB).
        max_bytes: Maximum bytes allowed (default: 15 MB).
        
    Returns:
        int: Exact byte length.
        
    Raises:
        FileSizeOutOfBoundsError: If size is out of bounds.
    """
    size = len(file_bytes)
    if size < min_bytes:
        raise FileSizeOutOfBoundsError(
            f"File size ({size} bytes) is below minimum allowable threshold ({min_bytes} bytes)."
        )
    if size > max_bytes:
        raise FileSizeOutOfBoundsError(
            f"File size ({size / (1024*1024):.2f} MB) exceeds maximum allowable ceiling ({max_bytes / (1024*1024):.2f} MB)."
        )
    return size


def validate_image_dimensions(
    stream: io.BytesIO,
    min_dim: int = 64,
    max_dim: int = 4096,
    max_pixels: int = 16_777_216
) -> Tuple[int, int]:
    """
    Inspects image dimensions via header decoding without loading full raster buffer into RAM.
    
    Args:
        stream: BytesIO stream positioned at byte 0.
        min_dim: Minimum width and height allowed (pixels).
        max_dim: Maximum width and height allowed (pixels).
        max_pixels: Maximum total pixel area (width * height).
        
    Returns:
        Tuple[int, int]: (width, height)
        
    Raises:
        DimensionOutOfBoundsError: If image is smaller than min_dim or larger than max_dim.
    """
    stream.seek(0)
    try:
        with PIL.Image.open(stream) as img:
            width, height = img.size
    except Exception as e:
        stream.seek(0)
        raise MalformedFileError(f"Failed to decode image header: {str(e)}")
    
    stream.seek(0)
    
    if width < min_dim or height < min_dim:
        raise DimensionOutOfBoundsError(
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({min_dim}x{min_dim} px)."
        )
    if width > max_dim or height > max_dim:
        raise DimensionOutOfBoundsError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_dim}x{max_dim} px)."
        )
    if width * height > max_pixels:
        raise DimensionOutOfBoundsError(
            f"Total pixel count ({width * height:,}) exceeds safe decompression threshold ({max_pixels:,} px)."
        )
        
    return width, height


def sanitize_filename(raw_filename: str) -> str:
    """
    Sanitizes user-provided filename, stripping directory traversal sequences and unsafe characters.
    
    Args:
        raw_filename: Client-provided filename (e.g. '../../etc/passwd.jpg').
        
    Returns:
        str: Whitelisted alphanumeric filename (e.g. 'passwd.jpg'), max 128 characters.
    """
    if not raw_filename:
        return "unnamed_image.jpg"
    
    # Strip any path separators
    clean_name = raw_filename.replace("\\", "/").split("/")[-1]
    
    # Strip null bytes and non-whitelisted characters, collapsing multiples
    clean_name = re.sub(r"[^\w\.\-]+", "_", clean_name)
    
    # Avoid hidden files or empty names
    clean_name = clean_name.lstrip(".")
    if not clean_name:
        clean_name = "unnamed_image.jpg"
        
    # Truncate length
    if len(clean_name) > 128:
        base, ext = (clean_name.rsplit(".", 1) + ["jpg"])[:2]
        clean_name = f"{base[:120]}.{ext}"
        
    return clean_name


def safe_ingest_image(
    raw_bytes: bytes,
    filename: str,
    config: Optional[dict] = None
) -> Tuple[bool, Optional[PIL.Image.Image], Optional[dict], Optional[str]]:
    """
    Orchestrates complete security validation: file size, magic bytes, dimensions, and safe decode.
    
    Args:
        raw_bytes: Binary payload of uploaded file.
        filename: Client-supplied filename string.
        config: System configuration dictionary loaded from configs/preprocess.json.
        
    Returns:
        Tuple:
            - is_valid (bool): True if image passed all security barriers.
            - image (PIL.Image.Image or None): Decoded PIL Image instance.
            - telemetry (dict or None): Ingestion metadata (sha256, size, dimensions, format).
            - error_message (str or None): Sanitized human-readable error description if invalid.
    """
    sec_cfg = (config or {}).get("security_limits", {})
    min_bytes = sec_cfg.get("min_file_size_bytes", 100)
    max_bytes = sec_cfg.get("max_file_size_bytes", 15_728_640)
    min_dim = sec_cfg.get("min_dimension_px", 64)
    max_dim = sec_cfg.get("max_dimension_px", 4096)
    max_pixels = sec_cfg.get("max_image_pixels", 16_777_216)
    
    sanitized_name = sanitize_filename(filename)
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    
    try:
        # Step 1: Validate file size
        filesize = validate_file_size(raw_bytes, min_bytes=min_bytes, max_bytes=max_bytes)
        
        # Step 2: Validate magic bytes
        fmt = validate_magic_bytes(raw_bytes)
        
        # Step 3: Validate dimensions
        stream = io.BytesIO(raw_bytes)
        width, height = validate_image_dimensions(
            stream,
            min_dim=min_dim,
            max_dim=max_dim,
            max_pixels=max_pixels
        )
        
        # Step 4: Full raster decode
        stream.seek(0)
        img = PIL.Image.open(stream)
        img.load()  # Force decode into memory within safe limit
        
        telemetry = {
            "filename": sanitized_name,
            "sha256": sha256_hash,
            "format": fmt,
            "filesize_bytes": filesize,
            "dimensions": [width, height],
            "mode": img.mode,
            "aspect_ratio": round(width / max(1, height), 3)
        }
        
        return True, img, telemetry, None
        
    except SecurityValidationError as sve:
        return False, None, {"filename": sanitized_name, "sha256": sha256_hash}, str(sve)
    except PIL.Image.DecompressionBombError:
        return (
            False,
            None,
            {"filename": sanitized_name, "sha256": sha256_hash},
            "Security Rejection: Image pixel count exceeds safe decompression limits (Decompression Bomb Defense)."
        )
    except Exception as e:
        return (
            False,
            None,
            {"filename": sanitized_name, "sha256": sha256_hash},
            f"Image Processing Error: Could not decode image payload safely ({type(e).__name__})."
        )
