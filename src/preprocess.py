"""
Forensic Preprocessing and Metadata Harvesting Module
Handles deterministic image orientation, RGB color space standardization,
anti-aliased resizing, ImageNet normalization, and blur/entropy quality analysis.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ExifTags
import torch
import torchvision.transforms.functional as TF


def extract_forensic_metadata(image: PIL.Image.Image) -> Dict[str, Any]:
    """
    Harvests EXIF metadata, camera hardware tags, timestamps, and ICC color profile presence.
    
    Args:
        image: Decoded PIL Image instance.
        
    Returns:
        dict: Normalized metadata telemetry.
    """
    metadata: Dict[str, Any] = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "exposure_time": None,
        "f_number": None,
        "iso_speed": None,
        "icc_profile_present": bool(image.info.get("icc_profile")),
        "raw_exif_count": 0
    }
    
    try:
        exif = image.getexif()
        if exif:
            metadata["has_exif"] = True
            metadata["raw_exif_count"] = len(exif)
            
            for tag_id, value in exif.items():
                tag_name = PIL.ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name == "Make" and value:
                    metadata["camera_make"] = str(value).strip()
                elif tag_name == "Model" and value:
                    metadata["camera_model"] = str(value).strip()
                elif tag_name == "Software" and value:
                    metadata["software"] = str(value).strip()
                elif tag_name in ("DateTimeOriginal", "DateTime") and value:
                    metadata["datetime_original"] = str(value).strip()
                elif tag_name == "ExposureTime" and value:
                    metadata["exposure_time"] = str(value)
                elif tag_name == "FNumber" and value:
                    try:
                        metadata["f_number"] = float(value)
                    except (ValueError, TypeError):
                        metadata["f_number"] = str(value)
                elif tag_name in ("ISOSpeedRatings", "PhotographicSensitivity") and value:
                    try:
                        metadata["iso_speed"] = int(value)
                    except (ValueError, TypeError):
                        metadata["iso_speed"] = str(value)
    except Exception:
        # Non-critical: images without EXIF or with corrupted chunks degrade gracefully
        pass
        
    return metadata


def normalize_exif_orientation(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Applies physical transposition to rotated portrait images based on EXIF orientation tag.
    
    Args:
        image: Source PIL Image.
        
    Returns:
        PIL.Image.Image: Transposed upright image.
    """
    try:
        transposed = PIL.ImageOps.exif_transpose(image)
        return transposed if transposed is not None else image
    except Exception:
        return image


def ensure_three_channel_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts Grayscale ('L'), Palette ('P'), Transparency ('RGBA'), or CMYK modes into standard 3-channel RGB.
    Transparency layers are composited onto a solid white background canvas (255, 255, 255).
    
    Args:
        image: Source PIL Image.
        
    Returns:
        PIL.Image.Image: Normalized image strictly in 'RGB' mode.
    """
    if image.mode == "RGB":
        return image
    
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = PIL.Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas
        
    return image.convert("RGB")


def resize_image_deterministic(
    image: PIL.Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    interpolation: str = "bilinear",
    antialias: bool = True
) -> PIL.Image.Image:
    """
    Scales image to target dimensions using anti-aliased resampling.
    
    Args:
        image: Standardized RGB image.
        target_size: (width, height) tuple.
        interpolation: 'bilinear' or 'bicubic'.
        antialias: Whether to enable anti-aliasing.
        
    Returns:
        PIL.Image.Image: Resized PIL image.
    """
    resample_mode = (
        PIL.Image.Resampling.BICUBIC
        if interpolation.lower() == "bicubic"
        else PIL.Image.Resampling.BILINEAR
    )
    return image.resize(target_size, resample=resample_mode, reducing_gap=3.0 if antialias else None)


def image_to_normalized_tensor(
    image: PIL.Image.Image,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Transforms PIL image to float32 Tensor, normalizes with ImageNet constants, and adds batch dimension.
    
    Args:
        image: Resized RGB image.
        mean: Channel-wise normalization means (default: ImageNet [0.485, 0.456, 0.406]).
        std: Channel-wise normalization standard deviations (default: ImageNet [0.229, 0.224, 0.225]).
        
    Returns:
        torch.Tensor: FloatTensor with shape [1, 3, 224, 224].
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    # to_tensor converts PIL [0, 255] HWC to FloatTensor [0.0, 1.0] CHW
    tensor = TF.to_tensor(image)
    normalized = TF.normalize(tensor, mean=mean, std=std)
    return normalized.unsqueeze(0)


def compute_image_quality_metrics(image: PIL.Image.Image) -> Dict[str, Any]:
    """
    Calculates Laplacian blur variance and Shannon entropy over grayscale luminance.
    
    Args:
        image: PIL Image (any mode).
        
    Returns:
        dict:
            - laplacian_variance (float): Focus/sharpness metric (variance < 100 indicates severe blur).
            - entropy (float): Information density / texture metric.
            - is_low_quality (bool): True if focus or information density fails thresholds.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    
    # 3x3 Discrete Laplacian Kernel convolution
    # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    h, w = gray.shape
    if h < 3 or w < 3:
        return {"laplacian_variance": 0.0, "entropy": 0.0, "is_low_quality": True}
        
    padded = np.pad(gray, 1, mode="reflect")
    laplacian = (
        padded[0:h, 1:w+1] +
        padded[2:h+2, 1:w+1] +
        padded[1:h+1, 0:w] +
        padded[1:h+1, 2:w+2] -
        4.0 * gray
    )
    laplacian_var = float(np.var(laplacian))
    
    # Shannon Entropy computation over 256 grayscale histogram bins
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
    
    is_low_quality = (laplacian_var < 100.0) or (entropy_val < 3.5)
    
    return {
        "laplacian_variance": round(laplacian_var, 2),
        "entropy": round(entropy_val, 2),
        "is_low_quality": is_low_quality
    }
