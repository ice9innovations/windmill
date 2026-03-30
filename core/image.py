"""
Image validation, normalization, and hashing utilities.

No Flask dependency. All operations are in-memory — no temp files, no disk I/O.
"""
import io
import os

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # adds HEIC/HEIF support to PIL globally

# Formats that downstream workers (VLMs, SAM3, etc.) cannot handle.
_WEB_SAFE = {'JPEG', 'PNG', 'WEBP'}


def validate_and_normalize_image(image_bytes, max_dimension=None):
    """Validate image bytes and resize/transcode if necessary. No disk I/O ever.

    PIL's verify() performs an integrity check but closes the stream and
    invalidates the Image object, so we re-open from the original bytes
    for actual processing.

    Returns:
        (
            normalized_bytes,
            original_width,
            original_height,
            normalized_width,
            normalized_height,
        )
    Raises ValueError with a safe message if the bytes are not a valid image.
    """
    if max_dimension is None:
        max_dimension = int(os.getenv('MAX_IMAGE_DIMENSION', '2048'))

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise ValueError("Invalid or corrupt image")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        raise ValueError("Failed to decode image")

    original_format = img.format or 'JPEG'
    original_width, original_height = img.size

    # Normalize EXIF orientation so all downstream workers (VLMs, SAM3, etc.)
    # receive pixels in display orientation. Without this, a portrait phone
    # photo stored as landscape pixels with an EXIF rotation tag would cause
    # bounding box coordinates to be in the raw (unrotated) space while the
    # browser displays the image rotated — producing misaligned overlays.
    transposed = ImageOps.exif_transpose(img)
    orientation_changed = transposed is not img
    img = transposed
    width, height = img.size

    needs_transcode = original_format not in _WEB_SAFE

    if max(width, height) <= max_dimension and not needs_transcode and not orientation_changed:
        return image_bytes, original_width, original_height, width, height

    if max(width, height) > max_dimension:
        ratio    = max_dimension / max(width, height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        img      = img.resize(new_size, Image.LANCZOS)
        width, height = img.size

    save_format = original_format if original_format in _WEB_SAFE else 'JPEG'
    if save_format == 'JPEG' and img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format=save_format)
    return buf.getvalue(), original_width, original_height, width, height
