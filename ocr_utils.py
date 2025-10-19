"""OCR utilities for parsing Karuta drop images.

This module is an adaptation of logic found in `ref.py`, refactored to be:
 - Non-blocking (uses aiohttp for network I/O)
 - Defensive against missing optional dependencies (OpenCV / Tesseract)
 - Easier to test in isolation (pure function style where possible)

Public functions:
 - slice_drop_image(image: np.ndarray, card_count: int) -> list[np.ndarray]
 - extract_card(card_image: np.ndarray, index: int) -> dict
 - extract_cards_from_drop(image_bytes: bytes, card_count: int) -> list[dict]
 - extract_print_edition(card_image: np.ndarray, index: int) -> dict
 - extract_print_edition_from_drop(image_bytes: bytes, card_count: int) -> list[dict]
 - fetch_image(url: str) -> bytes
 - ocr_cards(image_bytes: bytes, card_count: int) -> list[dict]
 - ocr_prints(image_bytes: bytes, card_count: int) -> list[dict]

Each returned card dict includes:
 {"index": int, "name": str, "series": str, "raw_name": str, "raw_series": str}

If OCR dependencies are unavailable, a RuntimeError is raised with a helpful
message so the caller can surface it to the user.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:  # Lazy availability flags
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import pytesseract  # type: ignore

    _OCR_AVAILABLE = True
except Exception as e:  # pragma: no cover - best effort
    _OCR_AVAILABLE = False
    _IMPORT_ERROR = e

try:
    import aiohttp
    import os
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

REFERENCE_IMAGE_WIDTH = 836.0
REFERENCE_CARD_WIDTH = 277.0
REFERENCE_CARD_HEIGHT = 419.0

CARD_START_X_RATIO = 46.0 / REFERENCE_IMAGE_WIDTH
CARD_X_STEP_RATIO = 277.0 / REFERENCE_IMAGE_WIDTH
CARD_WIDTH_RATIO = (277.0 - 10.0) / REFERENCE_IMAGE_WIDTH  # trim borders slightly

NAME_Y_RATIO = 55.0 / REFERENCE_CARD_HEIGHT
SERIES_Y_RATIO = 307.0 / REFERENCE_CARD_HEIGHT
ROW_HEIGHT_RATIO = 53.0 / REFERENCE_CARD_HEIGHT
NAME_WIDTH_RATIO = 180.0 / REFERENCE_CARD_WIDTH

PRINT_ROI_WIDTH_RATIO = 140.0 / REFERENCE_CARD_WIDTH
PRINT_ROI_HEIGHT_RATIO = 26.0 / REFERENCE_CARD_HEIGHT
PRINT_BOTTOM_PADDING_RATIO = 26.0 / REFERENCE_CARD_HEIGHT
PRINT_RIGHT_PADDING_RATIO = 60.0 / REFERENCE_CARD_WIDTH


def _ensure_ocr_available():
    if not _OCR_AVAILABLE:
        raise RuntimeError(
            "OCR dependencies not installed: requires opencv-python-headless, numpy, pytesseract. "
            f"Original import error: {_IMPORT_ERROR}"
        )


async def fetch_image(url: str) -> bytes:
    """Fetch image bytes asynchronously.

    Falls back to a blocking urllib import if aiohttp is missing (should be
    uncommon). In that fallback we run the blocking call in a thread executor
    to keep the event loop responsive.
    """
    if aiohttp:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to fetch image: HTTP {resp.status}")
                return await resp.read()
    # Fallback (blocking) path
    import urllib.request

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: urllib.request.urlopen(url).read(),  # type: ignore
    )


def slice_drop_image(image: "np.ndarray", card_count: int) -> List["np.ndarray"]:
    """Split a decoded drop image into per-card slices.

    The returned list contains each card column slice (full height) by dividing
    the image width evenly among `card_count` slots (using floor division).
    Any remainder pixels are included in the final slice.
    """
    if card_count <= 0:
        return []

    _, w = image.shape[:2]
    slice_width = max(1, w // card_count)

    slices: List["np.ndarray"] = []
    x_start = 0
    for idx in range(card_count):
        x_end = x_start + slice_width
        if idx == card_count - 1:
            x_end = w  # include any remainder in the last slice
        x_end = min(w, max(x_end, x_start + 1))
        slices.append(image[:, x_start:x_end])
        x_start = x_end

    return slices


def extract_cards_from_drop(
    image_bytes: bytes, card_count: int
) -> List[Dict[str, Any]]:
    """Extract card (series, name) pairs from a Karuta drop image.

    The layout assumptions mirror those from `ref.py`.
    `card_count` is parsed from the Discord message content.
    """
    _ensure_ocr_available()

    if card_count <= 0:
        return []

    # Decode image (keep original color)
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV")

    # No preprocessing - use original image
    image_slices = slice_drop_image(image, card_count)

    cards: List[Dict[str, Any]] = []
    for idx, card_slice in enumerate(image_slices):
        cards.append(extract_card(card_slice, idx))

    return cards


def _preprocess_roi_for_ocr(
    roi: "np.ndarray", source_dpi: int = 72, target_dpi: int = 300
) -> "np.ndarray":
    """Preprocess ROI to improve OCR accuracy.

    Steps:
    1. Convert to grayscale
    2. Scale from source DPI (typically 72) to target DPI (300 minimum for OCR)
    3. Apply Otsu's thresholding to handle varying lighting
    4. Denoise to remove artifacts
    5. Unsharp masking to enhance text edges

    Args:
        roi: Input region of interest (BGR or grayscale)
        source_dpi: DPI of source image (default 72, common for web images)
        target_dpi: Target DPI for OCR (default 300, minimum recommended)

    Returns:
        Preprocessed image optimized for OCR
    """
    if roi.size == 0:
        return roi

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # First, scale based on DPI conversion (72 DPI -> 300 DPI = 4.166x scale)
    # dpi_scale_factor = target_dpi / source_dpi
    dpi_scale_factor = 4

    # Apply DPI scaling
    if dpi_scale_factor != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=dpi_scale_factor,
            fy=dpi_scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    # Apply Otsu's thresholding for better text contrast
    # Otsu automatically determines the optimal threshold value
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise using fastNlMeansDenoising (removes small artifacts)
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # Apply unsharp masking to enhance text edges
    # Create a blurred version and subtract it from the original (weighted)
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)

    # Ensure pixel values are in valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def _clean_ocr_text(s: str) -> str:
    # Normalize whitespace and strip non-letter except space
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-zA-Z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_card(card_image: "np.ndarray", index: int) -> Dict[str, Any]:
    """Extract series/name information for a single card slice."""
    height, width = card_image.shape[:2]
    if height <= 0 or width <= 0:
        return {
            "index": index,
            "name": "",
            "series": "",
            "raw_name": "",
            "raw_series": "",
        }

    name_width = max(1, int(round(NAME_WIDTH_RATIO * width)))
    name_y = int(round(NAME_Y_RATIO * height))
    name_height = max(1, int(round(ROW_HEIGHT_RATIO * height)))
    name_y_end = min(name_y + name_height, height)

    series_y = int(round(SERIES_Y_RATIO * height))
    series_height = name_height  # same config
    series_y_end = min(series_y + series_height, height)

    if name_width <= 0 or name_y >= name_y_end:
        raw_name = ""
        clean_name = ""
    else:
        name_roi = card_image[name_y:name_y_end, 0:name_width]
        # Preprocess ROI before OCR
        preprocessed_name_roi = _preprocess_roi_for_ocr(name_roi)
        raw_name = pytesseract.image_to_string(
            preprocessed_name_roi, lang="eng", config="--psm 6"
        )
        clean_name = _clean_ocr_text(raw_name)

    if name_width <= 0 or series_y >= series_y_end:
        raw_series = ""
        clean_series = ""
    else:
        series_roi = card_image[series_y:series_y_end, 0:name_width]
        # Preprocess ROI before OCR
        preprocessed_series_roi = _preprocess_roi_for_ocr(series_roi)
        raw_series = pytesseract.image_to_string(
            preprocessed_series_roi, lang="eng", config="--psm 6"
        )
        clean_series = _clean_ocr_text(raw_series)

    return {
        "index": index,
        "name": clean_name,
        "series": clean_series,
        "raw_name": raw_name.strip(),
        "raw_series": raw_series.strip(),
    }


def extract_print_edition(card_image: "np.ndarray", index: int) -> Dict[str, Any]:
    """Extract print and edition metadata for a single card slice."""
    height, width = card_image.shape[:2]
    if height <= 0 or width <= 0:
        return {"index": index, "print_number": None, "edition": None, "raw": ""}

    candidate_y = []
    roi_height = max(1, int(round(PRINT_ROI_HEIGHT_RATIO * height)))
    roi_width = max(1, int(round(PRINT_ROI_WIDTH_RATIO * width)))
    right_padding = int(round(PRINT_RIGHT_PADDING_RATIO * width))

    bottom_candidate = (
        height - roi_height - int(round(PRINT_BOTTOM_PADDING_RATIO * height))
    )
    bottom_candidate = max(0, bottom_candidate)
    if bottom_candidate + roi_height <= height:
        candidate_y.append(bottom_candidate)

    logger.info(f"Card {index} OCR candidates: {candidate_y}")

    raw_capture = None
    found = None
    roi_x_start = max(width - roi_width, 0)
    for y in candidate_y:
        roi = card_image[y : y + roi_height, roi_x_start : width - right_padding]
        if roi.size == 0:
            continue

        # Preprocess ROI before OCR
        preprocessed_roi = _preprocess_roi_for_ocr(roi)

        txt = pytesseract.image_to_string(
            preprocessed_roi,
            lang="eng",
            config="--psm 6 -c tessedit_char_whitelist=0123456789·.:•-",
        )
        raw_capture = txt.strip()
        cleaned = raw_capture.replace("\n", " ")

        logger.info(f"Card {index} OCR raw print capture: '{raw_capture}'")

        cleaned = cleaned.replace(":", "·").replace("-", "·").replace("•", "·")
        cleaned = re.sub(r"\s+", "", cleaned)
        m = re.match(r"(\d{2,})[·.](\d{1,2})", cleaned)
        if not m:
            m2 = re.match(r"(\d{2,})(\d{1,3})", cleaned)
            if m2:
                m = m2
        if m:
            try:
                pn = int(m.group(1))
                ed = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else None
            except Exception:
                pn, ed = None, None
            found = (pn, ed, raw_capture)
            break

    if found:
        pn, ed, raw = found
        return {"index": index, "print_number": pn, "edition": ed, "raw": raw}

    return {
        "index": index,
        "print_number": None,
        "edition": None,
        "raw": raw_capture or "",
    }


async def ocr_cards(image_bytes: bytes, card_count: int) -> List[Dict[str, Any]]:
    """Asynchronously OCR card names/series from already-fetched image bytes.

    Use this when you already have the image bytes (e.g., to run multiple OCR
    passes without refetching)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: extract_cards_from_drop(image_bytes, card_count)
    )


def extract_print_edition_from_drop(
    image_bytes: bytes, card_count: int
) -> List[Dict[str, Any]]:
    """Attempt to read bottom-right print metadata per card."""
    _ensure_ocr_available()
    if card_count <= 0:
        return []

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV")

    card_slices = slice_drop_image(image, card_count)

    results: List[Dict[str, Any]] = []
    for idx, card_slice in enumerate(card_slices):
        results.append(extract_print_edition(card_slice, idx))

    return results


async def ocr_prints(image_bytes: bytes, card_count: int) -> List[Dict[str, Any]]:
    """Asynchronously OCR print / edition metadata from already-fetched bytes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: extract_print_edition_from_drop(image_bytes, card_count)
    )


__all__ = [
    "slice_drop_image",
    "extract_card",
    "ocr_cards",
    "extract_cards_from_drop",
    "fetch_image",
    "extract_print_edition",
    "extract_print_edition_from_drop",
    "ocr_prints",
]
