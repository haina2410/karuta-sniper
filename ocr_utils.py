"""OCR utilities for parsing Karuta drop images.

This module is an adaptation of logic found in `ref.py`, refactored to be:
 - Non-blocking (uses aiohttp for network I/O)
 - Defensive against missing optional dependencies (OpenCV / Tesseract)
 - Easier to test in isolation (pure function style where possible)

Public functions:
 - extract_cards_from_drop(image_bytes: bytes, card_count: int) -> list[dict]
 - extract_print_edition_from_drop(image_bytes: bytes, card_count: int) -> list[dict]
 - fetch_image(url: str) -> bytes
 - ocr_cards_from_url(url: str, card_count: int) -> list[dict]
 - ocr_prints_from_url(url: str, card_count: int) -> list[dict]

Each returned card dict includes:
 {"index": int, "name": str, "series": str, "raw_name": str, "raw_series": str}

If OCR dependencies are unavailable, a RuntimeError is raised with a helpful
message so the caller can surface it to the user.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Dict

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
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

NAME_Y = 55
SERIES_Y = 307
ROW_HEIGHT = 53
NAME_WIDTH = 180
NAME_X_START = 46
CARD_X_STEP = 277


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
        None, lambda: urllib.request.urlopen(url).read()  # type: ignore
    )


def extract_cards_from_drop(image_bytes: bytes, card_count: int) -> List[Dict[str, str]]:
    """Extract card (series, name) pairs from a Karuta drop image.

    The layout assumptions mirror those from `ref.py`.
    `card_count` is parsed from the Discord message content.
    """
    _ensure_ocr_available()

    if card_count <= 0:
        return []

    # Decode image (grayscale)
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV")

    # Binarize using OTSU
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    names: List[str] = []
    series: List[str] = []

    # Iterate columns
    for idx in range(card_count):
        x = NAME_X_START + idx * CARD_X_STEP

        # Name row region
        name_roi = thresh[NAME_Y : NAME_Y + ROW_HEIGHT, x : x + NAME_WIDTH]
        raw_name = pytesseract.image_to_string(name_roi, lang="eng", config="--psm 6")
        clean_name = _clean_ocr_text(raw_name)
        names.append((clean_name, raw_name))

        # Series row region
        series_roi = thresh[SERIES_Y : SERIES_Y + ROW_HEIGHT, x : x + NAME_WIDTH]
        raw_series = pytesseract.image_to_string(
            series_roi, lang="eng", config="--psm 6"
        )
        clean_series = _clean_ocr_text(raw_series)
        series.append((clean_series, raw_series))

    cards: List[Dict[str, str]] = []
    for i in range(card_count):
        name_clean, name_raw = names[i]
        series_clean, series_raw = series[i]
        cards.append(
            {
                "index": i,
                "name": name_clean,
                "series": series_clean,
                "raw_name": name_raw.strip(),
                "raw_series": series_raw.strip(),
            }
        )
    return cards


def _clean_ocr_text(s: str) -> str:
    # Normalize whitespace and strip non-letter except space
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-zA-Z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def ocr_cards_from_url(url: str, card_count: int) -> List[Dict[str, str]]:
    bytes_ = await fetch_image(url)
    loop = asyncio.get_running_loop()
    # Run CPU-heavy OCR in default executor to avoid event loop blockage
    return await loop.run_in_executor(
        None, lambda: extract_cards_from_drop(bytes_, card_count)
    )


def extract_print_edition_from_drop(image_bytes: bytes, card_count: int):
    """Attempt to read bottom-right print metadata per card.

    Heuristic approach: for each card we probe several candidate Y offsets
    near the lower portion of the card to find a pattern like:
        75846·1  or  75846-1  or  75846 1

    Returns list[dict]: {index, print_number (int|None), edition (int|None), raw}
    """
    _ensure_ocr_available()
    if card_count <= 0:
        return []

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV")

    h, w = image.shape[:2]

    # Constants derived from earlier x-layout + experimentation guidelines
    CARD_WIDTH_APPROX = CARD_X_STEP - 10  # some padding
    PRINT_ROI_WIDTH = 130
    PRINT_ROI_HEIGHT = 42
    # Candidate Y starts (from near 60% height downward) - will clamp if beyond image
    candidate_y = []
    base_candidates = [int(h * r) for r in (0.58, 0.62, 0.66, 0.70, 0.74)]
    seen = set()
    for cy in base_candidates:
        if cy + PRINT_ROI_HEIGHT < h and cy not in seen:
            candidate_y.append(cy)
            seen.add(cy)

    results = []
    for idx in range(card_count):
        base_x = NAME_X_START + idx * CARD_X_STEP
        card_right = min(base_x + CARD_WIDTH_APPROX, w)
        roi_x = max(card_right - PRINT_ROI_WIDTH, base_x)

        found = None
        raw_capture = None
        for y in candidate_y:
            roi = image[y : y + PRINT_ROI_HEIGHT, roi_x : roi_x + PRINT_ROI_WIDTH]
            if roi.size == 0:
                continue
            # Increase contrast
            _, bin_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            txt = pytesseract.image_to_string(
                bin_roi,
                lang="eng",
                config="--psm 6 -c tessedit_char_whitelist=0123456789·.:•-",
            )
            raw_capture = txt.strip()
            cleaned = raw_capture.replace("\n", " ")
            cleaned = cleaned.replace(":", "·").replace("-", "·").replace("•", "·")
            cleaned = re.sub(r"\s+", "", cleaned)
            m = re.match(r"(\d{2,})[·.](\d{1,3})", cleaned)
            if not m:
                # Sometimes separator dropped; try space split
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
            results.append(
                {
                    "index": idx,
                    "print_number": pn,
                    "edition": ed,
                    "raw": raw,
                }
            )
        else:
            results.append(
                {
                    "index": idx,
                    "print_number": None,
                    "edition": None,
                    "raw": raw_capture or "",
                }
            )
    return results


async def ocr_prints_from_url(url: str, card_count: int):
    bytes_ = await fetch_image(url)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: extract_print_edition_from_drop(bytes_, card_count)
    )


__all__ = [
    "ocr_cards_from_url",
    "extract_cards_from_drop",
    "fetch_image",
    "extract_print_edition_from_drop",
    "ocr_prints_from_url",
]
