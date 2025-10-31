"""OpenAI-powered OCR helpers.

This module mirrors the public interface exposed by ``ocr_utils`` but hands the
OCR work over to the OpenAI Responses API instead of running local Tesseract.
It deliberately skips all image preprocessing and slicing: the original drop
image (hosted by Discord) is sent to the model together with task-specific
instructions. Responses are validated with Pydantic models so downstream code
receives a predictable shape.

Exports:
 - ocr_cards(image_url: str, card_count: int) -> list[dict]
 - ocr_prints(image_url: str, card_count: int) -> list[dict]

Both OCR entry points return the same data structures produced by ``ocr_utils``
so the rest of the application can remain unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The `openai` package is required for ocr_openai. Install it with `uv add openai`."
    ) from exc

logger = logging.getLogger(__name__)


class OCRCard(BaseModel):
    """Single card OCR payload."""

    model_config = ConfigDict(extra="ignore")

    index: int
    series: str = ""
    name: str = ""
    raw_series: str = ""
    raw_name: str = ""


class CardsResponse(BaseModel):
    """Structured response for card OCR."""

    model_config = ConfigDict(extra="ignore")

    cards: List[OCRCard]


class OCRPrint(BaseModel):
    """Single print OCR payload."""

    model_config = ConfigDict(extra="ignore")

    index: int
    print_number: Optional[int] = None
    edition: Optional[int] = None
    raw: str = ""


class PrintsResponse(BaseModel):
    """Structured response for print OCR."""

    model_config = ConfigDict(extra="ignore")

    prints: List[OCRPrint]
    lowest_print_index: Optional[int] = None


_OPENAI_CLIENT: Optional[AsyncOpenAI] = None
T = TypeVar("T", bound=BaseModel)


def _get_openai_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client, initialising it on first use."""

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        _OPENAI_CLIENT = AsyncOpenAI(api_key=api_key)
    return _OPENAI_CLIENT


async def _call_openai(
    prompt: str,
    image_url: str,
    response_model: Type[T],
    *,
    max_tokens: int = 800,
) -> T:
    """Send the image and prompt to OpenAI and return a validated response model."""

    client = _get_openai_client()
    model = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")

    logger.debug("Calling OpenAI model %s", model)
    response = await client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_output_tokens=max_tokens,
        text_format=response_model,
    )

    output_text = getattr(response, "output_text", None)
    if not output_text:
        # Fall back to extracting the first text block manually
        try:
            first_block = response.output[0].content[0]
            if first_block.type == "output_text":
                output_text = first_block.text
        except Exception:  # pragma: no cover - defensive fallback
            output_text = None

    if not output_text:
        logger.error("Unexpected OpenAI response payload: %s", response)
        raise RuntimeError("OpenAI response did not contain text output")

    logger.debug("OpenAI response text: %s", output_text)
    try:
        return response_model.model_validate_json(output_text)
    except Exception as exc:  # pragma: no cover - validation safety
        logger.error("Failed to validate OpenAI response: %s", output_text)
        raise RuntimeError("OpenAI response did not match expected schema") from exc


async def ocr_cards(image_url: str, card_count: int) -> List[Dict[str, Any]]:
    """Use OpenAI to extract card names/series from the full drop image."""

    if card_count <= 0:
        return []

    prompt = (
        "You are analysing a Karuta card drop screenshot. "
        f"There are exactly {card_count} cards arranged from left to right. "
        "For each card, read the SERIES (upper text) and NAME (lower text). "
        "Return ONLY valid JSON with the following structure:\n"
        "{\n"
        '  "cards": [\n'
        '    {"index": 1, "series": "...", "name": "...", "raw_series": "...", "raw_name": "..."}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- index starts at 1 for the left-most card.\n"
        "- raw_* fields must contain the literal text you read, even if it looks incorrect.\n"
        "- If you cannot read something, use an empty string but keep the key.\n"
        "- Respond with JSON only, no explanations or markdown."
    )

    payload = await _call_openai(prompt, image_url, CardsResponse)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload.cards):
        raw_index = item.index if item.index is not None else idx + 1
        try:
            zero_based = int(raw_index) - 1
        except Exception:
            zero_based = idx
        series = item.series.strip()
        name = item.name.strip()
        raw_series = item.raw_series.strip()
        raw_name = item.raw_name.strip()
        results.append(
            {
                "index": max(zero_based, 0),
                "series": series,
                "name": name,
                "raw_series": raw_series,
                "raw_name": raw_name,
            }
        )

    return results


async def ocr_prints(image_url: str, card_count: int) -> List[Dict[str, Any]]:
    """Use OpenAI to extract print numbers and editions from the full drop image."""

    if card_count <= 0:
        return []

    prompt = (
        "You are reading the print and edition numbers from a Karuta card drop screenshot. "
        f"There are exactly {card_count} cards arranged left to right. "
        "For each card return its print number (digits before the separator) and edition (digits after). "
        "Also decide which card has the lowest print number; break ties by the lowest edition, then by left-most position. "
        "Return ONLY JSON with structure:\n"
        "{\n"
        '  "prints": [\n'
        '    {"index": 1, "print_number": 123, "edition": 4, "raw": "123·4"}\n'
        "  ],\n"
        '  "lowest_print_index": 1\n'
        "}\n"
        "Rules:\n"
        "- index starts at 1 for the left-most card.\n"
        "- If edition is missing, use null.\n"
        "- raw field should contain exactly what you read (include separators).\n"
        "- Respond with JSON only, no extra commentary or markdown."
    )

    payload = await _call_openai(prompt, image_url, PrintsResponse)

    lowest_idx = payload.lowest_print_index
    try:
        lowest_zero_based = int(lowest_idx) - 1 if lowest_idx is not None else None
    except Exception:
        lowest_zero_based = None

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload.prints):
        raw_index = item.index if item.index is not None else idx + 1
        try:
            zero_based = int(raw_index) - 1
        except Exception:
            zero_based = idx
        raw_value = item.raw.strip()
        try:
            print_number = (
                int(item.print_number) if item.print_number is not None else None
            )
        except Exception:
            print_number = None
        try:
            edition = int(item.edition) if item.edition is not None else None
        except Exception:
            edition = None
        entry: Dict[str, Any] = {
            "index": max(zero_based, 0),
            "print_number": print_number,
            "edition": edition,
            "raw": raw_value,
        }
        if lowest_zero_based is not None:
            entry["is_lowest"] = zero_based == lowest_zero_based
        results.append(entry)

    return results


__all__ = ["ocr_cards", "ocr_prints"]
