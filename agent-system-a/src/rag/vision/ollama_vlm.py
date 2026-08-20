"""Small dependency-light Ollama client for Qwen2.5-VL."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .prompt import VISION_PROMPT
from .validation import normalize_description


LOGGER = logging.getLogger("rag.vision.ollama")


def _encoded_image(path: Path, max_size: int = 1600) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON is not an object")
    return value


def describe_page(
    image_path: Path,
    model: str,
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 300,
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [_encoded_image(image_path)]}],
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
        "keep_alive": "10m",
    }
    request = Request(
        f'{base_url.rstrip("/")}/api/chat',
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc.reason}") from exc
    content = result.get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama returned an empty vision response")
    LOGGER.debug("Raw VLM response image=%s content=%s", image_path, content[:12000])
    return normalize_description(_extract_json(content))
