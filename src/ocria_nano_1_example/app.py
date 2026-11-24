"""FastAPI OCR service backed by LlamaIndex OCRia-nano-1 through LiteParse."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from liteparse import LiteParse, models

log = logging.getLogger("ocria.api")

ALLOWED_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".bmp", ".webp", ".docx", ".pptx", ".xlsx",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_state: dict[str, Any] = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pull the OCRia-nano ONNX models once, before serving requests."""
    if os.getenv("OCRIA_AUTO_PULL", "1").lower() in {"1", "true", "yes"}:
        try:
            await asyncio.to_thread(models.pull, "ocria-nano")
            _state["ready"] = True
        except Exception as exc:  # noqa: BLE001 - report through /health
            _state["error"] = str(exc)
            log.exception("Could not pull OCRia-nano models")
    else:
        # Assumes the models were pulled earlier on this machine.
        _state["ready"] = True
    yield


app = FastAPI(title="OCRia-nano-1 OCR API", version="0.1.0", lifespan=lifespan)


def _jsonable(value: Any) -> Any:
    """Convert LiteParse result objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report service and model status."""
    return {
        "status": "ok",
        "ocr_backend": "ocria-nano",
        "models_ready": _state["ready"],
        "error": _state["error"],
    }


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    output_format: Literal["markdown", "text", "json"] = "markdown",
    extract_blocks: bool = False,
) -> dict[str, Any]:
    """Parse an uploaded document with OCRia-nano-1."""
    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"OCRia-nano models are not ready: {_state['error']}",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")

    try:
        parser = LiteParse(
            ocr_backend="ocria-nano",
            output_format=output_format,
            extract_blocks=extract_blocks,
            quiet=True,
        )
        result = await asyncio.to_thread(parser.parse, data)
    except Exception as exc:  # noqa: BLE001 - surface parse failures to the client
        log.exception("OCR failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    response: dict[str, Any] = {
        "filename": file.filename,
        "format": output_format,
        "text": result.text,
    }
    if extract_blocks:
        response["blocks"] = [
            _jsonable(block)
            for page in getattr(result, "pages", [])
            for block in (getattr(page, "blocks", None) or [])
        ]
    return response


def main() -> None:
    """Run the service locally with uvicorn."""
    import uvicorn

    uvicorn.run("ocria_nano_1_example.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
