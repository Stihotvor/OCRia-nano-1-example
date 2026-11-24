# OCRia-nano-1 OCR API

A small FastAPI service that parses documents with
[OCRia-nano-1](https://huggingface.co/llamaindex/OCRia-nano-1), the 8.1M-parameter
ONNX OCR and document-layout pipeline from LlamaIndex. The models run locally on CPU
through [LiteParse](https://github.com/run-llama/liteparse); no cloud OCR API is used.

## Supported input

PDF, PNG, JPEG, TIFF, BMP, WebP, DOCX, PPTX and XLSX, up to 50 MB. Script detection is
automatic (Latin, CJK, RTL and a world-script recognizer). Output can be Markdown,
plain text or JSON with bounding boxes; tables and layout blocks can be returned too.

## Setup

```bash
uv sync
uv run uvicorn ocria_nano_1_example.app:app --port 8055 --reload
```

On first startup LiteParse pulls the `ocria-nano` models (about 60-75 MB, cached on
disk afterwards). To skip the automatic pull when the models are already present:

```bash
OCRIA_AUTO_PULL=0 uv run uvicorn ocria_nano_1_example.app:app --port 8055
```

Alternatively, run `uv run ocria-api`, which listens on port 8055 by default. Set
`PORT` to use a different port with that command.

## Endpoints

- `GET /health` — service status and whether the models are ready.
- `POST /ocr` — upload a document via multipart form field `file`.

Query parameters for `POST /ocr`:

| Parameter | Values | Default |
| --- | --- | --- |
| `output_format` | `markdown`, `text`, `json` | `markdown` |
| `extract_blocks` | `true`, `false` | `false` |

## Example

```bash
curl -X POST "http://localhost:8055/ocr?output_format=markdown" \
  -F "file=@/path/to/scan.pdf"
```

With layout blocks as JSON:

```bash
curl -X POST "http://localhost:8055/ocr?output_format=json&extract_blocks=true" \
  -F "file=@/path/to/scan.pdf"
```

The response contains `filename`, `format`, `text` and, when requested, `blocks`.

## License note

OCRia-nano-1 uses the `ocria-nano-1-license` (based on AI Pubs Open RAIL-M). It permits
research, personal use and commercial use by organizations under **$2M annual revenue
and $2M total funding**; a commercial license is required above those thresholds and
for products that compete with LlamaParse/LlamaCloud. LiteParse itself is Apache 2.0.
Review the [model card](https://huggingface.co/llamaindex/OCRia-nano-1) before use.
