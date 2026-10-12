#!/usr/bin/env bash
set -euo pipefail

# Find the project folder regardless of where the script is run.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python environments for the two pipeline stages.
VISION_PYTHON="${VISION_PYTHON:-$ROOT/../manga-vision-env/bin/python}"
OCR_PYTHON="${OCR_PYTHON:-$ROOT/.venv/bin/python}"

# Stop with a useful message if an environment is missing.
if [[ ! -x "$VISION_PYTHON" ]]; then
    echo "Vision environment not found: $VISION_PYTHON"
    exit 1
fi

if [[ ! -x "$OCR_PYTHON" ]]; then
    echo "OCR environment not found: $OCR_PYTHON"
    exit 1
fi

echo "Stage 1: Detect manga panels and speech balloons"
"$VISION_PYTHON" scripts/export_character_panel_detections.py --split test

echo "Stage 2: OCR, text filtering and panel-aware ordering"
"$OCR_PYTHON" scripts/predict_panel.py \
    --split test \
    --output submissions/test_predictions_reproduced.jsonl

echo "Inference completed."
echo "Output: submissions/test_predictions_reproduced.jsonl"