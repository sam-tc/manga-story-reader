# Manga Story Reader

An OCR-based machine learning pipeline that extracts story text from manga pages, arranges the extracted dialogue in approximate Japanese manga reading order, and assigns speaker labels across a sequence of three consecutive pages.

The project was developed for **Applied ML Task 2 — The Manga Task**.

## 1. Project Overview

The goal of this project is to process manga pages automatically and generate structured story dialogue.

Given three consecutive manga pages, the system attempts to:

- Detect and recognize English text in manga panels.
- Separate useful story text from sound effects and background text.
- Combine fragmented OCR detections into readable dialogue.
- Arrange the extracted text in an approximate manga reading order.
- Assign consistent speaker labels within the three-page sequence.
- Export the results in the required JSONL submission format.

The system runs locally using open-source OCR models and a trained machine learning classifier. It does not require a hosted AI inference API.

## 2. Dataset

The provided dataset contains:

| Split | Sequences | Pages |
|---|---:|---:|
| Development | 80 | 240 |
| Test | 15 | 45 |
| Total | 95 | 285 |

Every sequence consists of three manga pages.

The development data includes text and speaker annotations, while the test data is used to generate predictions for submission.

### Training and validation split

I created a deterministic split of the development sequences using random seed `42`.

- **Training:** 64 sequences (192 pages)
- **Validation:** 16 sequences (48 pages)

The split is performed at the sequence level so that pages from the same sequence never appear in both training and validation.

The sequence IDs are saved in:

- `experiments/splits/train_ids.json`
- `experiments/splits/val_ids.json`

## 3. Methodology

The solution combines pretrained OCR models, a trained text-relevance classifier, and geometric heuristics.

### 3.1 Tiled OCR

I initially tested OCR on entire manga pages, but the results were incomplete because manga images contain small dialogue balloons, irregular text placement, and stylized lettering.

To improve detection, I divided pages into four overlapping regions and ran **RapidOCR** on each region.

The resulting detections were mapped back to the original image coordinates and deduplicated to remove repeated detections caused by overlapping regions.

RapidOCR uses pretrained models executed locally with ONNX Runtime.

### 3.2 Story Text Filtering

OCR detects more than dialogue. It can also recognize sound effects, signs, and other text that should not necessarily appear in the final story transcription.

To address this, I trained a binary classifier that predicts whether an OCR detection is relevant story text.

The classifier uses:

- OCR-recognized text
- Word and character features
- OCR confidence
- Bounding-box geometry and position

I generated weak training labels by comparing OCR detections against the provided training annotations.

The training process produced:

| Category | Examples |
|---|---:|
| Positive story-text examples | 4,723 |
| Negative examples | 633 |
| Uncertain examples excluded | 1,480 |

The implementation uses scikit-learn's `DictVectorizer` and `LogisticRegression`.

The trained model is saved at:

`models/text_filter.joblib`

This text filter is the main learned component adapted to the provided manga dataset.

### 3.3 Text Grouping and Reading Order

OCR frequently separates a single sentence into several detections.

I implemented a geometry-based grouping method that combines nearby text fragments using their bounding boxes, vertical alignment, and spacing.

The grouped text is then ordered using approximate right-to-left and top-to-bottom manga reading conventions.

This improved the reading-order validation score considerably compared with the initial OCR baseline.

However, the method does not explicitly detect panels or speech balloons, so unusual panel layouts may still produce incorrect ordering.

### 3.4 Speaker Assignment

The final pipeline uses an alternating two-speaker baseline.

Speaker labels alternate between:

- `character1`
- `character2`

The assignment continues across all three pages of each sequence rather than restarting for every page.

This approach performed better on the official balanced speaker-aware validation metric than assigning all dialogue to a single speaker.

However, it is important to note that this is **not a visual character-recognition system**. It cannot reliably associate speech balloons with characters or distinguish scenes involving several speakers.

### 3.5 Narration Classification Experiment

I also trained a separate text-based classifier to identify narration.

The model was trained using 58 narration examples and 1,350 other story-text examples from the training split.

Although the classifier identified some narration-like passages, integrating it reduced the official validation score.

Therefore, I preserved the experiment and model for comparison but did not include it in the final inference pipeline.

The experimental model is located at:

`models/narration_detector.joblib`

## 4. Experiments and Results

All reported scores were calculated on the same held-out validation set of 16 development sequences using the official task evaluation script.

| Experiment | Text Order Score | Balanced Joint F1 |
|---|---:|---:|
| Initial OCR baseline | 0.4897 | 0.1596 |
| Learned text filtering | 0.5303 | 0.1790 |
| Text grouping with one speaker | 0.6798 | 0.2138 |
| **Two-speaker alternation (selected)** | **0.6798** | **0.2993** |
| Three-speaker alternation | 0.6798 | 0.2495 |
| Two-speaker baseline with narration classifier | 0.6798 | 0.2800 |

### Observations

**Text filtering:** Training a classifier to distinguish story text from other OCR detections improved both the text-order score and the balanced joint F1 score.

**Fragment grouping:** Combining nearby OCR detections produced the largest improvement in reading-order performance.

**Speaker assignment:** Alternating two speakers gave the best balanced joint F1 among the tested speaker baselines, although it is not a reliable method of identifying actual characters.

**Narration detection:** The narration experiment did not improve performance, demonstrating the difficulty of identifying narration using text alone.

The selected final pipeline achieved:

- **Text Order Score:** `0.6798`
- **Balanced Joint F1:** `0.2993`
- **Joint F1:** `0.4081`
- **Matched Token Coverage:** `0.7686`

These are validation scores, not hidden test results.

Detailed experiment predictions and evaluation outputs are available in `experiments/validation/`.

## 5. Project Structure

```text
manga-story-reader/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── inspect_labels.py
│   ├── create_split.py
│   ├── ocr_probe.py
│   ├── ocr_tiled_probe.py
│   ├── batch_ocr.py
│   ├── train_text_filter.py
│   ├── evaluate_baseline.py
│   ├── evaluate_text_filter.py
│   ├── evaluate_grouping.py
│   ├── evaluate_speaker_baselines.py
│   ├── analyze_speaker_errors.py
│   ├── evaluate_narration.py
│   └── predict.py
│
├── models/
│   ├── text_filter.joblib
│   └── narration_detector.joblib
│
├── experiments/
│   ├── splits/
│   ├── training/
│   └── validation/
│
├── submissions/
│   └── test_predictions.jsonl
│
├── requirements.txt
├── .gitignore
└── README.md
```

The `dataset/` directory and generated OCR caches are excluded from version control.

## 6. Installation and Setup

### Prerequisites

- Python 3.14.8 (version used for development and testing)
- Git
- Internet access for cloning the repository, installing dependencies, and obtaining the provided dataset

The pipeline was developed and tested on macOS.

### Step 1: Clone the repository

```bash
git clone https://github.com/sam-tc/manga-story-reader.git
cd manga-story-reader
```

The repository must be accessible to the person cloning it.

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses RapidOCR, ONNX Runtime, NumPy, Pillow, scikit-learn, SciPy, and Joblib.

### Step 4: Add the dataset

Download the official task dataset from:

https://drive.google.com/drive/folders/1UDZ6es3VdDyxd3mEsBQBrrYa7fD2lWYp

Arrange it inside the project as follows:

```text
dataset/
├── sequences.json
├── sample_submission.jsonl
├── score.py
├── development/
│   ├── labels.jsonl
│   └── images/
└── test/
    └── images/
```

Preserve the image paths and filenames from the official dataset.

## 7. Running Inference

### Generate test predictions

Run:

```bash
python scripts/predict.py --split test
```

The script processes all 15 test sequences and saves the results to:

```text
submissions/test_predictions.jsonl
```

### Specify a custom output location

```bash
python scripts/predict.py --split test --output /tmp/manga-test.jsonl
```

### Generate validation predictions

```bash
python scripts/predict.py --split val
```

By default, these predictions are saved to:

```text
experiments/validation/pipeline_predictions.jsonl
```

The inference script first checks whether cached OCR results are available. If a cache file does not exist, it runs RapidOCR directly on the corresponding image.

The trained text-filter model is loaded from `models/text_filter.joblib`.

## 8. Output Format

The system generates a JSONL file containing one JSON object per sequence.

Each object includes a `sequence_id` and a `pages` array containing predictions for the three pages.

Example:

```json
{
  "sequence_id": "example_sequence",
  "pages": [
    [
      {
        "speaker": "character1",
        "text": "Example dialogue."
      }
    ],
    [
      {
        "speaker": "character2",
        "text": "Another dialogue line."
      }
    ],
    []
  ]
}
```

The final test submission contains:

- 15 sequence records
- 45 pages
- 419 predicted text entries

The predictions were generated programmatically without manually editing the test outputs.

## 9. Reproducibility

The project includes:

- A deterministic sequence-level training and validation split.
- Training scripts for the text-relevance classifier.
- Saved trained model weights.
- Scripts and outputs for multiple validation experiments.
- A single inference script for generating final predictions.
- Pinned dependencies in `requirements.txt`.

I verified that rerunning test inference with the existing OCR cache produced a prediction file identical to the committed submission, as confirmed by matching SHA-256 hashes.

I also verified that fresh OCR extraction works on an official test image without using the saved OCR cache.

A complete reproduction on a separate machine has not yet been tested.

## 10. Limitations

The current approach has several limitations.

**Speaker identification:** The system uses alternating speaker labels rather than identifying characters visually. This can fail when more than two characters are speaking, when one character speaks multiple times consecutively, or when narration is present.

**Reading order:** The geometric ordering algorithm may struggle with complex manga layouts, overlapping balloons, and unusual panel arrangements.

**OCR accuracy:** Small text, stylized fonts, punctuation-only dialogue, and low-quality scans can cause missed or incorrect detections.

**Text filtering:** The classifier is trained using weak labels, so it may retain some sound effects or discard relevant dialogue.

**Narration detection:** The experimental narration classifier did not improve validation performance and was therefore excluded from the final system.

## 11. Future Improvements

Potential improvements include:

- Speech-balloon and panel detection.
- Speech-balloon tail analysis for identifying speakers.
- Character detection and visual tracking across pages.
- Better distinction between narration, thoughts, and dialogue.
- Reading-order prediction using learned layout features.
- Stronger handling of manga sound effects and stylized text.

These improvements would need to be evaluated on the same held-out validation set before being incorporated into the final pipeline.

## 12. Tools and References

This project uses the following tools and resources:

- **RapidOCR:** https://github.com/RapidAI/RapidOCR
- **ONNX Runtime:** https://onnxruntime.ai/
- **scikit-learn:** https://scikit-learn.org/
- **NumPy:** https://numpy.org/
- **Pillow:** https://pillow.readthedocs.io/

The manga images, annotation data, sample submission, and official evaluation script were provided as part of the recruitment task.

The pipeline integrates pretrained OCR with project-specific preprocessing, weak-label construction, text filtering, grouping, evaluation experiments, and inference logic.