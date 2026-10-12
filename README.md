# Manga Story Reader

This is my submission for the Manga Task. The goal is to read three consecutive manga pages, extract the story text in the correct reading order, and assign a speaker to each line.

I built a pipeline using OCR, a trained text classifier, and manga panel detection. I also tried different approaches for speaker identification and narration detection to see what actually improved the results.

## Approach

My pipeline has four main parts.

**1. Text extraction**

I used RapidOCR to extract text from manga pages. Instead of relying only on full-page OCR, I divided each page into overlapping regions to improve the chances of detecting smaller text.

I also removed duplicate OCR detections caused by overlapping regions.

**2. Story-text filtering**

OCR can detect sound effects, page numbers, labels, and other text that isn't part of the story.

To filter these out, I trained a logistic regression classifier using OCR text and position-based features. The model was trained using the labeled development data and is saved in `models/text_filter.joblib`.

**3. Text grouping and reading order**

OCR often detects separate lines of the same speech balloon as individual pieces of text, so I grouped nearby text fragments together.

My first reading-order method relied mainly on text coordinates. Later, I used a pretrained manga segmentation model to detect panels and speech balloons.

Using panel information improved the reading order, especially on pages where a simple right-to-left coordinate sort was insufficient.

**4. Speaker assignment**

This was the most difficult part of the task.

I initially used alternating speaker labels that remain consistent across the three pages of a sequence. I also experimented with character detection, speech-balloon association, and ResNet18 embeddings to identify characters visually.

However, the visual approaches were not reliable enough to improve the score consistently, so I kept the simpler speaker-assignment method in the selected submission.

## Dataset and Evaluation

The dataset contains 80 labeled development sequences and 15 unlabeled test sequences, with three pages in each sequence.

I split the development data into:

- 64 training sequences
- 16 validation sequences

I used a fixed random seed of 42 and kept all three pages of each sequence together to avoid data leakage.

I evaluated the different approaches using the official `dataset/score.py` script.

### Validation Results

| Approach | Text Order | Balanced Joint F1 |
|---|---:|---:|
| Original grouped OCR pipeline | 0.6798 | 0.2993 |
| Panel-center ordering | 0.7076 | 0.3075 |
| Panel-top ordering | **0.7145** | 0.3139 |
| Panel + balloon ordering | 0.7061 | **0.3187** |
| Panel-top with narration detection | 0.7145 | 0.3169 |

I selected **panel-top ordering** for my test submission because it gave the best text-order score while also improving balanced joint F1 over my original pipeline.

These results are from the validation split, not the hidden test set.

## Experiments and What I Learned

Some approaches worked better than others.

Tiled OCR, text filtering, and text grouping helped improve text extraction. Adding panel detection also gave a noticeable improvement in reading order.

I trained a narration classifier and combined it with visual caption detection. It correctly identified a small amount of narration on validation data, but it made no narration predictions on the test data at the selected threshold. I therefore did not force narration labels into the test submission.

I also tried comparing character appearances using pretrained ResNet18 embeddings. The problem was that visually similar characters sometimes received high similarity scores even when they were different people. Using these matches for speaker assignment reduced the validation score.

These experiments helped me understand why manga reading is more complicated than applying OCR to a normal document.

The experiment scripts and results are included in the repository.

## Setup

The project was developed on macOS using Python 3.14.8.

I used two separate Python environments during development: one for OCR and classification, and another for the YOLO-based vision pipeline.

Clone the repository:

```bash
git clone https://github.com/sam-tc/manga-story-reader.git
cd manga-story-reader
```

Create both environments:

```bash
python3 -m venv .venv
python3 -m venv ../manga-vision-env
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
../manga-vision-env/bin/python -m pip install -r requirements-vision.txt
```

The pretrained segmentation model is downloaded automatically from Hugging Face when first used.

Place the official dataset inside the `dataset/` folder, preserving the original directory structure and image paths.

Dataset: https://drive.google.com/drive/folders/1UDZ6es3VdDyxd3mEsBQBrrYa7fD2lWYp

## Running Inference

To run the complete pipeline on the test images:

```bash
bash scripts/run_test_inference.sh
```

This runs manga panel detection first, followed by OCR, text filtering, grouping, reading-order prediction, and speaker assignment.

The generated output is saved to:

`submissions/test_predictions_reproduced.jsonl`

My selected test predictions are saved in:

`submissions/test_predictions_panel.jsonl`

I tested the launcher on all 15 test sequences and confirmed that it reproduced the selected predictions exactly in my development environment.

To run panel-aware validation:

```bash
.venv/bin/python scripts/predict_panel.py --split val
```

To try the experimental narration-enabled version:

```bash
.venv/bin/python scripts/predict_panel.py --split val --narration
```

## Submission Format

The output contains one JSON object per sequence, with three lists representing the three pages.

Each text entry contains a `speaker` and `text` field. Character labels are consistent within each three-page sequence, and `NARRATION` is reserved for narrator text.

The selected test submission contains 15 sequences, 45 pages, and 419 extracted text entries. I checked the JSONL format, sequence IDs, page counts, and text preservation.

## Limitations and Future Improvements

The biggest limitation of my current approach is speaker identification. Alternating speaker labels do not actually recognize which character is speaking, especially when more than two characters appear.

OCR also struggles with stylized fonts, very small text, some punctuation-only balloons, and complicated page layouts. Narration and internal thoughts are another area where the system needs improvement.

If I continued this project, I would focus on training a model that uses speech-balloon tails, panel information, and character appearances together to identify speakers more accurately. I would also improve recognition of punctuation-only balloons and distinguish narration from internal thoughts more reliably.

## Tools and References

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR
- [scikit-learn](https://scikit-learn.org/) — trained text classifier
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO inference
- [Manga Page Element Segmentation](https://huggingface.co/anonimkaq4/manga-page-element-segmentation) — pretrained panel, character, and speech-balloon segmentation
- [PyTorch](https://pytorch.org/) — character-embedding experiments
- [YOLO Manga Bubble Detector](https://github.com/ebhon/YOLO-manga-bubble-detector) — experimental balloon and narration detection

I used AI assistance to help understand some concepts, write and debug code, and plan experiments. I ran the models and experiments locally, checked the outputs, and used the official validation scores to compare approaches.