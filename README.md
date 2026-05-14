# Multilingual Spell Error Detection using Transformer Models

## Overview

This project is a multilingual spell error detection system for Indian languages using transformer-based NLP models including MuRIL, mT5, and IndicTrans2.

The system analyzes sentences from multiple Indian languages and predicts whether a sentence likely contains spelling mistakes using confidence-based semantic and linguistic evaluation.

Instead of directly correcting sentences, the project focuses on identifying potentially incorrect sentences and generating confidence scores for each model.

---

# Supported Languages

* Hindi
* Bhojpuri
* Maithili
* Kannada
* Telugu

---

# Models Used

## MuRIL

A multilingual transformer model developed by Google specifically for Indian languages.

### Used For

* semantic anomaly detection
* contextual understanding
* multilingual sentence confidence scoring

---

## mT5

A multilingual text-to-text transformer model.

### Used For

* sequence confidence estimation
* sentence-level linguistic validation

---

## IndicTrans2

A multilingual Indian language transformer model developed by AI4Bharat.

### Used For

* multilingual robustness
* low-resource language evaluation

---

# Features

* Multilingual spell error detection
* Confidence-based sentence classification
* Transformer-based NLP pipeline
* Automatic dataset processing
* Wrong sentence identification
* Separate evaluation for each model
* Structured output generation

---

# Project Structure

```text
project/
│
├── dataset/
│   ├── hindi.txt
│   ├── bhojpuri.txt
│   ├── maithili.txt
│   ├── kannada.txt
│   └── telugu.txt
│
├── outputs/
│
├── wrong_sentences/
│
├── muril_detector.py
├── mt5_detector.py
├── indictrans_detector.py
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-link>
cd <repository-name>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```text
transformers
torch
sentencepiece
scikit-learn
numpy
```

---

# Dataset Format

The `dataset/` folder contains `.txt` files for each language.

Example:

```text
भारत एक महान देश है
भारत एक महन देश है
```

Each line represents one sentence.

---

# Running the Project

```bash
python main.py
```

---

# Output Files

## Model Results

Generated inside:

```text
outputs/
```

Example:

```text
Sentence 1 | CORRECT | Confidence: 0.9421
Sentence 2 | WRONG | Confidence: 0.5123
```

---

## Wrong Sentence Numbers

Generated inside:

```text
wrong_sentences/
```

Example:

```text
2
7
11
```

---

# Detection Logic

Each model generates a confidence score for every sentence.

If the confidence score falls below a predefined threshold, the sentence is classified as:

```text
WRONG
```

Otherwise:

```text
CORRECT
```

---

# Thresholds

| Model       | Threshold |
| ----------- | --------- |
| MuRIL       | 0.75      |
| mT5         | 0.65      |
| IndicTrans2 | 0.60      |

These thresholds can be tuned experimentally.

---

# Research Motivation

Spell error detection for multilingual Indian languages remains a challenging NLP problem due to:

* low-resource languages
* spelling variations
* rich morphology
* multilingual tokenization challenges
* code-mixed linguistic structures

This project explores transformer confidence behavior for multilingual spell error detection across Indian languages.

---

# Evaluation Metrics

The project can be extended using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

# Future Improvements

* Ensemble confidence scoring
* Word-level error localization
* OCR noise handling
* Grammar error detection
* Visualization dashboard
* Real-time API deployment
* Fine-tuned transformer models

---

# Applications

* Multilingual spell checking
* Indian language NLP research
* OCR post-processing
* Educational tools
* Low-resource language processing
* AI writing assistants

---

# References

## MuRIL

https://huggingface.co/google/muril-base-cased

## mT5

https://huggingface.co/google/mt5-small

## IndicTrans2

https://github.com/AI4Bharat/IndicTrans2

---

# Author

Developed as a multilingual NLP research and spell error detection project using transformer-based architectures for Indian languages.
