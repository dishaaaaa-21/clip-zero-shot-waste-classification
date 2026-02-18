# Zero-Shot Waste Classification using CLIP

This project investigates zero-shot waste classification using Vision–Language Models, specifically CLIP (ViT-B/32). The system performs semantic image classification without task-specific training data and evaluates prompt engineering, prompt ensembling, and linear probing strategies.

---

## 🚀 Project Overview

Traditional CNN-based waste classification systems require large labeled datasets and retraining when categories change. This project explores the use of CLIP, a pretrained multimodal model, to perform:

- Zero-shot classification using natural language prompts
- Prompt ensembling for improved semantic alignment
- Linear probing on CLIP embeddings
- Comparative evaluation against supervised baselines

---

## 🧠 Methodology

### 1️⃣ Zero-Shot Classification
Images are encoded using CLIP’s image encoder. Class labels are converted into natural language prompts and encoded using CLIP’s text encoder. Classification is performed via cosine similarity in the shared embedding space.

### 2️⃣ Prompt Ensembling
Multiple prompt templates are averaged per class to improve robustness and semantic alignment.

### 3️⃣ Linear Probing
CLIP image embeddings are used to train a logistic regression classifier to evaluate feature separability.

---

## 📊 Results (TrashNet Dataset)

| Method | Accuracy |
|--------|----------|
| Zero-Shot CLIP | ~70% |
| Linear Probe on CLIP | ~91% |

These results demonstrate strong semantic generalization without task-specific fine-tuning.

---
