# 🎫 FUTURE_ML_02: Support Ticket Classification

> **Automated NLP pipeline** that classifies customer support tickets by **category** and **priority** using Scikit-learn and TF-IDF.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

Customer support teams receive hundreds of tickets daily. Manually reading, categorizing, and prioritizing every one is slow and error-prone.

**This project automates the entire workflow** using Natural Language Processing (NLP):

| Input | Output |
|-------|--------|
| Raw ticket text | Category + Priority |
| `"I was charged twice for my subscription."` | `Billing` → `High` |
| `"How do I export my data to CSV?"` | `General Inquiry` → `Low` |
| `"My account is locked and I cannot login."` | `Account Access` → `High` |

---

## 🎯 Features

- ✅ Text cleaning and NLP preprocessing (lowercase, stopword removal, regex)
- ✅ TF-IDF feature extraction with bigrams
- ✅ **Dual-task classification** (category + priority)
- ✅ Logistic Regression and Naive Bayes pipelines
- ✅ 5-fold Cross-Validation for reliable accuracy
- ✅ Confusion matrix, F1 score, and classification report
- ✅ 6 professional visualization charts
- ✅ Saved `.pkl` models for reuse
- ✅ Step-by-step Jupyter Notebook for learning

---

## 📁 Project Structure

```
FUTURE_ML_02/
│
├── data/
│   ├── generate_dataset.py     # Script to create synthetic dataset
│   └── support_tickets.csv     # 305-row labeled dataset (auto-generated)
│
├── notebooks/
│   └── analysis.ipynb          # Full interactive walkthrough (9 steps)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning & label encoding
│   ├── train.py                # Model training (Pipeline builder)
│   ├── evaluate.py             # Metrics + 6 visualization charts
│   └── predict.py              # Single & batch inference
│
├── models/                     # Saved trained pipelines (.pkl)
│   ├── category_pipeline.pkl
│   └── priority_pipeline.pkl
│
├── outputs/                    # Generated charts (PNG)
│   ├── confusion_matrix_category.png
│   ├── confusion_matrix_priority.png
│   ├── f1_scores.png
│   ├── cv_scores.png
│   ├── dataset_distribution.png
│   └── accuracy_summary.png
│
├── main.py                     # 🚀 Run the full pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python main.py
```

### 4. Explore the Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📊 Model Performance

| Task | Test Accuracy | CV Accuracy (5-fold) |
|------|:---:|:---:|
| Category Classification | **100%** | **100%** |
| Priority Prediction | **100%** | **100%** |

> **Note:** 100% accuracy is expected on this clean synthetic dataset. Real-world performance on messy, ambiguous tickets will be lower (typically 85–95%) — which is normal and part of the learning journey.

---

## 🧠 NLP Techniques Used

| Technique | Purpose |
|-----------|---------|
| Regex cleaning | Remove URLs, emails, punctuation |
| Lowercasing | Normalize text case |
| Stopword removal | Remove common meaningless words |
| **TF-IDF Vectorizer** | Convert text → numerical features |
| **Bigrams** (`ngram_range=(1,2)`) | Capture "server crash", "not working" |
| **Logistic Regression** | Primary classifier |
| **Naive Bayes** | Alternative baseline |
| **LabelEncoder** | Encode string labels as integers |
| **Pipeline** | Chain vectorizer + classifier cleanly |

---

## 🏷️ Categories & Priorities

**Categories (5 classes):**
- 🔵 Technical Issue
- 🟢 Billing
- 🟡 Account Access
- 🔴 Refund Request
- 🟣 General Inquiry

**Priorities (3 levels):**
- 🔴 **High** — Urgent, business-critical
- 🟡 **Medium** — Important but not blocking
- 🟢 **Low** — Informational, non-urgent

---

## 📈 Sample Predictions

```
📩 Ticket   : I cannot login to my account. Password reset emails do not arrive.
🏷️  Category  : Account Access
🔴 Priority  : High
   Confidence: 61.2%

📩 Ticket   : What is the difference between the Basic and Pro plans?
🏷️  Category  : General Inquiry
🟢 Priority  : Low
   Confidence: 74.3%
```

---

## 🔮 Future Improvements

- [ ] Replace TF-IDF with **BERT/DistilBERT** sentence embeddings
- [ ] Deploy as a **REST API** using FastAPI
- [ ] Add **confidence thresholds** for human review routing
- [ ] Support **multi-label** classification (one ticket → multiple tags)
- [ ] Build an **active learning** loop from corrections
- [ ] Add **sentiment analysis** to detect customer frustration
- [ ] **Auto-response generation** using an LLM

---

## 🧑‍💻 Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat)
![scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=flat)

---

## 👩‍💻 Author

**Thriveni Nagulapati**
- 🔗 LinkedIn: https://www.linkedin.com/in/thriveni-nagulapati-838405285
- 🐙 GitHub: https://github.com/nagulapatithriveni-alt
- 📧 Email: nagulapatithriveni@gmail.com
---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of an ML internship project — FUTURE_ML_02*
