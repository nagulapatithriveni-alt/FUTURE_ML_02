"""
src/train.py
------------
This module handles model training for BOTH tasks:

  Task 1 – Category Classification
    Input  : ticket_text (cleaned)
    Output : Technical Issue / Billing / Account Access / etc.

  Task 2 – Priority Prediction
    Input  : ticket_text (cleaned)
    Output : High / Medium / Low

We use a PIPELINE (from scikit-learn) which chains:
  ┌─────────────────────────────────────────────────────────┐
  │  Raw Text → TF-IDF Vectorizer → Logistic Regression     │
  └─────────────────────────────────────────────────────────┘

What is TF-IDF?
  TF  = Term Frequency      → how often a word appears in THIS ticket
  IDF = Inverse Doc Freq    → how rare the word is across ALL tickets
  Together they assign a score to each word that captures
  "how important is this word for THIS document?"

What is Logistic Regression?
  Despite the name, it's a CLASSIFICATION algorithm.
  It learns a weight for each word feature and predicts a class.
  Great starting model: fast, interpretable, and often very accurate.
"""

import os
import pickle
import pandas as pd
from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import MultinomialNB
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                       accuracy_score,
                                       confusion_matrix)
import numpy as np


# ──────────────────────────────────────────────────────────
# Helper: build one scikit-learn Pipeline
# ──────────────────────────────────────────────────────────
def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """
    Build a text-classification Pipeline.

    Parameters
    ----------
    model_type : "logistic"  → Logistic Regression (default, recommended)
                 "naive_bayes" → MultinomialNB (good baseline)

    The Pipeline has two steps:
      1. tfidf   : convert cleaned text to a numerical matrix
      2. clf     : the classifier that learns from that matrix
    """
    if model_type == "naive_bayes":
        classifier = MultinomialNB()
    else:
        # max_iter=1000 prevents "ConvergenceWarning" on small datasets
        classifier = LogisticRegression(max_iter=1000, random_state=42)

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,   # use top 5000 most informative words
            ngram_range=(1, 2),  # unigrams + bigrams ("not working", "server error")
            sublinear_tf=True,   # apply log scaling to term frequencies
        )),
        ("clf", classifier),
    ])


# ──────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame,
                 model_type: str = "logistic",
                 test_size: float = 0.20,
                 save_models: bool = True):
    """
    Train two models (category + priority) and return results.

    Parameters
    ----------
    df         : preprocessed DataFrame with 'clean_text', 'category', 'priority'
    model_type : "logistic" or "naive_bayes"
    test_size  : fraction of data held out for testing (0.20 = 20 %)
    save_models: if True, save pipelines as .pkl files in models/

    Returns
    -------
    dict with keys:
      cat_pipeline, pri_pipeline,
      X_test, y_cat_test, y_pri_test,
      cat_results, pri_results
    """

    X = df["clean_text"]

    # ── Split ONCE so both tasks share the same test set ───
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index,
        test_size=test_size,
        random_state=42,
        stratify=df["category"]   # keep class proportions balanced
    )

    y_cat_train = df.loc[idx_train, "category"]
    y_cat_test  = df.loc[idx_test,  "category"]
    y_pri_train = df.loc[idx_train, "priority"]
    y_pri_test  = df.loc[idx_test,  "priority"]

    print(f"\n📊 Train / Test split")
    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing  samples : {len(X_test)}")

    # ── Task 1: Category Classification ───────────────────
    print(f"\n🏷️  Training CATEGORY model ({model_type})...")
    cat_pipeline = build_pipeline(model_type)
    cat_pipeline.fit(X_train, y_cat_train)

    cat_preds   = cat_pipeline.predict(X_test)
    cat_acc     = accuracy_score(y_cat_test, cat_preds)
    cat_report  = classification_report(y_cat_test, cat_preds, output_dict=True)
    cat_cm      = confusion_matrix(y_cat_test, cat_preds,
                                   labels=sorted(df["category"].unique()))

    # Cross-validation gives a more honest accuracy estimate
    cat_cv_scores = cross_val_score(cat_pipeline, X, df["category"], cv=5)

    print(f"   Test Accuracy    : {cat_acc:.4f} ({cat_acc*100:.1f}%)")
    print(f"   CV Accuracy (5k) : {cat_cv_scores.mean():.4f} ± {cat_cv_scores.std():.4f}")

    # ── Task 2: Priority Prediction ────────────────────────
    print(f"\n🚨  Training PRIORITY model ({model_type})...")
    pri_pipeline = build_pipeline(model_type)
    pri_pipeline.fit(X_train, y_pri_train)

    pri_preds   = pri_pipeline.predict(X_test)
    pri_acc     = accuracy_score(y_pri_test, pri_preds)
    pri_report  = classification_report(y_pri_test, pri_preds, output_dict=True)
    pri_cm      = confusion_matrix(y_pri_test, pri_preds,
                                   labels=sorted(df["priority"].unique()))

    pri_cv_scores = cross_val_score(pri_pipeline, X, df["priority"], cv=5)

    print(f"   Test Accuracy    : {pri_acc:.4f} ({pri_acc*100:.1f}%)")
    print(f"   CV Accuracy (5k) : {pri_cv_scores.mean():.4f} ± {pri_cv_scores.std():.4f}")

    # ── Save models ────────────────────────────────────────
    if save_models:
        os.makedirs("models", exist_ok=True)
        with open("models/category_pipeline.pkl", "wb") as f:
            pickle.dump(cat_pipeline, f)
        with open("models/priority_pipeline.pkl", "wb") as f:
            pickle.dump(pri_pipeline, f)
        print("\n💾  Models saved to models/")

    return {
        "cat_pipeline"  : cat_pipeline,
        "pri_pipeline"  : pri_pipeline,
        "X_test"        : X_test,
        "y_cat_test"    : y_cat_test,
        "y_pri_test"    : y_pri_test,
        "cat_preds"     : cat_preds,
        "pri_preds"     : pri_preds,
        "cat_acc"       : cat_acc,
        "pri_acc"       : pri_acc,
        "cat_report"    : cat_report,
        "pri_report"    : pri_report,
        "cat_cm"        : cat_cm,
        "pri_cm"        : pri_cm,
        "cat_cv"        : cat_cv_scores,
        "pri_cv"        : pri_cv_scores,
        "cat_labels"    : sorted(df["category"].unique()),
        "pri_labels"    : sorted(df["priority"].unique()),
    }
