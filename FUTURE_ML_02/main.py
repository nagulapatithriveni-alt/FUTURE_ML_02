"""
main.py
=======
FUTURE_ML_02: Support Ticket Classification
--------------------------------------------
Run this file to execute the COMPLETE pipeline:
  1. Load dataset
  2. Preprocess text
  3. Train models (category + priority)
  4. Evaluate and print metrics
  5. Generate all visualizations
  6. Run sample predictions

Usage:
  python main.py                  # default: Logistic Regression
  python main.py --model nb       # use Naive Bayes instead
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.metrics import classification_report

# ── Our own modules ───────────────────────────────────────
from src.preprocess import preprocess_dataframe
from src.train      import train_models
from src.evaluate   import generate_all_plots
from src.predict    import predict_batch, print_prediction


# ─────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Support Ticket Classifier")
parser.add_argument("--model", choices=["logistic", "nb"],
                    default="logistic",
                    help="Model type: 'logistic' (default) or 'nb' (Naive Bayes)")
args = parser.parse_args()
MODEL_TYPE = "naive_bayes" if args.model == "nb" else "logistic"


# ─────────────────────────────────────────────────────────
# STEP 0 – Banner
# ─────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║        FUTURE_ML_02: Support Ticket Classifier           ║
║        NLP + Scikit-learn Pipeline                       ║
╚══════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "support_tickets.csv")

if not os.path.exists(DATA_PATH):
    print("⚠️  Dataset not found. Generating it now...")
    import subprocess
    subprocess.run([sys.executable, os.path.join("data", "generate_dataset.py")],
                   check=True)

print(f"\n📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"\n   First 3 rows:")
print(df[["ticket_text", "category", "priority"]].head(3).to_string(index=False))


# ─────────────────────────────────────────────────────────
# STEP 2 – Preprocess
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("STEP 2: Text Preprocessing")
print("═" * 55)

df, cat_encoder, pri_encoder = preprocess_dataframe(df)

# Show a before/after example
sample = df.iloc[0]
print(f"\n   Example — Before : {sample['ticket_text'][:80]}")
print(f"             After  : {sample['clean_text'][:80]}")


# ─────────────────────────────────────────────────────────
# STEP 3 – Train Models
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print(f"STEP 3: Model Training  [{MODEL_TYPE.upper()}]")
print("═" * 55)

results = train_models(df, model_type=MODEL_TYPE, test_size=0.20,
                       save_models=True)


# ─────────────────────────────────────────────────────────
# STEP 4 – Evaluation Report
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("STEP 4: Evaluation Metrics")
print("═" * 55)

print(f"\n{'─'*20} CATEGORY CLASSIFICATION {'─'*10}")
print(f"Test Accuracy : {results['cat_acc']:.4f}  ({results['cat_acc']*100:.1f}%)")
print(f"CV  Accuracy  : {results['cat_cv'].mean():.4f} ± {results['cat_cv'].std():.4f}")
print("\nDetailed Classification Report:")
print(classification_report(results["y_cat_test"], results["cat_preds"]))

print(f"\n{'─'*20} PRIORITY PREDICTION {'─'*14}")
print(f"Test Accuracy : {results['pri_acc']:.4f}  ({results['pri_acc']*100:.1f}%)")
print(f"CV  Accuracy  : {results['pri_cv'].mean():.4f} ± {results['pri_cv'].std():.4f}")
print("\nDetailed Classification Report:")
print(classification_report(results["y_pri_test"], results["pri_preds"]))


# ─────────────────────────────────────────────────────────
# STEP 5 – Visualizations
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("STEP 5: Generating Visualizations")
print("═" * 55)

generate_all_plots(results, df)


# ─────────────────────────────────────────────────────────
# STEP 6 – Sample Predictions
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("STEP 6: Sample Predictions on Unseen Tickets")
print("═" * 55)

sample_tickets = [
    "I cannot login to my account. The password reset email never arrives.",
    "I was charged twice this month. Please refund the duplicate charge immediately.",
    "The app crashes every time I try to export my data. Very frustrating.",
    "Do you offer any discount for annual subscriptions?",
    "My account has been suspended without any warning. I need this fixed urgently!",
    "Where can I find the documentation for your REST API?",
]

batch_results = predict_batch(
    sample_tickets,
    results["cat_pipeline"],
    results["pri_pipeline"]
)

for r in batch_results:
    print_prediction(r)


# ─────────────────────────────────────────────────────────
# STEP 7 – Summary
# ─────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║  ✅  Pipeline Complete!                                   ║
╠══════════════════════════════════════════════════════════╣
║  📁 Saved Models    →  models/                           ║
║  📊 Saved Charts    →  outputs/                          ║
║  📓 Notebook        →  notebooks/analysis.ipynb          ║
╚══════════════════════════════════════════════════════════╝
""")
