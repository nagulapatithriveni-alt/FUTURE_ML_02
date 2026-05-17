"""
src/evaluate.py
---------------
All evaluation metrics and visualizations go here.

We generate 5 charts:
  1. Confusion Matrix  – Category model
  2. Confusion Matrix  – Priority model
  3. F1 Score per class – both tasks side by side
  4. CV Score distribution (box plot)
  5. Dataset class distribution (bar chart)

All charts are saved to outputs/ so you can include them in your report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works everywhere)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# Color palette (professional & colorblind-friendly)
# ──────────────────────────────────────────────────────────
PALETTE = {
    "primary"  : "#2563EB",   # blue
    "success"  : "#16A34A",   # green
    "warning"  : "#D97706",   # amber
    "danger"   : "#DC2626",   # red
    "purple"   : "#7C3AED",
    "gray"     : "#6B7280",
    "bg"       : "#F8FAFC",
}

PRIORITY_COLORS = {"High": PALETTE["danger"],
                   "Medium": PALETTE["warning"],
                   "Low": PALETTE["success"]}

CATEGORY_COLORS = [PALETTE["primary"], PALETTE["success"],
                   PALETTE["warning"], PALETTE["danger"], PALETTE["purple"]]


def _save(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"   📊 Saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────
# 1 & 2.  Confusion Matrix
# ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, labels: list,
                          title: str, filename: str):
    """
    Plot a labeled, color-coded confusion matrix.

    Rows    = Actual class
    Columns = Predicted class
    Diagonal cells = correct predictions (we want these high!)
    Off-diagonal   = mistakes
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Normalize to percentages for readability
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm_norm / row_sums * 100, 0)

    im = ax.imshow(cm_pct, interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=100)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of Actual Class", fontsize=10)

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Annotate each cell
    thresh = 50
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_pct[i, j] > thresh else "#1e293b"
            ax.text(j, i,
                    f"{int(cm[i,j])}\n({cm_pct[i,j]:.0f}%)",
                    ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    _save(fig, filename)


# ──────────────────────────────────────────────────────────
# 3.  Per-class F1 Score bar chart
# ──────────────────────────────────────────────────────────
def plot_f1_scores(cat_report: dict, pri_report: dict, filename: str = "f1_scores.png"):
    """
    Side-by-side bar charts showing F1 score for each class.
    F1 = harmonic mean of Precision and Recall.
    A high F1 means the model is both precise AND catches most cases.
    """
    # Extract per-class F1 (skip 'accuracy', 'macro avg', 'weighted avg')
    skip = {"accuracy", "macro avg", "weighted avg"}

    cat_f1 = {k: v["f1-score"] for k, v in cat_report.items() if k not in skip}
    pri_f1 = {k: v["f1-score"] for k, v in pri_report.items() if k not in skip}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor=PALETTE["bg"])
    fig.suptitle("F1 Score per Class", fontsize=14, fontweight="bold")

    # Category F1
    bars1 = ax1.barh(list(cat_f1.keys()), list(cat_f1.values()),
                     color=CATEGORY_COLORS[:len(cat_f1)], edgecolor="white",
                     linewidth=0.5, height=0.55)
    ax1.set_xlim(0, 1.15)
    ax1.set_title("Category Classification", fontweight="bold")
    ax1.set_xlabel("F1 Score")
    ax1.set_facecolor(PALETTE["bg"])
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars1, cat_f1.values()):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    # Priority F1
    p_colors = [PRIORITY_COLORS.get(k, PALETTE["gray"]) for k in pri_f1]
    bars2 = ax2.barh(list(pri_f1.keys()), list(pri_f1.values()),
                     color=p_colors, edgecolor="white",
                     linewidth=0.5, height=0.45)
    ax2.set_xlim(0, 1.15)
    ax2.set_title("Priority Prediction", fontweight="bold")
    ax2.set_xlabel("F1 Score")
    ax2.set_facecolor(PALETTE["bg"])
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, pri_f1.values()):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, filename)


# ──────────────────────────────────────────────────────────
# 4.  Cross-Validation score distribution
# ──────────────────────────────────────────────────────────
def plot_cv_scores(cat_cv, pri_cv, filename: str = "cv_scores.png"):
    """
    Box plot of 5-fold CV accuracy for both models.
    The box shows the middle 50% of scores; the line is the median.
    Tight, high boxes = consistent, reliable model.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    data   = [cat_cv * 100, pri_cv * 100]
    labels = ["Category\nClassification", "Priority\nPrediction"]
    colors = [PALETTE["primary"], PALETTE["danger"]]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Overlay actual data points (jittered)
    for i, (scores, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(scores))
        ax.scatter(i + jitter, scores, color=color, zorder=5,
                   edgecolors="white", linewidths=0.8, s=60, alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(50, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, filename)


# ──────────────────────────────────────────────────────────
# 5.  Dataset distribution (stacked bar)
# ──────────────────────────────────────────────────────────
def plot_dataset_distribution(df: pd.DataFrame,
                               filename: str = "dataset_distribution.png"):
    """
    Stacked bar: for each category, how many are High / Medium / Low priority?
    This helps spot class imbalance — if one combination is very rare the
    model might struggle with it.
    """
    pivot = df.groupby(["category", "priority"]).size().unstack(fill_value=0)
    # Reorder priority columns
    for col in ["High", "Medium", "Low"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["High", "Medium", "Low"]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bottom = np.zeros(len(pivot))
    for priority, color in PRIORITY_COLORS.items():
        vals = pivot[priority].values
        bars = ax.bar(pivot.index, vals, bottom=bottom,
                      color=color, label=priority,
                      edgecolor="white", linewidth=0.7)
        # Label bars if big enough
        for bar, v in zip(bars, vals):
            if v >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(v)), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += vals

    ax.set_xlabel("Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Tickets", fontsize=11)
    ax.set_title("Dataset Distribution: Category × Priority",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=15, ha="right", fontsize=9)
    ax.legend(title="Priority", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)

    _save(fig, filename)


# ──────────────────────────────────────────────────────────
# 6.  Accuracy summary bar
# ──────────────────────────────────────────────────────────
def plot_accuracy_summary(cat_acc: float, pri_acc: float,
                           filename: str = "accuracy_summary.png"):
    """Simple bar showing overall test accuracy for each task."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    tasks  = ["Category\nClassification", "Priority\nPrediction"]
    accs   = [cat_acc * 100, pri_acc * 100]
    colors = [PALETTE["primary"], PALETTE["danger"]]

    bars = ax.bar(tasks, accs, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                color=bar.get_facecolor())

    ax.set_ylim(0, 115)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.axhline(y=100, color=PALETTE["gray"], linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)

    _save(fig, filename)


# ──────────────────────────────────────────────────────────
# Convenience: run all plots at once
# ──────────────────────────────────────────────────────────
def generate_all_plots(results: dict, df: pd.DataFrame):
    print("\n📈 Generating visualizations...")

    plot_confusion_matrix(results["cat_cm"], results["cat_labels"],
                          "Confusion Matrix — Category Classification",
                          "confusion_matrix_category.png")

    plot_confusion_matrix(results["pri_cm"], results["pri_labels"],
                          "Confusion Matrix — Priority Prediction",
                          "confusion_matrix_priority.png")

    plot_f1_scores(results["cat_report"], results["pri_report"])

    plot_cv_scores(results["cat_cv"], results["pri_cv"])

    plot_dataset_distribution(df)

    plot_accuracy_summary(results["cat_acc"], results["pri_acc"])

    print(f"\n✅ All charts saved to outputs/")
