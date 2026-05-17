"""
src/predict.py
--------------
Use this module to classify NEW support tickets once the models are trained.

Three ways to use:
  1. predict_single()  →  classify one ticket at a time
  2. predict_batch()   →  classify a list of tickets
  3. load_and_predict()→  load saved models from disk and predict
"""

import pickle
import os
from src.preprocess import clean_text


def predict_single(ticket_text: str,
                   cat_pipeline,
                   pri_pipeline) -> dict:
    """
    Predict category and priority for ONE ticket.

    Parameters
    ----------
    ticket_text  : the raw customer support ticket string
    cat_pipeline : trained sklearn Pipeline for category
    pri_pipeline : trained sklearn Pipeline for priority

    Returns
    -------
    dict with:
      'original'    : original ticket text
      'clean'       : cleaned text used for prediction
      'category'    : predicted category string
      'priority'    : predicted priority string
      'cat_proba'   : confidence scores for each category class
      'pri_proba'   : confidence scores for each priority class
    """
    cleaned = clean_text(ticket_text)

    category = cat_pipeline.predict([cleaned])[0]
    priority = pri_pipeline.predict([cleaned])[0]

    # predict_proba gives confidence % for each class
    cat_proba = dict(zip(cat_pipeline.classes_,
                         cat_pipeline.predict_proba([cleaned])[0]))
    pri_proba = dict(zip(pri_pipeline.classes_,
                         pri_pipeline.predict_proba([cleaned])[0]))

    return {
        "original" : ticket_text,
        "clean"    : cleaned,
        "category" : category,
        "priority" : priority,
        "cat_proba": {k: round(v, 3) for k, v in cat_proba.items()},
        "pri_proba": {k: round(v, 3) for k, v in pri_proba.items()},
    }


def predict_batch(ticket_list: list,
                  cat_pipeline,
                  pri_pipeline) -> list:
    """
    Predict for a list of ticket strings.
    Returns a list of dicts (same format as predict_single).
    """
    return [predict_single(t, cat_pipeline, pri_pipeline) for t in ticket_list]


def load_and_predict(ticket_text: str,
                     model_dir: str = "models") -> dict:
    """
    Load saved models from disk and predict for one ticket.
    Use this when you just want to run inference without re-training.
    """
    cat_path = os.path.join(model_dir, "category_pipeline.pkl")
    pri_path = os.path.join(model_dir, "priority_pipeline.pkl")

    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"Model not found at {cat_path}. Run main.py first.")

    with open(cat_path, "rb") as f:
        cat_pipeline = pickle.load(f)
    with open(pri_path, "rb") as f:
        pri_pipeline = pickle.load(f)

    return predict_single(ticket_text, cat_pipeline, pri_pipeline)


def print_prediction(result: dict):
    """Pretty-print a single prediction result."""
    priority_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    icon = priority_icons.get(result["priority"], "⚪")

    print("\n" + "─" * 55)
    print(f"📩 Ticket   : {result['original'][:80]}{'...' if len(result['original']) > 80 else ''}")
    print(f"🏷️  Category  : {result['category']}")
    print(f"{icon} Priority  : {result['priority']}")
    print(f"🧹 Cleaned   : {result['clean'][:60]}...")
    print("\n   Category confidence:")
    for cls, prob in sorted(result["cat_proba"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"     {cls:<20} {bar:<20} {prob:.1%}")
    print("\n   Priority confidence:")
    for cls, prob in sorted(result["pri_proba"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"     {cls:<10} {bar:<20} {prob:.1%}")
    print("─" * 55)
