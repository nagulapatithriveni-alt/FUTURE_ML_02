"""
src/preprocess.py
-----------------
All text cleaning and preprocessing lives here.

What this module does:
1. Converts text to lowercase              → "Hello WORLD" → "hello world"
2. Removes punctuation/numbers/symbols    → "ticket #123!" → "ticket"
3. Strips extra whitespace               → "hello   world" → "hello world"
4. Removes very common words (stopwords) → "I am the customer" → "customer"

Why preprocessing matters:
  Machine learning models work on NUMBERS not words.
  Before converting, we clean the text so that:
  - "CRASH" and "crash" are treated as the same word
  - "ticket!!!" and "ticket" are treated the same
  - Common words like "the", "is", "a" don't skew the model
"""

import re
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# Simple English stopwords (common words with little meaning)
# We don't use NLTK here so the project works without internet.
# ─────────────────────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","also","every","even","get","got","may","might","much","one","two",
    "three","us","would","could","hi","hello","dear","please","thank","thanks",
    "good","morning","afternoon","evening","need","want","help","support",
    "like","know","make","use","using","used","still","already","back","time",
    "day","days","week","try","tried","trying",
}


def clean_text(text: str) -> str:
    """
    Clean a single ticket string.

    Steps:
      1. Lowercase everything
      2. Remove URLs (http://...)
      3. Remove email addresses
      4. Remove punctuation and digits
      5. Remove extra spaces
      6. Remove stopwords
      7. Remove very short words (≤ 2 characters)

    Parameters
    ----------
    text : str  →  raw ticket string

    Returns
    -------
    str  →  cleaned string
    """
    if not isinstance(text, str):
        return ""

    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Step 3: remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 4: remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 5: collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 6 & 7: remove stopwords AND very short tokens
    tokens = [word for word in text.split()
              if word not in STOPWORDS and len(word) > 2]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing to the dataset DataFrame.

    What this adds to the DataFrame:
      - 'clean_text'       : cleaned version of ticket_text
      - 'category_encoded': integer label for category  (0,1,2,...)
      - 'priority_encoded': integer label for priority  (0,1,2)

    Parameters
    ----------
    df : pd.DataFrame with columns ['ticket_text', 'category', 'priority']

    Returns
    -------
    pd.DataFrame  →  enriched with 3 new columns
    """
    print("🔧 Preprocessing text...")

    # Clean the raw ticket text
    df = df.copy()
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    # ── Label Encoding ───────────────────────────────────────
    # LabelEncoder turns categories into numbers:
    #   "Billing" → 0, "General Inquiry" → 1, "Refund Request" → 3, etc.
    # We save the encoders so we can DECODE predictions later.

    cat_encoder  = LabelEncoder()
    pri_encoder  = LabelEncoder()

    df["category_encoded"] = cat_encoder.fit_transform(df["category"])
    df["priority_encoded"] = pri_encoder.fit_transform(df["priority"])

    print(f"   Category classes : {list(cat_encoder.classes_)}")
    print(f"   Priority classes : {list(pri_encoder.classes_)}")
    print(f"   Total rows       : {len(df)}")
    print(f"   Empty clean_text : {(df['clean_text'] == '').sum()}")

    return df, cat_encoder, pri_encoder
