# models/save_model.py
import os, re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV = "../IMDB-Movie-Dataset(2023-1951).csv"  # run from models/ folder or adapt path
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

print("Loading CSV:", CSV)
df = pd.read_csv(CSV)

# normalize column names to lowercase
df.columns = [c.strip().lower() for c in df.columns]

# determine key columns (adapt if your CSV uses different names)
title_col = "movie_name" if "movie_name" in df.columns else ("title" if "title" in df.columns else None)
if title_col is None:
    raise ValueError("No title column found (movie_name/title) in CSV.")

# if you used 'classification' already, keep it; else combine genre+overview
if "classification" not in df.columns:
    # create classification from genre + overview
    g = df["genre"] if "genre" in df.columns else ""
    ov = df["overview"] if "overview" in df.columns else ""
    df["classification"] = (g.fillna("") + " " + ov.fillna(""))

# ensure cast and director exist
for col in ["cast", "director"]:
    if col not in df.columns:
        df[col] = ""

# make combined text
df["combined"] = (df["classification"].fillna("") + " " +
                  df["cast"].fillna("") + " " + df["director"].fillna("")).apply(clean_text)

print("Preparing TF-IDF matrix (this may take a minute)...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["combined"].astype(str))

print("Computing cosine similarity (this may take a minute)...")
cos_sim = cosine_similarity(X, X).astype(np.float32)  # float32 to save space

print("Saving artifacts...")
joblib.dump(vectorizer, os.path.join(OUT_DIR, "vectorizer.joblib"))
np.save(os.path.join(OUT_DIR, "cosine_sim.npy"), cos_sim)

# save meta (index mapping for titles + optional movie_id + year)
meta = pd.DataFrame({
    "idx": df.index,
    "movie_name": df[title_col].astype(str),
    "movie_id": df["movie_id"].astype(str) if "movie_id" in df.columns else df.index.astype(str),
    "year": df["year"].astype(str) if "year" in df.columns else ""
})
meta.to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)

print("Saved vectorizer.joblib, cosine_sim.npy and meta.csv in", OUT_DIR)
