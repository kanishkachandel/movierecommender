import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB-Movie-Dataset(2023-1951).csv")
    
    # Agar 'classification' column nahi hai, toh bana lo
    if "classification" not in df.columns:
        genre_col = "genre" if "genre" in df.columns else ""
        overview_col = "overview" if "overview" in df.columns else "description" if "description" in df.columns else ""
        
        df["classification"] = (
            df.get(genre_col, "").fillna("").astype(str) + " " +
            df.get(overview_col, "").fillna("").astype(str)
        )
    else:
        df["classification"] = df["classification"].fillna("")
    
    return df


data = load_data()

# --- Vectorization ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["classification"])
cosine_sim = cosine_similarity(tfidf_matrix)

# Index mapping (lowercase to fix case-sensitivity issue)
indices = pd.Series(data.index, index=data["movie_name"].str.lower())
indices = pd.Series(data.index, index=data["movie_name"].str.lower()).drop_duplicates()

def recommend(movie_title, n=5):
    movie_title = movie_title.lower()
    if movie_title not in indices:
        return f"Sorry no idea about {movie_title}"
    
    #idx = int(indices[movie_title])     #int to fix valueerror(error occurs when passed array rather index)
    idx_series = indices[movie_title]

# If multiple matches, take the first one
    if isinstance(idx_series, pd.Series):
        idx = idx_series.iloc[0]
    else:
        idx = idx_series
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return data.iloc[movie_indices][["movie_name", "year", "director"]]

# --- Streamlit UI ---
st.title("🎬 Bollywood Movie Recommender System")

movie = st.text_input("Enter a Bollywood Movie Name:")

if st.button("Recommend"):
    results = recommend(movie, 5)
    if results is not None and not results.empty:
        st.write("### Recommended Movies 🎥")
        st.dataframe(results)
    else:
        st.warning("Movie not found in dataset.")
