# 🎬 Bollywood Movie Recommender System

Welcome to the **Bollywood Movie Recommender System** – a smart application that suggests similar Bollywood movies based on your favorite one.  
The system uses **machine learning (content-based filtering)** with a touch of Bollywood flair 🎥✨.  

🌐 **Live Demo:** https://movierecommenderbykc.streamlit.app/

## 🚀 About the Project
This project is built to showcase how **Data Science and Machine Learning** can enhance user experiences in entertainment.  
Instead of random suggestions, it uses **movie metadata** like genres, plot overviews, cast, and directors to find meaningful similarities between films.  

So, if you search for a movie like *Devdas*, you’ll instantly get recommendations such as *Madhumati*, *Pakeezah*, or *Mughal-E-Azam*.  

## ⚡ How It Works
1. **Data Preprocessing** → Cleaned Bollywood movie dataset and combined key features (genre, overview, cast, director).  
2. **Feature Engineering** → Applied **TF-IDF Vectorization** to convert text into numerical vectors.  
3. **Similarity Computation** → Used **Cosine Similarity** to measure closeness between movies.  
4. **Interactive App** → Built with **Streamlit**, where the user enters a movie name and receives top recommendations with posters.  
