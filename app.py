import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from fastapi import FastAPI
from pydantic import BaseModel
import re

# ---------- Step 1: Load Dataset ----------
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')
movies['title'] = movies['title'].fillna('')

# ---------- Step 2: TF-IDF Vectorization ----------
genre_vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
genre_features = genre_vectorizer.fit_transform(movies['genres'])

title_vectorizer = TfidfVectorizer(stop_words='english')
title_features = title_vectorizer.fit_transform(movies['title'])

combined_features = hstack([genre_features, title_features])
combined_features = normalize(combined_features)

# ---------- Step 3: KNN Model ----------
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=30)
knn.fit(combined_features)

# ---------- Helper: Strip Year ----------
def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()

# ---------- FastAPI Setup ----------
app = FastAPI()

class RecommendRequest(BaseModel):
    title: str
    top_n: int = 10

@app.post("/recommend")
def recommend_movies_api(req: RecommendRequest):
    title = req.title
    top_n = req.top_n

    if title not in movies['title'].values:
        return {"error": f"Movie '{title}' not found."}

    idx = movies[movies['title'] == title].index[0]
    movie_vector = combined_features[idx]
    distances, indices = knn.kneighbors(movie_vector, n_neighbors=30)

    input_base = clean_title(title)
    recommended = []

    for i in indices.flatten():
        rec_title = movies.iloc[i]['title']
        rec_base = clean_title(rec_title)
        if rec_title != title and rec_base != input_base:
            recommended.append({
                "title": rec_title,
                "genres": movies.iloc[i]['genres']
            })
        if len(recommended) == top_n:
            break

    return {"input": title, "recommendations": recommended}

    @app.get("/")
def root():
    return {"message": "FastAPI is working!"}

