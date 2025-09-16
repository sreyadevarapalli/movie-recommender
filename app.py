# app.py
import streamlit as st
import pandas as pd
from src.recommender import MovieData, ContentRecommender, CollaborativeRecommender

@st.cache_data
def load_data(movies_path, ratings_path):
    md = MovieData(movies_path, ratings_path)
    return md

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Mini Project 5 — Movie Recommendation System")

# Edit these if your CSV names differ
MOVIES_CSV = "data/movies.csv"
RATINGS_CSV = "data/ratings.csv"

md = load_data(MOVIES_CSV, RATINGS_CSV)
st.sidebar.header("Choose model")
model_type = st.sidebar.radio("Model", ("Content-based", "Collaborative", "Popularity"))

if model_type == "Content-based":
    st.header("Content-based Recommendations")
    st.write("Pick a movie you like:")
    selected = st.selectbox("Movie", md.movies['title'].tolist())
    if st.button("Recommend"):
        content_rec = ContentRecommender(md.movies)
        res = content_rec.recommend_by_movie(selected, topn=10)
        st.table(res.reset_index(drop=True))

elif model_type == "Collaborative":
    st.header("Collaborative (SVD + NearestNeighbors)")
    st.write("Provide a user id to get personalized recommendations:")
    user_ids = sorted(md.ratings['userId'].unique().tolist())
    user = st.selectbox("User ID", user_ids)
    if st.button("Get recommendations"):
        collab = CollaborativeRecommender(md.ratings, md.movies, n_components=50)
        res = collab.predict_ratings_for_user(user, topn=10)
        st.table(res.reset_index(drop=True))

else:
    st.header("Popularity baseline")
    topk = md.ratings.groupby('movieId').agg({'rating':['mean','count']})
    topk.columns = ['mean_rating','count']
    topk = topk.merge(md.movies, left_on='movieId', right_on='movieId')
    topk = topk.sort_values(['count','mean_rating'], ascending=False).head(20)
    st.table(topk[['movieId','title','genres','mean_rating','count']].reset_index(drop=True))
