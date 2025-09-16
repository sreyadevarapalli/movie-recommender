# src/recommender.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class MovieData:
    def __init__(self, movies_csv, ratings_csv):
        # movies_csv expected columns: movieId, title, genres
        # ratings_csv expected columns: userId, movieId, rating, timestamp (timestamp optional)
        self.movies = pd.read_csv(movies_csv)
        self.ratings = pd.read_csv(ratings_csv)
        self._prepare()

    def _prepare(self):
        # basic cleanup
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['title'] = self.movies['title'].astype(str)
        # make a combined text field for content-based
        self.movies['meta'] = self.movies['title'] + " " + self.movies['genres'].str.replace('|', ' ')
        # merge ratings with movies for convenience
        self.ratings = self.ratings.merge(self.movies[['movieId','title']], on='movieId', how='left')

class ContentRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['meta'])
        # precompute cosine similarity index
        # we'll use linear_kernel which is faster for TF-IDF
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_by_movie(self, title, topn=10):
        # find movie index
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        if title not in indices:
            return []
        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:topn+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices][['movieId','title','genres']]

    def recommend_for_user_by_likes(self, liked_titles, topn=10):
        # make user profile as mean of liked movie vectors
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        liked_idx = [indices[t] for t in liked_titles if t in indices]
        if not liked_idx:
            return pd.DataFrame(columns=['movieId','title','genres'])
        user_vec = self.tfidf_matrix[liked_idx].mean(axis=0)
        sims = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        best_idx = sims.argsort()[::-1]
        # filter out liked movies
        best_idx = [i for i in best_idx if i not in liked_idx]
        return self.movies.iloc[best_idx[:topn]][['movieId','title','genres']]

class CollaborativeRecommender:
    def __init__(self, ratings_df, movies_df, n_components=50):
        self.ratings = ratings_df
        self.movies = movies_df
        self.user_map = None
        self.item_map = None
        self.n_components = n_components
        self.svd = None
        self.item_factors = None
        self.user_factors = None
        self.knn = None
        self._build_matrix()

    def _build_matrix(self):
        # create user-item pivot
        users = self.ratings['userId'].unique()
        items = self.ratings['movieId'].unique()
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {m:i for i,m in enumerate(items)}
        n_users, n_items = len(users), len(items)
        rows = self.ratings['userId'].map(self.user_map)
        cols = self.ratings['movieId'].map(self.item_map)
        data = self.ratings['rating']
        self.ratin_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        # apply TruncatedSVD to item-user matrix (items as rows)
        # compute SVD on user-item -> to get latent user & item factors, do SVD on the sparse matrix transposed
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        # fit on item-feature space (transpose so items are rows)
        self.item_factors = self.svd.fit_transform(self.ratin_matrix.T)  # shape (n_items, n_components)
        self.user_factors = self.svd.transform(self.ratin_matrix)        # shape (n_users, n_components) approx
        # build a nearest neighbors for item similarity
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn.fit(self.item_factors)

    def recommend_items_similar_to(self, movieId, topn=10):
        if movieId not in self.item_map:
            return pd.DataFrame(columns=['movieId','title','genres'])
        idx = self.item_map[movieId]
        vec = self.item_factors[idx].reshape(1, -1)
        dists, inds = self.knn.kneighbors(vec, n_neighbors=topn+1)
        similar_idx = inds.flatten()[1:]  # skip itself
        inv_item_map = {v:k for k,v in self.item_map.items()}
        movie_ids = [inv_item_map[i] for i in similar_idx]
        return self.movies[self.movies['movieId'].isin(movie_ids)][['movieId','title','genres']]

    def predict_ratings_for_user(self, userId, topn=10):
        # return top-n predicted movieIds for a user (not yet rated)
        if userId not in self.user_map:
            return pd.DataFrame(columns=['movieId','title','genres'])
        uidx = self.user_map[userId]
        user_vector = self.user_factors[uidx]  # latent user vector
        # predicted scores = item_factors dot user_vector
        scores = self.item_factors.dot(user_vector)
        # map back to movieIds and filter seen
        inv_map = {v:k for k,v in self.item_map.items()}
        seen_movies = set(self.ratings[self.ratings['userId']==userId]['movieId'])
        ranked = np.argsort(scores)[::-1]
        recommendations = []
        for rank_idx in ranked:
            mid = inv_map[rank_idx]
            if mid in seen_movies:
                continue
            recommendations.append(mid)
            if len(recommendations) >= topn:
                break
        return self.movies[self.movies['movieId'].isin(recommendations)][['movieId','title','genres']]
