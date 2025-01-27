import streamlit as st
import pandas as pd
import pickle
import os
from surprise import Dataset, Reader


# Define paths to local data files
RATINGS_PATH = "data/u.data"
MOVIES_PATH = "data/u.item"


# Function to load data from local files
@st.cache_data
def load_data():
    try:
        # Load ratings dataset
        ratings = pd.read_csv(
            RATINGS_PATH,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Load movies dataset
        movies = pd.read_csv(
            MOVIES_PATH,
            sep='|',
            encoding='ISO-8859-1',
            names=['movie_id', 'title'],
            usecols=[0, 1]  # Only read the first two columns
        )
        
        # Merge datasets
        return pd.merge(ratings, movies, on='movie_id')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# Load the data
data = load_data()

# Stop the app if data is missing
if data.empty:
    st.error("No data available. Please ensure the data files are in the correct location.")
    st.stop()

# Streamlit app UI
st.title("Movie Recommender System")
st.write("Rate a few movies, and we'll give you personalized recommendations!")

# Find the top 5 most rated movies
most_rated_movies = data.groupby('title').size().sort_values(ascending=False).head(5)
top_movie_titles = most_rated_movies.index.tolist()

# Allow users to rate these movies
st.write("Please rate the following movies:")
user_ratings = {}
for movie in top_movie_titles:
    avg_rating = data[data['title'] == movie]['rating'].mean()
    user_ratings[movie] = st.slider(f"Rate {movie}", 1, 5, int(avg_rating) or 3)

if st.button("Get Recommendations"):
    new_user_ratings = []
    for movie, rating in user_ratings.items():
        if not data[data['title'] == movie].empty:
            movie_id = data[data['title'] == movie].iloc[0]['movie_id']
            new_user_ratings.append({
                'user_id': 'new_user',
                'movie_id': movie_id,
                'rating': rating
            })

    if new_user_ratings:
        new_user_df = pd.DataFrame(new_user_ratings)
        maindf_with_new_user = pd.concat(
            [data[['user_id', 'movie_id', 'rating']], new_user_df],
            ignore_index=True
        )

        reader = Reader(rating_scale=(1, 5))
        data_with_new_user = Dataset.load_from_df(maindf_with_new_user[['user_id', 'movie_id', 'rating']], reader)
        trainset = data_with_new_user.build_full_trainset()

        with open('final_svd_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        loaded_model.fit(trainset)

        all_movie_ids = data['movie_id'].unique()
        rated_movie_ids = [row['movie_id'] for row in new_user_ratings]
        unrated_movie_ids = [movie for movie in all_movie_ids if movie not in rated_movie_ids]

        recommendations = []
        for movie_id in unrated_movie_ids:
            pred = loaded_model.predict('new_user', movie_id)
            recommendations.append((movie_id, pred.est))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        st.write("Top 10 Recommendations:")
        for movie_id, rating in recommendations[:10]:
            movie_title = data[data['movie_id'] == movie_id].iloc[0]['title']
            st.write(f"{movie_title} (Predicted Rating: {rating:.2f})")
    else:
        st.error("No user ratings were provided. Please rate at least one movie.")