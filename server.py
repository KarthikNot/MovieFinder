import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from src.constants import *
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(selected_movies : str, top_n : int) -> List[str]:
    try:
        
        if not os.path.exists(PREPROCESSED_DATASET_PATH):
            st.info("Preprocessed dataset doesn't exist.")
            return []
        
        preprocessed_data = pd.read_csv(PREPROCESSED_DATASET_PATH, encoding = 'utf-8')
        
        movie_id = preprocessed_data[preprocessed_data['title'] == selected_movies]['id'].values[0]
        st.info(f"Selected movie_id: {movie_id}")
        
        row_idx = preprocessed_data[preprocessed_data['id'] == movie_id].index
        st.info(f"got the movie: {row_idx}")
        
        with open(VECTORIZED_MATRIX, 'rb') as file:
            vectors = pickle.load(file)

        sim_scores = cosine_similarity(vectors[row_idx].reshape(1, -1), vectors).flatten()

        sim_scores[row_idx] = -1

        top_indices = np.argsort(sim_scores)[-top_n:][::-1]

        top_movies = preprocessed_data.loc[top_indices, ['title', 'poster_path']]

        st.info(f"Top Indices: {[int(index) for index in top_indices]}")
        st.info(f"Top Movies: {[str(movie) for movie in top_movies]}")

        return top_movies
    except Exception as e:
        st.error(f"An error occured: {str(e)}")
    return []


@st.cache_data
def get_movie_names():
    try:
        if not os.path.exists(PREPROCESSED_DATASET_PATH):
            return []
        df = pd.read_csv(PREPROCESSED_DATASET_PATH, encoding='utf-8')
        return sorted(df['title'].unique())
    except Exception as e:
        st.error(f"An error occured: {str(e)}")
        return []


def main():
    st.set_page_config(
        page_title='Movie Finder',
        page_icon='🎬',
        layout='wide'
    )

    # Custom CSS for a polished look
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Movie Recommendation System")
    st.caption("Find your next favorite movie using Content-Based Filtering.")

    try:

        # Sidebar for settings/info
        with st.sidebar:
            st.header("Settings")
            num_rec = st.slider("Number of recommendations", 5, 20, 10)
            st.info("The system uses cosine similarity on movie metadata.")
            
        if not os.path.exists(PREPROCESSED_DATASET_PATH):
            st.error(
                body = "Preprocessed dataframe doesn't exist.",
                width = 'stretch'
            )
        
        preprocessed_data = pd.read_csv(PREPROCESSED_DATASET_PATH, encoding = 'utf-8')
        
        selected_movie = st.selectbox(
            'Search or select a movie you liked:',
            options=get_movie_names(),
            index=None,
            placeholder="Type a movie name..."
        )

        btn = st.button("Recommend", type="primary", width = 'stretch')

        if btn and selected_movie:
            with st.status("Searching for matches...", expanded=True) as status:
                st.write("Analyzing metadata...")
                recommendations = recommend_movies(selected_movie, top_n=num_rec)

            # grid la supiyyu
            if not recommendations.empty:
                status.update(label="Recommendations Ready!", state="complete", expanded=False)
                st.subheader(f"Because you liked {selected_movie}:")

                cols = st.columns(5)

                for idx, (movie, poster_path) in enumerate(recommendations.itertuples(index=False)):
                    with cols[idx % 5]:
                        if poster_path:
                            st.image(f"https://image.tmdb.org/t/p/w500{poster_path}")
                        else:
                            st.write("No Image 🎬")
                        st.markdown(f"**{movie}**")
            else:
                st.warning("No similar movies found.")
        
        elif btn and not selected_movie:
            st.error("Please select a movie first!")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.button("Retry")

if __name__ == "__main__":
    main()