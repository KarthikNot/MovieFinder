import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from typing import List
from src.constants import *
from src.pipelines.training_pipeline import ModelInference
from sklearn.metrics.pairwise import cosine_similarity
import time

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
        return list(df['title'].unique())
    except Exception as e:
        st.error(f"An error occured: {str(e)}")
        return []


def main():
    st.set_page_config(
        page_title='Movie Finder',
        page_icon='🎬',
        layout='wide'
    )

    st.markdown("""
        <style>
        /* App layout */
        .block-container { max-width: 1100px; padding: 1.5rem 2rem; margin: 0 auto; }

        /* Typography */
        h1 { color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; font-size: 2.2rem; margin-bottom: 0.2rem; }
        .stCaption { color: #9aa6b2; }


        /* Buttons */
        .stButton>button { width: 100%; border-radius: 8px; height: 3.2em; background-color:#1f6feb; color:#fff; border: none; box-shadow: 0 6px 18px rgba(31,111,235,0.12); transition: all 0.3s ease;}
        .stButton>button:hover { scale : 1.01; }

        /* Recommendation cards */
        .recommend-card { border-radius: 10px; margin: 0.5rem; padding: 0.6rem; box-shadow: 0 6px 12px rgba(2,6,23,0.3); text-align: center; width: 100%; }
        .recommend-card img { width: 100%; height: auto; border-radius: 8px; display:block; margin: 0 auto; }
        .recommend-title { margin-top: 0.5rem; font-weight: 700; font-size: 0.95rem; }

        /* Columns responsive - supports the 5-column layout on desktop and wraps on smaller screens */
        .stColumns > div { flex: 0 0 20% !important; max-width: 20% !important; display: flex; justify-content: center; }
        @media (max-width: 1000px) { .stColumns > div { flex: 0 0 33.3333% !important; max-width: 33.3333% !important; } }
        @media (max-width: 700px) { .stColumns > div { flex: 0 0 50% !important; max-width: 50% !important; } }
        @media (max-width: 420px) { .stColumns > div { flex: 0 0 100% !important; max-width: 100% !important; } }

        /* Minor tweaks */
        [data-testid="stStatusWidget"] { background: transparent; }
        .stTextInput>div>div>input { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Movie Recommendation System")
    st.caption("Find your next favorite movie using Content-Based Filtering.")

    if not os.path.exists(PREPROCESSED_DATASET_PATH):
        st.warning("Preprocessed dataset is missing. Initializing the pipeline...")

        temp = ModelInference()

        with st.spinner("Downloading dataset..."):
            try:
                temp.download_dataset()
                time.sleep(1)
            except Exception as e:
                st.error(f"Failed to download dataset: {e}")
                return

        with st.spinner("Running the preprocessing pipeline..."):
            try:
                temp.end_to_end_pipeline()
                time.sleep(1)
            except Exception as e:
                st.error(f"Failed to run the pipeline: {e}")
                return

        del temp
        st.success("Pipeline completed successfully! You can now search for movies.")

    try:

        # Sidebar for settings/info
        with st.sidebar:
            st.header("Settings")
            num_rec = st.slider("Number of recommendations", 5, 20, 5)
            st.info("The system uses cosine similarity on movie metadata.")
        if not os.path.exists(PREPROCESSED_DATASET_PATH):
            st.error(
                body = "Preprocessed dataframe doesn't exist.",
                width = 'stretch'
            )

        movie_names = get_movie_names()

        search_query = st.text_input("Search for a movie:")

        selected_movie = None

        if search_query:
            matches = [
                m for m in movie_names
                if search_query.lower() in m.lower()
            ][:20]  # limit results

            selected_movie = st.selectbox(
                "Select from matches:",
                matches
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
                            st.markdown(f"""
                                <div class='recommend-card'>
                                    <img src="https://image.tmdb.org/t/p/w500{poster_path}" alt="{movie}" />
                                    <div class='recommend-title'><b>{movie}</b></div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class='recommend-card'>
                                    <div style='padding:32px 8px; font-size:1.1rem;'>No Image 🎬</div>
                                    <div class='recommend-title'><b>{movie}</b></div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No similar movies found.")

        elif btn and not selected_movie:
            st.error("Please select a movie first!")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.button("Retry")

if __name__ == "__main__":
    main()