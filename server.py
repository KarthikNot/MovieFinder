import os
import polars as pl
import streamlit as st
from src.core.models import get_recommender
from src.core.config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME
import chromadb

@st.cache_data
def get_movie_names():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        result = collection.get(include=["metadatas"])
        titles = sorted({row["title"] for row in result["metadatas"] if row and row.get("title")})
        return titles
    except Exception as e:
        st.error(f"Failed to fetch movies from ChromaDB: {str(e)}")
        return []

def main():
    st.set_page_config(page_title='MovieFinder', page_icon='🎬', layout='wide')

    st.markdown("""
        <style>
        .block-container { max-width: 1100px; padding: 1.5rem 2rem; margin: 0 auto; }
        h1 { color: #ffffff; font-size: 2.2rem; margin-bottom: 0.2rem; }
        .stCaption { color: #9aa6b2; }
        .stButton>button { width: 100%; border-radius: 8px; height: 3.2em; background-color:#1f6feb; color:#fff; border: none; box-shadow: 0 6px 18px rgba(31,111,235,0.12); transition: all 0.3s ease;}
        .stButton>button:hover { scale : 1.01; }
        .recommend-card { border-radius: 10px; margin: 0.5rem; padding: 0.6rem; box-shadow: 0 6px 12px rgba(2,6,23,0.3); text-align: center; width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: space-between; background-color: #1a1c23; border: 1px solid #30363d;}
        .recommend-card img { width: 100%; height: auto; border-radius: 8px; display:block; margin: 0 auto; }
        .recommend-title { margin-top: 0.5rem; font-weight: 700; font-size: 0.95rem; color: #e6edf3;}
        .recommend-reason { font-size: 0.75rem; color: #8b949e; margin-top: 0.4rem; font-style: italic; text-align: left; padding: 0 5px;}
        .recommend-score { font-size: 0.8rem; font-weight: bold; color: #58a6ff; margin-top: 0.3rem;}
        .stColumns > div { flex: 0 0 20% !important; max-width: 20% !important; display: flex; justify-content: center; }
        @media (max-width: 1000px) { .stColumns > div { flex: 0 0 33.3333% !important; max-width: 33.3333% !important; } }
        @media (max-width: 700px) { .stColumns > div { flex: 0 0 50% !important; max-width: 50% !important; } }
        @media (max-width: 420px) { .stColumns > div { flex: 0 0 100% !important; max-width: 100% !important; } }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 MovieFinder")
    st.caption("A Production-Grade Hybrid Recommendation System")

    with st.sidebar:
        st.header("Algorithm Settings")
        algorithm = st.radio(
            "Recommendation Algorithm:",
            ["Hybrid", "TF-IDF (Sparse)", "LLM (Dense)", "Popularity"],
            help="Hybrid uses Reciprocal Rank Fusion. TF-IDF uses Exact matches. LLM uses semantic matching."
        )
        num_rec = st.slider("Number of recommendations", 5, 20, 5)
        
        st.markdown("---")
        st.markdown("### Benchmarks")
        st.markdown("""
        **Model Performance**
        - **Hybrid**: Best Quality
        - **TF-IDF**: High Relevance, Low Diversity
        - **LLM**: High Semantic Diversity
        - **Popularity**: Baseline metric
        """)

    movie_names = get_movie_names()
    
    if not movie_names:
        st.error("No movies found. Please run the training pipeline.")
        st.stop()

    search_query = st.text_input("Search for a movie:")
    selected_movie = ""

    if search_query:
        matches = [m for m in movie_names if isinstance(m, str) and search_query.lower() in m.lower()][:20]
        selected_movie = st.selectbox("Select from matches:", matches)

    btn = st.button("Recommend", type="primary")

    if btn:
        if algorithm != "Popularity" and not selected_movie:
            st.error("Please select a movie first!")
        else:
            with st.spinner(f"Running {algorithm} Recommender..."):
                try:
                    recommender = get_recommender(algorithm)
                    recommendations = recommender.recommend(selected_movie, top_n=num_rec)
                    
                    if not recommendations:
                        st.warning("No recommendations found.")
                    else:
                        st.subheader(f"Recommendations using {algorithm}:")
                        cols = st.columns(5)
                        
                        for idx, rec in enumerate(recommendations):
                            with cols[idx % 5]:
                                poster_path = rec.get('poster_path')
                                poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path not in (None, "") else ""
                                img_html = f"<img src='{poster}' alt='{rec['title']}' />" if poster else "<div style='padding:32px 8px; font-size:1.1rem; text-align:center;'>No Image 🎬</div>"
                                
                                st.markdown(f"""
                                    <div class='recommend-card'>
                                        <div>
                                            {img_html}
                                            <div class='recommend-title'>{rec['title']} ({rec['year']})</div>
                                            <div class='recommend-score'>Score: {rec['score']:.4f}</div>
                                        </div>
                                        <div class='recommend-reason'>{rec['reason']}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred during recommendation: {str(e)}")

if __name__ == "__main__":
    main()
