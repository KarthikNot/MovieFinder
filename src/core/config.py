import os
from dotenv import load_dotenv

load_dotenv()

KAGGLE_DATASET_LINK: str = "ggtejas/tmdb-imdb-merged-movies-dataset"

BASE_PATH: str = os.getcwd()
ARTIFACTS_DIRECTORY: str = os.path.join(BASE_PATH, 'artifacts')
FITTED_VECTORIZER_PATH: str = os.path.join(ARTIFACTS_DIRECTORY, 'fitted_vectorizer.pkl')
SPARSE_VECTORS_PATH: str = os.path.join(ARTIFACTS_DIRECTORY, 'sparse_vectors.pkl')
DENSE_EMBEDDINGS_DIRECTORY: str = os.path.join(ARTIFACTS_DIRECTORY, 'dense_embeddings')
VECTORIZED_METADATA_PATH: str = os.path.join(ARTIFACTS_DIRECTORY, 'vectorized_metadata.csv')


MOVIES_DATASET_PATH: str = os.path.join(os.getcwd(), 'data', 'TMDB  IMDB Movies Dataset.csv')
PREPROCESSED_DATASET_PATH: str = os.path.join(os.getcwd(), 'data', 'preprocessed_dataset.csv')

SPACY_LEMMATIZER_MODEL: str = "en_core_web_sm"
MAX_PROCESSES: int | None = os.cpu_count() if os.cpu_count() is not None else 1

EMBEDDING_MODEL: str = "qwen3-embedding:4b"
OLLAMA_BASE_SERVER_URL: str = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
MAX_TFIDF_FEATURES: int = 50000

BATCH_SIZE: int = 2000

CHROMA_PERSIST_DIRECTORY: str = os.path.join(ARTIFACTS_DIRECTORY, "chroma")
CHROMA_COLLECTION_NAME: str = "movies"