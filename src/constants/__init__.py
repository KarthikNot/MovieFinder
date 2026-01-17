import os

MOVIES_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'movies_metadata.csv')
KEYWORDS_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'keywords.csv')
CREDITS_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'credits.csv')
PREPROCESSED_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'preprocessed_dataset.csv')

COLUMNS_TO_DROP = [
    'adult', 'belongs_to_collection', 'budget', 'homepage', 'imdb_id', 'runtime', 
    'tagline', 'original_language', 'popularity', 'poster_path', 'revenue', 
    'vote_average', 'vote_count', 'video'
]

MOVIES_MAP_PATH = os.path.join(os.getcwd(), 'src', 'constants', 'movies_map.py')

ARTIFACTS_DIRECTORY = os.path.join(os.getcwd(), 'artifacts')
FITTED_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIRECTORY, 'fitted_vectorized.pkl')
VECTORIZED_MATRIX = os.path.join(ARTIFACTS_DIRECTORY, 'vectorized_matrix.pkl')