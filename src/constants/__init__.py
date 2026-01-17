import os

MOVIES_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'TMDB_movie_dataset_v11.csv')
PREPROCESSED_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'preprocessed_dataset.csv')

COLUMNS_TO_DROP = [
    'adult', 'budget', 'backdrop_path', 'homepage', 'imdb_id', 
    'runtime', 'tagline', 'popularity', 'revenue', 'vote_average', 
    'vote_count', 'original_language'
]

MOVIES_MAP_PATH = os.path.join(os.getcwd(), 'src', 'constants', 'movies_map.py')

ARTIFACTS_DIRECTORY = os.path.join(os.getcwd(), 'artifacts')
FITTED_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIRECTORY, 'fitted_vectorized.pkl')
VECTORIZED_MATRIX = os.path.join(ARTIFACTS_DIRECTORY, 'vectorized_matrix.pkl')