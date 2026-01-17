import os
import re
import pandas as pd
from nltk.stem import SnowballStemmer
from src.logger import logger
from src.constants import MOVIES_DATASET_PATH, COLUMNS_TO_DROP, PREPROCESSED_DATASET_PATH

class DataPreprocessing:
    def __init__(self):
        
        self.movies_dataset_path = MOVIES_DATASET_PATH
        self.snowball_stemmer = SnowballStemmer(language = 'english', ignore_stopwords = True)

    
    def clean_tags(self, tags) -> str | None:
        try:
            cleaned = []
            if not isinstance(tags, list):
                return None
            for tag in tags:
                tag = tag.lower()
                tag = re.sub(r'[^a-z0-9\s]', '', tag)
                tag = re.sub(r'\s+', ' ', tag).strip()
                tag = self.snowball_stemmer.stem(tag)
                if tag:
                    cleaned.append(tag)
            return " ".join(cleaned)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return None

    
    def removing_null_values(self, dataframe: pd.DataFrame, subset = None) -> pd.DataFrame:
        try:
            if dataframe.empty:
                logger.warning("Dataframe is empty")
                return pd.DataFrame()
            logger.info(f"Null values before: {[(k,v) for k,v in dataframe.isnull().sum().items() if v > 0]}")
            dataframe = dataframe.dropna(subset = subset)
            logger.info(f"Null values after: {[(k,v) for k,v in dataframe.isnull().sum().items() if v > 0]}")
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return dataframe
    
    
    def removing_duplicate_values(self, dataframe: pd.DataFrame, subset = None) -> pd.DataFrame:
        try:
            if dataframe.empty:
                logger.warning("Dataframe is empty")
                return pd.DataFrame()
            logger.info(f"Duplicates before: {dataframe.duplicated().sum()}")
            dataframe = dataframe.drop_duplicates(subset = subset)
            logger.info(f"Duplicates after: {dataframe.duplicated().sum()}")
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return dataframe
    
    
    def preprocess_dataframe(self) -> bool:
        try:
            if os.path.exists(PREPROCESSED_DATASET_PATH):
                logger.info("Preprocessed dataframe exists.")
                return True
            
            if not os.path.exists(self.movies_dataset_path):
                logger.info("Movies dataset does not exists.")
                return False
            movies = pd.read_csv(self.movies_dataset_path, encoding = 'utf-8')
            if movies.empty: 
                logger.error("Movies dataset is empty")
                return False
            
            logger.info("Movies dataframe loaded.")

            df = movies.drop(columns=[c for c in COLUMNS_TO_DROP if c in movies.columns])
            
            df = self.removing_null_values(df, subset = 'title')
            df = self.removing_duplicate_values(df, subset = 'title')
            df = self.removing_null_values(df, subset = 'overview')
            df = self.removing_duplicate_values(df, subset = 'overview')
            df = self.removing_null_values(df, subset = 'poster_path')
            df = self.removing_duplicate_values(df, subset = 'poster_path')
            df = self.removing_null_values(df, subset = 'release_date')
            
            df = df[pd.to_numeric(df['id'], errors='coerce').notna()]
            
            if not 'id' in df.columns:
                logger.info(f"'id' column doesn't exist dataframe.")
                return False
            df['id'] = df['id'].astype(int)
            
            logger.info("Data cleaning has started.")
            
            if 'genres' in df.columns:
                df['genres'] = df['genres'].apply(lambda x : str(x).split(','))
            if 'production_companies' in df.columns:
                df['production_companies'] = df['production_companies'].apply(lambda x : str(x).split(','))
            if 'production_countries' in df.columns:
                df['production_countries'] = df['production_countries'].apply(lambda x : str(x).split(','))
            if 'spoken_languages' in df.columns:
                df['spoken_languages'] = df['spoken_languages'].apply(lambda x : str(x).split(','))
            if 'keywords' in df.columns:
                df['keywords'] = df['keywords'].apply(lambda x : str(x).split(','))
            if 'overview' in df.columns:
                df['overview'] = df['overview'].apply(lambda x: str(x).split())
            
            df['all_tags'] = df['all_tags'] = df['overview'] + df['genres'] + df['production_companies'] + df['production_countries'] + df['spoken_languages'] + df['keywords']
            df = df[['id', 'title', 'poster_path', 'all_tags', 'release_date', 'status']]
            
            df['all_tags'] = df['all_tags'].apply(self.clean_tags)
            
            if not 'release_date' in df.columns:
                logger.info(f"'release_date' column not in df")

            df['year'] = df['release_date'].str[:4].astype(int)
            df = df[df['year'] >= 1975]

            df = df[(df['status'] != 'Canceled') | (df['status'] != 'Planned')]
            
            logger.info("Dataframe cleaning completed.")
            
            df.to_csv(PREPROCESSED_DATASET_PATH, index = False, encoding = 'utf-8')
            return True
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return False
    
if __name__ == '__main__':
    obj = DataPreprocessing()
    obj.preprocess_dataframe()