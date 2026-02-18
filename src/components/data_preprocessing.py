import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from src.logger import logger
from src.constants import MOVIES_DATASET_PATH, COLUMNS_TO_DROP, PREPROCESSED_DATASET_PATH

class DataPreprocessing:
    def __init__(self):
        
        self.movies_dataset_path = MOVIES_DATASET_PATH
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        self.stop_words = set(stopwords.words('english'))
        self._pattern = re.compile(r"[^a-z0-9\s]")


    def clean_tags(self, tags) -> str | None:
        try:
            if not isinstance(tags, list):
                return ""
            
            stop_words = self.stop_words  # local reference (faster)
            
            cleaned = [
                token
                for tag in tags
                for token in self._pattern.sub("", tag.lower()).split()
                if token and token not in stop_words
            ]

            return " ".join(cleaned)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return None


    @staticmethod
    def preprocess_tags(text, separator=" ") -> str:
        try:
            if not isinstance(text, str):
                text = str(text)

            text = text.lower()
            texts = text.split(separator)

            return ' '.join([t.replace(' ', '') for t in texts])
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return ""


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

            df = movies[movies['adult'] == False]

            df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

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

            df['genres'] = df['genres'].apply(self.preprocess_tags, separator = ',')
            df['keywords'] = df['keywords'].apply(self.preprocess_tags, separator = ',')
            df['overview'] = df['overview'].apply(self.preprocess_tags)

            cols = [
                'title', 'genres', 'keywords', 'overview',
            ]

            df['all_tags'] = df[cols].astype(str).agg(' '.join, axis=1)
            df['all_tags'] = df['all_tags'].apply(lambda x: x.lower())

            final_df = df[['id', 'title', 'poster_path', 'all_tags', 'release_date', 'status']].copy()
            final_df['all_tags'] = final_df['all_tags'].apply(lambda x: x.split())

            final_df['all_tags'] = final_df['all_tags'].apply(self.clean_tags)

            if not 'release_date' in df.columns:
                logger.info(f"'release_date' column not in df")
            else:
                final_df['year'] = final_df['release_date'].str[:4].astype(int)
                final_df = final_df[final_df['year'] >= 1990]

            final_df = final_df[(final_df['status'] == 'Released')]
            
            logger.info("Dataframe cleaning completed.")
            
            final_df.to_csv(PREPROCESSED_DATASET_PATH, index = False, encoding = 'utf-8')
            return True
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    obj = DataPreprocessing()
    obj.preprocess_dataframe()