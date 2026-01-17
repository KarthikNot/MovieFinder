import os
import re
import ast
import nltk
import pandas as pd
from typing import List
from src.logger import logger
from nltk.stem import SnowballStemmer
from src.constants import MOVIES_DATASET_PATH, KEYWORDS_DATASET_PATH, CREDITS_DATASET_PATH, COLUMNS_TO_DROP, PREPROCESSED_DATASET_PATH

class DataPreprocessing:
    def __init__(self):
        
        self.movies_dataset_path = MOVIES_DATASET_PATH
        self.keywords_dataset_path = KEYWORDS_DATASET_PATH
        self.credits_dataset_path = CREDITS_DATASET_PATH
        
        
        nltk.download('stopwords')
        self.snow_stemmer = SnowballStemmer(language='english', ignore_stopwords=True)
    
    
    def clean_tags(self, tags) -> str | None:
        try:
            cleaned = []
            if not isinstance(tags, list):
                return None
            for tag in tags:
                tag = tag.lower()
                tag = re.sub(r'[^a-z0-9\s]', '', tag)
                tag = re.sub(r'\s+', ' ', tag).strip()
                tag = self.snow_stemmer.stem(tag)
                if tag:
                    cleaned.append(tag)
            return " ".join(cleaned)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return None

    
    @staticmethod
    def extract_data(data_dict : str) -> List[str] | None:
        try:
            if not isinstance(data_dict, str):
                return []
            data_list = ast.literal_eval(data_dict)
            data_list = ["".join(data['name'].split()) for data in data_list]
            return data_list
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return []
    
    
    @staticmethod
    def extract_director_and_producer(crew_data : str) -> List[str] | None:
        try:
            if not isinstance(crew_data, str):
                return []
            crew_list = ast.literal_eval(crew_data)
            directors = ["".join(member['name'].split()) for member in crew_list if member['job'] == 'Director']
            producers = ["".join(member['name'].split()) for member in crew_list if member['job'] == 'Producer']
            directors.extend(producers)
            return directors
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)
        return []
    
    
    def removing_null_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            if dataframe.empty:
                logger.warning("Dataframe is empty")
                return pd.DataFrame()
            logger.info(f"Null values before: {[(k,v) for k,v in dataframe.isnull().sum().items()]}")
            dataframe = dataframe.dropna()
            logger.info(f"Null values after: {[(k,v) for k,v in dataframe.isnull().sum().items()]}")
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return dataframe
    
    
    def removing_duplicate_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            if dataframe.empty:
                logger.warning("Dataframe is empty")
                return pd.DataFrame()
            logger.info(f"Duplicates before: {dataframe.duplicated().sum()}")
            dataframe = dataframe.drop_duplicates()
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
                
            if not os.path.exists(self.keywords_dataset_path):
                logger.info("Keywords dataset does not exist")
                return False
            keywords = pd.read_csv(self.keywords_dataset_path, encoding = 'utf-8')
            if keywords.empty:
                logger.error("Keywords dataset is empty")
                return False
            
            logger.info("Keywords dataframe loaded.")
            
            if not os.path.exists(self.credits_dataset_path):
                logger.info("Keywords dataset does not exist")
                return False
            credits = pd.read_csv(self.credits_dataset_path, encoding = 'utf-8')
            if credits.empty:
                logger.error("Credits dataset is empty")
                return False
            
            logger.info("Credits dataframe loaded.")
            
            # null and duplicates
            keywords = self.removing_null_values(keywords)
            keywords = self.removing_duplicate_values(keywords)
            
            credits = self.removing_null_values(credits)
            credits = self.removing_duplicate_values(credits)
            
            if not ('id' in keywords.columns and 'id' in credits.columns):
                logger.info("'id' column does not exists in keywords or credits columns, failed to merge.")
                return False
            keywords_and_credits = keywords.merge(credits, on = 'id', how = 'left')
            
            logger.info("Keywords & Credits dataframes merged.")
            
            df = movies.drop(columns=[c for c in COLUMNS_TO_DROP if c in movies.columns])
            
            df = df[pd.to_numeric(df['id'], errors='coerce').notna()]
            
            if not 'id' in df.columns:
                logger.info(f"'id' column doesn't exist dataframe.")
                return False
            df['id'] = df['id'].astype(int)
            
            if not 'id' in keywords_and_credits.columns:
                logger.info(f"'id' column doesn't exist dataframe.")
                return False
            keywords_and_credits['id'] = keywords_and_credits['id'].astype(int)

            if not ('id' in keywords_and_credits and 'id' in df.columns):
                logger.info(f"'id' column doesn't exist in dataframe.")
                return False
            final_df = df.merge(keywords_and_credits, on='id')
            
            logger.info("Data cleaning has started.")
            
            if 'genres' in final_df.columns:
                final_df['genres'] = final_df['genres'].apply(self.extract_data)
            if 'production_companies' in final_df.columns:
                final_df['production_companies'] = final_df['production_companies'].apply(self.extract_data)
            if 'production_countries' in final_df.columns:
                final_df['production_countries'] = final_df['production_countries'].apply(self.extract_data)
            if 'spoken_languages' in final_df.columns:
                final_df['spoken_languages'] = final_df['spoken_languages'].apply(self.extract_data)
            if 'keywords' in final_df.columns:
                final_df['keywords'] = final_df['keywords'].apply(self.extract_data)
            if 'cast' in final_df.columns:
                final_df['cast'] = final_df['cast'].apply(self.extract_data)
            if 'crew' in final_df.columns:
                final_df['crew'] = final_df['crew'].apply(self.extract_director_and_producer)
            if 'overview' in final_df.columns:
                final_df['overview'] = final_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
            
            final_df['all_tags'] = final_df['overview'] + final_df['genres'] + final_df['production_companies'] + final_df['production_countries'] + final_df['spoken_languages'] + final_df['keywords'] + final_df['cast'] + final_df['crew']
            final_df = final_df[['id', 'title', 'all_tags', 'release_date']]
            
            
            final_df['all_tags'] = final_df['all_tags'].apply(self.clean_tags)
            
            
            final_df = final_df.dropna()
            final_df = final_df.drop_duplicates()
            
            final_df = final_df.dropna(subset = 'all_tags')
            final_df = final_df.drop_duplicates(subset = 'title')
            
            if not 'release_date' in final_df.columns:
                logger.info(f"'release_date' column not in final_df")
            final_df['year'] = final_df['release_date'].str[:4].astype(int)
            final_df = final_df[final_df['year'] >= 1975]
            
            logger.info("Dataframe cleaning completed.")
            
            final_df.to_csv(PREPROCESSED_DATASET_PATH, index = False, encoding = 'utf-8')
            return True
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
        return False