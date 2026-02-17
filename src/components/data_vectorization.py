import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logger
from src.constants import *


class DataVectorization:

    def get_vectors(self):
        try:
            if not os.path.exists(PREPROCESSED_DATASET_PATH):
                logger.info("Preprocessed dataset doesn't exist.")
                return None

            dataframe = pd.read_csv(PREPROCESSED_DATASET_PATH, encoding = 'utf-8')

            if 'all_tags' not in dataframe.columns:
                logger.info("'all_tags' column doesn't exist.")
                return None

            dataframe = dataframe.dropna(subset=['all_tags'])

            logger.info("Vectorization of tags started.")

            tfidf = TfidfVectorizer(stop_words='english')

            vectors = tfidf.fit_transform(dataframe['all_tags'])

            logger.info("Vectorization of tags completed.")

            os.makedirs(ARTIFACTS_DIRECTORY, exist_ok=True)

            try:
                with open(FITTED_VECTORIZER_PATH, 'wb+') as file:
                    pickle.dump(tfidf, file)
            except Exception as e:
                logger.error("Fitter Vectorizer File not saved.")

            try:
                with open(VECTORIZED_MATRIX, 'wb+') as file:
                    pickle.dump(vectors, file)
            except Exception as e:
                logger.error("Vectorized Matrix not saved.")

            logger.info(f"All feature names: {tfidf.get_feature_names_out()}")

            return vectors
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info = True)