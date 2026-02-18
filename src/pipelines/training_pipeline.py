import os
import zipfile
import subprocess
import kagglehub
from src.logger import logger
from src.constants import PREPROCESSED_DATASET_PATH, FITTED_VECTORIZER_PATH, MOVIES_DATASET_PATH
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_vectorization import DataVectorization

class ModelInference():

    def download_dataset(self):
        try:
            logger.info("Dataset download started.")

            if os.path.exists(MOVIES_DATASET_PATH):
                logger.info("Dataset already exists. Skipping download.")
                return None

            dataset_dir = os.path.dirname(MOVIES_DATASET_PATH)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_path = kagglehub.dataset_download(
                "asaniczka/tmdb-movies-dataset-2023-930k-movies",
                output_dir=dataset_dir,
            )

            if os.path.exists(MOVIES_DATASET_PATH):
                logger.info("Dataset ready.")
            else:
                logger.error("Dataset file not found after download.")

        except Exception as e:
            logger.error(
                f"An error occurred during dataset download: {str(e)}",
                exc_info=True
            )

    def end_to_end_pipeline(self):
        try:

            data_preprocessing = DataPreprocessing()
            data_vectorization = DataVectorization()

            preprocessing_result = data_preprocessing.preprocess_dataframe()
            if not preprocessing_result:
                logger.info("Pipeline failed: dataframe preprocessing failure.")
                return

            logger.info("Dataframe preprocessed successfully")

            if not os.path.exists(PREPROCESSED_DATASET_PATH):
                logger.info("Dataframe does not exist")
                return

            logger.info("Dataframe loaded.")

            data_vectorization.get_vectors()

            if not os.path.exists(FITTED_VECTORIZER_PATH):
                logger.info("Vectorized does not exist")
                return

        except Exception as e:
            logger.error(f"An error occured: {str(e)}", exc_info = True)


if __name__ == '__main__':
    obj = ModelInference()
    obj.download_dataset()
    obj.end_to_end_pipeline()