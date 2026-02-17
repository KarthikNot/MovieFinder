import os
from src.logger import logger
from src.constants import PREPROCESSED_DATASET_PATH, FITTED_VECTORIZER_PATH
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_vectorization import DataVectorization

class ModelInference():
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
    obj.end_to_end_pipeline()