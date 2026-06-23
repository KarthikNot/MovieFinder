import os
import re
import spacy
from spacy.cli.download import download
import polars as pl
from typing import List
from src.core.logger import logger
from src.core.config import *

class DataPreprocessing:
    def __init__(self):
        self.whitespace_pattern = re.compile(r"\s+")
        try:
            self.lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Downloading en_core_web_sm model...")
            download("en_core_web_sm")

            self.lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])


    def clean_text(self, text : str) -> str:
        try:
            if text is None:
                return ""
            text = str(text).lower()
            text = self.whitespace_pattern.sub(" ", text).strip()
            return text
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            raise e


    def clean_text_minimal(self, text: str) -> str:
        """Cleans whitespace and lowercases but preserves natural language for LLM embeddings."""
        try:
            if text is None:
                return ""
            text = str(text).lower()
            text = self.whitespace_pattern.sub(" ", text).strip()
            return text
        except Exception as e:
            logger.error(f"An error occurred in minimal cleaning: {str(e)}", exc_info=True)
            raise e


    def preprocess_text(self, texts : List[str]) -> List[str]:
        try:
            texts = [self.clean_text(x) for x in texts]

            docs = self.lemmatizer.pipe(texts, batch_size=1000, n_process=MAX_PROCESSES if MAX_PROCESSES is not None else 1)

            processed = []
            for doc in docs:
                processed.append(
                    " ".join(
                        token.text if token.pos_ == "PROPN" else token.lemma_
                        for token in doc
                        if not (token.is_punct or token.is_space)
                    )
                )
            return processed
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            raise e


    def removing_null_values(self, dataframe: pl.DataFrame, subset = None) -> pl.DataFrame:
        try:
            if dataframe.is_empty():
                logger.warning("Dataframe is empty")
                return pl.DataFrame()
            before = {col: dataframe.select(pl.col(col).null_count()).item() for col in dataframe.columns}
            logger.info(f"Null values before: {[(k,v) for k,v in before.items() if v > 0]}")
            dataframe = dataframe.drop_nulls(subset=subset)
            after = {col: dataframe.select(pl.col(col).null_count()).item() for col in dataframe.columns}
            logger.info(f"Null values after: {[(k,v) for k,v in after.items() if v > 0]}")
            return dataframe
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
            raise e


    def removing_duplicate_values(self, dataframe: pl.DataFrame, subset = None) -> pl.DataFrame:
        try:
            if dataframe.is_empty():
                logger.warning("Dataframe is empty")
                return pl.DataFrame()
            logger.info(f"Duplicates before: {dataframe.is_duplicated().sum()}")
            dataframe = dataframe.unique(subset=subset, keep="first")
            logger.info(f"Duplicates after: {dataframe.is_duplicated().sum()}")
            return dataframe
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
            raise e


    def preprocess_dataframe(self) -> bool:
        try:
            if os.path.exists(PREPROCESSED_DATASET_PATH):
                logger.info("Preprocessed dataframe exists.")
                return True

            if not os.path.exists(MOVIES_DATASET_PATH):
                logger.info("Movies dataset does not exists.")
                return False
            movies = pl.read_csv(MOVIES_DATASET_PATH, encoding='utf-8')
            if movies.is_empty(): 
                logger.error("Movies dataset is empty")
                return False

            logger.info("Movies dataframe loaded.")

            df = movies.filter(pl.col("adult") == False)
            
            df = self.removing_null_values(df, subset = "title")
            df = self.removing_null_values(df, subset = "overview")
            df = self.removing_null_values(df, subset = "poster_path")
            df = self.removing_null_values(df, subset = "release_date")

            df = self.removing_duplicate_values(df, subset = "id")
            df = self.removing_duplicate_values(df, subset=["title", "release_date"])

            df = df.filter(pl.col("id").cast(pl.Int64, strict=False).is_not_null())

            if not 'id' in df.columns:
                logger.info(f"'id' column doesn't exist dataframe.")
                return False

            df = df.with_columns([
                pl.col("id").cast(pl.Int64),
                pl.col("release_date").str.strptime(pl.Date, strict=False).dt.year().alias("release_date"),
            ])
            df = df.filter(pl.col("release_date") >= 1975)

            logger.info("Data cleaning has started...")

            # For Dense Embeddings: Use minimal cleaning to preserve semantic relationships for the LLM
            df = df.with_columns([
                (pl.col("overview").fill_null("") + pl.lit(" ") + pl.col("tagline").fill_null("")).map_elements(self.clean_text_minimal, return_dtype=pl.Utf8).alias("overview_text"),
                pl.col("keywords").map_elements(self.clean_text_minimal, return_dtype=pl.Utf8).alias("keywords_text"),
                pl.col("genres").map_elements(self.clean_text_minimal, return_dtype=pl.Utf8).alias("genre_text"),
                (
                    pl.col("directors").fill_null("") + pl.lit(" ") +
                    pl.col("writers").fill_null("") + pl.lit(" ") +
                    pl.col("cast").fill_null("")
                ).map_elements(self.clean_text_minimal, return_dtype=pl.Utf8).alias("people_text"),
                (
                    pl.col("production_companies").fill_null("") + pl.lit(" ") +
                    pl.col("production_countries").fill_null("") + pl.lit(" ") +
                    pl.col("spoken_languages").fill_null("") + pl.lit(" ") +
                    pl.col("original_language").fill_null("")
                ).map_elements(self.clean_text_minimal, return_dtype=pl.Utf8).alias("metadata_text"),
            ])
            
            # For Sparse Retrieval (TF-IDF): Use full lemmatization
            logger.info("Generating lemmatized sparse content...")
            lemma_overview = self.preprocess_text((df["overview"].fill_null("") + " " + df["tagline"].fill_null("")).to_list())
            lemma_keywords = self.preprocess_text(df["keywords"].to_list())
            lemma_genres = self.preprocess_text(df["genres"].to_list())
            lemma_people = self.preprocess_text(
                (df["directors"].fill_null("") + " " +
                df["writers"].fill_null("") + " " +
                df["cast"].fill_null("")).to_list()
            )
            sparse_text = [
                " ".join(
                    [
                        str(title),
                        str(title),
                        str(title),
                        lemma_keywords[i],
                        lemma_keywords[i],
                        lemma_genres[i],
                        lemma_genres[i],
                        lemma_people[i],
                        lemma_overview[i],
                    ]
                ).strip()
                for i, title in enumerate(df["title"].to_list())
            ]
            df = df.with_columns(pl.Series("sparse_text", sparse_text))

            final_df = df.select([
                'id',
                'title',
                'overview',
                'overview_text',
                'keywords_text',
                'genre_text',
                'people_text',
                'metadata_text',
                'sparse_text',
                'vote_average',
                'vote_count',
                'popularity',
                'runtime',
                'poster_path',
                'release_date'
            ])
            
            logger.info("Dataframe cleaning completed.")
            
            final_df.write_csv(PREPROCESSED_DATASET_PATH)
            return True
        except Exception as e:
            logger.warning(f"An error occurred: {str(e)}", exc_info=True)
            raise e
