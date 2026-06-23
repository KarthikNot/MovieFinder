import asyncio
import os
import json
from typing import List
from urllib import request, error
import pickle
import numpy as np
import polars as pl
from src.core.config import *
from src.core.logger import logger
import chromadb
from langchain_ollama import OllamaEmbeddings
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


class DataVectorization:

    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        self.required_columns = [
            "id",
            "title",

            "overview_text",
            "keywords_text",
            "genre_text",
            "people_text",
            "metadata_text",
            "sparse_text",

            "vote_average",
            "vote_count",
            "popularity",
            "release_date"
        ]
        self.embedding_fields = [
            "overview_text",
            "keywords_text",
            "genre_text",
            "people_text",
            "metadata_text"
        ]
        self.ollama_embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=OLLAMA_BASE_SERVER_URL
        )



    def load_dataframe(self) -> pl.DataFrame:
        try:
            if not os.path.exists(PREPROCESSED_DATASET_PATH):
                logger.error("Preprocessed dataset doesn't exist.")
                return pl.DataFrame()

            dataframe = pl.read_csv(PREPROCESSED_DATASET_PATH, encoding="utf-8")

            if dataframe.is_empty():
                logger.error("Preprocessed dataframe is empty.")
                return pl.DataFrame()

            missing_columns = [
                col for col in self.required_columns
                if col not in dataframe.columns
            ]

            if missing_columns:
                raise RuntimeError(f"Missing required columns: {missing_columns}")

            dataframe = dataframe.fill_null("")

            logger.info(f"Preprocessed dataframe loaded successfully. Shape: {dataframe.shape}")
            return dataframe
        except Exception as e:
            logger.error(f"Error while loading dataframe: {str(e)}", exc_info=True)
            raise e


    def build_sparse_vectors(self, dataframe: pl.DataFrame):
        try:
            logger.info("Sparse vectorization started.")

            tfidf = TfidfVectorizer(
                lowercase=True,
                stop_words=None,
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                max_features=MAX_TFIDF_FEATURES,
                min_df=2,
                max_df=0.90,
                sublinear_tf=True,
                norm="l2"
            )

            sparse_vectors = tfidf.fit_transform(dataframe["sparse_text"].to_list())
            logger.info(f"Sparse vectorization completed. Shape: {sparse_vectors.shape}")
            return tfidf, sparse_vectors
        except Exception as e:
            logger.error(f"Error while building sparse vectors: {str(e)}", exc_info=True)
            raise e


    def check_ollama_status(self):
        try:
            base_url = OLLAMA_BASE_SERVER_URL.rstrip('/')
            url = f"{base_url}/api/tags"
            req = request.Request(url)
            with request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            if not any(self.embedding_model in name for name in model_names):
                raise ValueError(f"Ollama is running, but model '{self.embedding_model}' is not pulled. Run: ollama pull {self.embedding_model}")

            logger.info(f"Ollama connection successful. Model '{self.embedding_model}' is available.")

        except error.URLError as e:
            logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_SERVER_URL}. Is it running? Error: {e}")
            raise RuntimeError(f"Ollama connection failed. Ensure Ollama is running at {OLLAMA_BASE_SERVER_URL}")


    def build_and_save_dense_embeddings(self, dataframe: pl.DataFrame) -> bool:
        try:
            self.check_ollama_status()
            total_rows = len(dataframe)
            logger.info(f"Dense embedding generation started for {total_rows} movies.")
            os.makedirs(DENSE_EMBEDDINGS_DIRECTORY, exist_ok=True)

            async def process_batch_embeddings(texts_list: List[str], batch_size: int = 100):
                """Helper to process texts in batches to avoid timeouts and memory spikes."""
                all_vectors = []
                for i in range(0, len(texts_list), batch_size):
                    batch = texts_list[i : i + batch_size]
                    # Use aembed_documents for multiple strings
                    batch_emb = await self.ollama_embeddings.aembed_documents(batch)
                    all_vectors.extend(batch_emb)
                    
                    if (i // batch_size) % 10 == 0:
                        logger.info(f"  Progress: {min(i + batch_size, len(texts_list))}/{len(texts_list)}")
                return np.array(all_vectors)

            for field in self.embedding_fields:
                save_path = os.path.join(DENSE_EMBEDDINGS_DIRECTORY, f"{field}.npy")
                
                if os.path.exists(save_path):
                    try:
                        existing = np.load(save_path, mmap_mode='r')
                        if existing.shape[0] == total_rows:
                            logger.info(f"Embeddings for {field} already exist. Skipping.")
                            continue
                    except Exception as e:
                        logger.warning(f"Invalid embeddings for {field}, regenerating: {e}")

                logger.info(f"Generating embeddings for: {field}")
                texts = dataframe[field].cast(pl.Utf8).to_list()
                
                # Run the async batch processing
                vectors = asyncio.run(process_batch_embeddings(texts))

                logger.info(f"Normalizing embeddings for: {field}")
                vectors = normalize(vectors, axis=1, copy=False)

                np.save(save_path, vectors)
                logger.info(f"Saved normalized embeddings: {field}")
                del vectors

            return True
        except Exception as e:
            logger.error(f"Error while generating dense embeddings: {str(e)}",exc_info=True)
            raise e


    def save_sparse_artifacts(self, tfidf, sparse_vectors) -> bool:
        try:
            os.makedirs(ARTIFACTS_DIRECTORY, exist_ok=True)
            with open(FITTED_VECTORIZER_PATH, "wb+") as file:
                pickle.dump(tfidf, file)
            with open(SPARSE_VECTORS_PATH, "wb+") as file:
                pickle.dump(sparse_vectors, file)
            logger.info("Sparse artifacts saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Error while saving sparse artifacts: {str(e)}", exc_info=True)
            raise e


    def push_metadata_to_chromadb(self, dataframe: pl.DataFrame) -> bool:
        try:
            metadata_columns = [
                "id", "title", "overview_text", "keywords_text",
                "genre_text", "people_text", "metadata_text",
                "poster_path",
                "vote_average", "vote_count", "popularity", "release_date"
            ]
            metadata = dataframe.select(metadata_columns).with_row_index("row_idx")

            logger.info("Clearing existing metadata in ChromaDB...")
            self.chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            batch_size = 1000
            rows = metadata.to_dicts()
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                ids = [str(row["row_idx"]) for row in batch]
                documents = [row["title"] for row in batch]
                metadatas = [
                    {
                        "id": int(row["id"]),
                        "title": row["title"],
                        "poster_path": row.get("poster_path", ""),
                        "release_date": int(row["release_date"]) if row.get("release_date") not in (None, "") else None,
                        "vote_average": float(row.get("vote_average", 0) or 0),
                        "vote_count": int(row.get("vote_count", 0) or 0),
                        "popularity": float(row.get("popularity", 0) or 0),
                        "row_idx": int(row["row_idx"]),
                    }
                    for row in batch
                ]
                self.chroma_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            logger.info(f"Successfully pushed {len(rows)} movies to ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error while pushing metadata to ChromaDB: {str(e)}", exc_info=True)
            raise e


    def run_vectorization_pipeline(self) -> bool:
        try:
            logger.info("Vectorization pipeline started.")

            dataframe = self.load_dataframe()

            if dataframe.is_empty():
                return False

            tfidf, sparse_vectors = self.build_sparse_vectors(dataframe)

            if tfidf is None or sparse_vectors is None:
                return False

            sparse_saved = self.save_sparse_artifacts(tfidf, sparse_vectors)

            dense_saved = self.build_and_save_dense_embeddings(dataframe)

            if not dense_saved:
                return False

            metadata_saved = self.push_metadata_to_chromadb(dataframe)

            if not all([sparse_saved, dense_saved, metadata_saved]):
                logger.error("Some artifacts failed to save.")
                return False

            logger.info("Complete vectorization pipeline finished successfully.")

            return True

        except Exception as e:
            logger.error(f"Error in vectorization pipeline: {str(e)}", exc_info=True)
            raise e
