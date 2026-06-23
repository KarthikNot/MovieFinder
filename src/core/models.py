import os
import pickle
import numpy as np
import chromadb
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from src.core.config import *
from src.core.logger import logger


class RecommenderBase:
    def __init__(self):
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            self.collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise e

    def get_movie_row(self, title: str) -> Dict[str, Any] | None:
        try:
            result = self.collection.get(include=["metadatas"])
            target = title.strip().lower()
            for row in result["metadatas"]:
                if row and str(row.get("title", "")).strip().lower() == target:
                    return row
            for row in result["metadatas"]:
                if row and target in str(row.get("title", "")).strip().lower():
                    return row
            return None
        except Exception as e:
            logger.error(f"Error while fetching movie row: {e}")
            raise e

    def fetch_metadata(self, indices: List[int], scores: List[float], reason: str) -> List[Dict[str, Any]]:
        try:
            ids = [str(i) for i in indices]
            result = self.collection.get(ids=ids, include=["metadatas"])
            by_id = {int(item["row_idx"]): item for item in result["metadatas"] if item and "row_idx" in item}
            rows = []
            for idx, score in zip(indices, scores):
                row = by_id.get(idx)
                if not row:
                    continue
                rows.append({
                    "title": row["title"],
                    "poster_path": row.get("poster_path", ""),
                    "score": float(score),
                    "reason": reason,
                    "year": int(row.get("release_date", 0)) if row.get("release_date") else "N/A",
                })
            return rows
        except Exception as e:
            logger.error(f"Error while fetching metadata: {e}")
            raise e

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PopularityRecommender(RecommenderBase):
    def recommend(self, movie_title: str = "None", top_n: int = 10) -> List[Dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        rows = sorted(
            result["metadatas"],
            key=lambda row: row.get("popularity", 0) if row else 0,
            reverse=True,
        )[:top_n]
        return [
            {
                "title": row["title"],
                "poster_path": row.get("poster_path", ""),
                "score": float(row.get("popularity", 0)),
                "reason": "Highly popular worldwide on TMDB.",
                "year": int(row.get("release_date", 0)) if row.get("release_date") else "N/A",
            }
            for row in rows
        ]


class SparseRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        if not os.path.exists(SPARSE_VECTORS_PATH):
            raise FileNotFoundError("Sparse vectors not found.")
        with open(SPARSE_VECTORS_PATH, "rb") as f:
            self.vectors = pickle.load(f)

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, Any]]:
        row = self.get_movie_row(movie_title)
        if not row:
            return []
        row_idx = int(row["row_idx"])
        sim_scores = cosine_similarity(self.vectors[row_idx], self.vectors).flatten()
        sim_scores[row_idx] = -1
        top_indices = np.argsort(sim_scores)[-top_n:][::-1]
        return self.fetch_metadata(top_indices.tolist(), sim_scores[top_indices].tolist(), "High overlap in keywords, genres, and cast (TF-IDF).")


class DenseRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.embeddings = {}
        fields = ["overview_text", "keywords_text", "genre_text"]
        for field in fields:
            path = os.path.join(DENSE_EMBEDDINGS_DIRECTORY, f"{field}.npy")
            if os.path.exists(path):
                self.embeddings[field] = np.load(path, mmap_mode="r")
        self.total_rows = next(iter(self.embeddings.values())).shape[0] if self.embeddings else 0

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, Any]]:
        row = self.get_movie_row(movie_title)
        if not row or not self.embeddings or self.total_rows == 0:
            return []
        row_idx = int(row["row_idx"])
        total_sim = np.zeros(self.total_rows)
        for vectors in self.embeddings.values():
            query_vec = vectors[row_idx].reshape(1, -1)
            total_sim += np.dot(vectors, query_vec.T).flatten()
        total_sim /= len(self.embeddings)
        total_sim[row_idx] = -1
        top_indices = np.argsort(total_sim)[-top_n:][::-1]
        return self.fetch_metadata(top_indices.tolist(), total_sim[top_indices].tolist(), "Semantic plot and thematic similarity (LLM Embeddings).")


class HybridRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.sparse = SparseRecommender()
        self.dense = DenseRecommender()

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, Any]]:
        sparse_res = self.sparse.recommend(movie_title, top_n=50)
        dense_res = self.dense.recommend(movie_title, top_n=50)
        if not sparse_res and not dense_res:
            return []
        k = 60
        scores = {}
        metadata = {}
        for rank, res in enumerate(sparse_res):
            scores[res["title"]] = scores.get(res["title"], 0) + (1.0 / (k + rank))
            metadata[res["title"]] = res
        for rank, res in enumerate(dense_res):
            scores[res["title"]] = scores.get(res["title"], 0) + (1.0 / (k + rank))
            metadata[res["title"]] = res
        sorted_titles = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
        results = []
        for title in sorted_titles:
            item = metadata[title]
            item["score"] = scores[title]
            item["reason"] = "Strong overlap in both exact keywords and semantic themes."
            results.append(item)
        return results


def get_recommender(algorithm: str) -> RecommenderBase:
    if algorithm == "Popularity":
        return PopularityRecommender()
    if algorithm == "TF-IDF (Sparse)":
        return SparseRecommender()
    if algorithm == "LLM (Dense)":
        return DenseRecommender()
    return HybridRecommender()
