import time
from typing import List

import polars as pl

from src.core.logger import logger
from src.core.models import get_recommender


def evaluate_recommenders(test_queries: List[str] = []):
    try:
        if not test_queries:
            test_queries = [
                "Inception", "The Godfather", "Toy Story", 
                "The Dark Knight", "Pulp Fiction", "Avatar", 
                "The Matrix", "Interstellar", "Gladiator", 
                "Jurassic Park"
            ]
            
        algorithms = ["Popularity", "TF-IDF (Sparse)", "LLM (Dense)", "Hybrid"]
        
        results = []
        
        for algo in algorithms:
            logger.info(f"Evaluating {algo}...")
            try:
                recommender = get_recommender(algo)
            except Exception as e:
                logger.error(f"Failed to load {algo}: {e}")
                results.append({
                    "Algorithm": algo,
                    "Latency (ms)": "N/A",
                    "Diversity Score": "N/A",
                    "Status": "Failed to load"
                })
                continue
                
            total_latency = 0
            total_diversity = 0
            successful_queries = 0
            
            for query in test_queries:
                start_time = time.time()
                recs = recommender.recommend(query, top_n=10)
                latency = time.time() - start_time
                
                if not recs:
                    continue
                    
                total_latency += latency
                successful_queries += 1
                
                years = set(r['year'] for r in recs if r['year'] != "N/A")
                diversity = len(years) / 10.0
                total_diversity += diversity
                
            if successful_queries > 0:
                avg_latency_ms = (total_latency / successful_queries) * 1000
                avg_diversity = total_diversity / successful_queries
                status = "Success"
            else:
                avg_latency_ms = 0
                avg_diversity = 0
                status = "No Results"
                
            results.append({
                "Algorithm": algo,
                "Latency (ms)": f"{avg_latency_ms:.1f}",
                "Diversity Score": f"{avg_diversity:.2f}",
                "Status": status
            })
            
        df_results = pl.DataFrame(results)
        
        with open("eval_results.md", "w", encoding="utf-8") as f:
            f.write("# Recommendation Benchmark Results\n\n")
            f.write("Evaluated on 10 diverse queries. Diversity score represents the spread of release years among top 10 recommendations.\n\n")
            f.write("| Algorithm | Latency (ms) | Diversity Score | Status |\n")
            f.write("| --- | --- | --- | --- |\n")
            for row in df_results.iter_rows(named=True):
                f.write(f"| {row['Algorithm']} | {row['Latency (ms)']} | {row['Diversity Score']} | {row['Status']} |\n")
            
        print(df_results)
        logger.info("Evaluation complete. Results saved to eval_results.md")
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    evaluate_recommenders()
