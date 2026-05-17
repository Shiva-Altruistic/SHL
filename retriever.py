import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self, catalog_path="catalog.json", index_path="faiss_index.bin"):
        self.catalog_path = catalog_path
        self.index_path = index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.metadata = []
        self.bm25 = None
        self.index = None
        self.load_or_build()

    def load_or_build(self):
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)

        self.metadata = []
        for item in raw_metadata:
            skills = item.get("skills_measured", item.get("skills tested", []))
            if isinstance(skills, str):
                skills = [skills]
                
            self.metadata.append({
                "name": item.get("name", item.get("assessment name", "")),
                "url": item.get("url", item.get("URL", "")),
                "description": item.get("description", ""),
                "test_type": item.get("test_type", item.get("test type", "")),
                "skills_measured": skills
            })

        # Prepare BM25 corpus (name + skills_measured)
        corpus = []
        for item in self.metadata:
            text = f"{item.get('name', '')} " + " ".join(item.get('skills_measured', []))
            tokens = text.lower().split()
            corpus.append(tokens if tokens else ["empty"])
        self.bm25 = BM25Okapi(corpus)

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Build Dense Index (descriptions)
            descriptions = [item.get('description', '') for item in self.metadata]
            embeddings = self.model.encode(descriptions, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_path)

    def search(self, query: str, top_k: int = 10, test_type_filters=None, remove_filters=None):
        query_lower = query.lower()
        
        # Dense
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, len(self.metadata))
        
        dense_scores = {idx: float(dist) for dist, idx in zip(distances[0], indices[0])}
        
        # Sparse
        tokenized_query = query_lower.split()
        bm25_scores_arr = self.bm25.get_scores(tokenized_query)
        # Normalize BM25
        max_bm25 = max(bm25_scores_arr) if len(bm25_scores_arr) > 0 and max(bm25_scores_arr) > 0 else 1.0
        
        combined_scores = []
        for idx, item in enumerate(self.metadata):
            ds = dense_scores.get(idx, 0.0)
            ss = bm25_scores_arr[idx] / max_bm25
            score = 0.6 * ds + 0.4 * ss
            
            # Apply filters
            item_type = item.get('test_type', '').lower()
            item_name = item.get('name', '').lower()
            
            if test_type_filters:
                if not any(f.lower() in item_type for f in test_type_filters):
                    continue
            if remove_filters:
                if any(r.lower() in item_name for r in remove_filters):
                    continue
                    
            combined_scores.append((score, item))
            
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate by URL
        seen_urls = set()
        final_results = []
        for score, item in combined_scores:
            url = item.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                final_results.append(item)
                if len(final_results) >= top_k:
                    break
                    
        return final_results
