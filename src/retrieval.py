import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

class Retriever:
    def __init__(self, data_path="data/processed/apple_support_pairs.csv"):
        self.data_path = data_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.df = None
        self.build_index()
        
    def build_index(self):
        print(f"Loading data from {self.data_path} for retrieval index...")
        self.df = pd.read_csv(self.data_path)
        
        # In a real scenario we might remove the test set from the index to avoid leakage.
        # For simplicity, we index the whole processed set.
        texts = self.df['customer_text'].tolist()
        
        print(f"Encoding {len(texts)} texts for FAISS index...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print("FAISS index built successfully.")
        
    def retrieve(self, query_text, top_k=3):
        query_embedding = self.model.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'customer_text': self.df.iloc[idx]['customer_text'],
                'brand_text': self.df.iloc[idx]['brand_text'],
                'distance': distances[0][i]
            })
        return results

if __name__ == "__main__":
    retriever = Retriever()
    sample_query = "the battery life of my 6S is horrendous after iOS REALLY need to address this ASAP"
    results = retriever.retrieve(sample_query)
    print(f"\nQuery: {sample_query}")
    for i, res in enumerate(results):
        print(f"\nMatch {i+1} (Dist: {res['distance']:.2f}):")
        print(f"Customer: {res['customer_text']}")
        print(f"Brand: {res['brand_text']}")
