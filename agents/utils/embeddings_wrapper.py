# utils/embeddings_wrapper.py
from typing import List
from sentence_transformers import SentenceTransformer

class STEmbeddings:
    """
    Minimal wrapper around SentenceTransformer to produce embeddings
    compatible with your vector search pipeline.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str):
        vec = self.model.encode(query)
        return vec.tolist()

    def embed_documents(self, docs: List[str]):
        embs = self.model.encode(docs)
        return [v.tolist() for v in embs]
