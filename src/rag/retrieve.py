from pathlib import Path
import json, faiss
from sentence_transformers import SentenceTransformer

class FinancialRetriever:
    def __init__(self, directory="data/knowledge"):
        d = Path(directory)
        self.index = faiss.read_index(str(d / "faiss.index"))
        self.metadata = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def search(self, query: str, k: int = 5):
        vector = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.metadata[idx]); item["score"] = float(score); results.append(item)
        return results
