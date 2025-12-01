# tools/vector_search.py
import os
from typing import List, Dict, Any
from sqlalchemy import create_engine
from langchain_postgres import PGVectorStore, PGEngine
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PG_CONN_STR = os.getenv("DATABASE_URL")
TABLE_NAME = "transaction_embeddings"

if not PG_CONN_STR:
    raise ValueError("DATABASE_URL not found in environment variables.")

embedding_service = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

engine_url = PG_CONN_STR.replace("postgresql://", "postgresql+psycopg://")
pg_engine = PGEngine.from_connection_string(url=engine_url)

vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    embedding_service=embedding_service,
    table_name=TABLE_NAME,
    id_column="id",
    content_column="text",
)


def search_transactions(query: str, k: int = 5, month: str | None = None, min_balance: float | None = None) -> List[
    Dict[str, Any]]:
    """
    Semantic search over transaction_embeddings using pgvector similarity.
    Returns structured dicts for the agent to reason over.
    """

    docs: List[Document] = vector_store.similarity_search(query=query, k=k)

    results = []
    for d in docs:
        item = {
            "text": d.page_content,
            "metadata": d.metadata or {},
        }
        results.append(item)

    if month is not None:
        results = [r for r in results if (r["metadata"].get("month") == month)]

    if min_balance is not None:
        def parse_balance(m):
            b = m.get("balance")
            if b is None: return None
            try:
                return float(b)
            except Exception:
                return None

        results = [r for r in results if
                   (parse_balance(r["metadata"]) is not None and parse_balance(r["metadata"]) >= min_balance)]

    return results[:k]