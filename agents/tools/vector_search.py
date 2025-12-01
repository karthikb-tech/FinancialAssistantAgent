# tools/vector_search.py
import os
import logging
from typing import List, Dict, Any
from langchain_postgres import PGVectorStore, PGEngine
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

# Set up logging for this module
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PG_CONN_STR = os.getenv("DATABASE_URL")
TABLE_NAME = "transaction_embeddings"

if not PG_CONN_STR:
    logger.critical("DATABASE_URL not found in environment variables.")
    raise ValueError("DATABASE_URL not found in environment variables.")

# 1. Initialize LangChain-compatible Embedding Service
logger.info(f"Initializing embedding service with model: {EMBEDDING_MODEL_NAME}")
embedding_service = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# 2. Create the SQLAlchemy Engine
engine_url = PG_CONN_STR.replace("postgresql://", "postgresql+psycopg://")
logger.info("Creating PGEngine and replacing connection string prefix.")
pg_engine = PGEngine.from_connection_string(url=engine_url)

# 3. Configure the PGVectorStore using the required synchronous class method
logger.info(f"Attempting to initialize PGVectorStore for table: {TABLE_NAME}")
try:
    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        embedding_service=embedding_service,
        table_name=TABLE_NAME,

        # FIX for schema mismatch
        id_column="id",
        content_column="text",
    )
    logger.info("PGVectorStore initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PGVectorStore with error: {e}")
    raise


def search_transactions(query: str, k: int = 5, month: str | None = None, min_balance: float | None = None) -> List[
    Dict[str, Any]]:
    """
    Semantic search over transaction_embeddings using pgvector similarity.
    Returns structured dicts for the agent to reason over.
    """
    logger.info(
        f"Executing vector search | Query: '{query}' | k: {k} | Filters: month={month}, min_balance={min_balance}")

    docs: List[Document] = []
    try:
        docs = vector_store.similarity_search(query=query, k=k)
        logger.info(f"Vector search returned {len(docs)} documents.")
    except Exception as e:
        logger.error(f"Error occurred during similarity search execution: {e}")
        # Return empty results on search failure
        return []

    results = []
    for d in docs:
        item = {
            "text": d.page_content,
            "metadata": d.metadata or {},
        }
        results.append(item)

    logger.debug(f"Pre-filtered results count: {len(results)}")

    # Optional post-filters (simplified for brevity)
    if month is not None:
        results = [r for r in results if (r["metadata"].get("month") == month)]
        logger.debug(f"Results after month filter ('{month}'): {len(results)}")

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
        logger.debug(f"Results after balance filter (>= {min_balance}): {len(results)}")

    # Trim to k if filters were applied
    final_results = results[:k]
    logger.info(f"Final vector search results count: {len(final_results)}")
    # Log the content of the final results (helpful for debugging agent reasoning)
    logger.debug(f"Final results content: {final_results}")

    return final_results