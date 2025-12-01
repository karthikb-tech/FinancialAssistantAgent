# tools/sql_query.py
import os, re, logging
from typing import Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, future=True)


ALLOWED_COLUMNS = {
    "id","transaction_date","description","debit","credit","category"
}

FORBIDDEN_RE = re.compile(r";|--|\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b|\bALTER\b|\bTRUNCATE\b", flags=re.I)
SELECT_SIMPLE_RE = re.compile(
    r"^\s*SELECT\s+(.+?)\s+FROM\s+transactions(\s+WHERE\s+.+)?(\s+GROUP\s+BY\s+.+)?(\s+ORDER\s+BY\s+.+)?(\s+LIMIT\s+\d+)?\s*$",
    flags=re.I | re.S
)

def _validate_select(sql_query: str) -> str:
    if FORBIDDEN_RE.search(sql_query):
        raise ValueError("Forbidden SQL keywords detected.")
    m = SELECT_SIMPLE_RE.match(sql_query.strip())
    if not m:
        raise ValueError("Only simple SELECT queries from 'transactions' table are allowed.")
    select_cols = m.group(1).strip()
    if select_cols != "*" and select_cols.lower() != "distinct *":
        cols = [c.strip().strip('"') for c in select_cols.split(",")]
        for c in cols:
            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", c)
            for t in tokens:
                if t.lower() in {"sum","avg","min","max","count","distinct","as"}:
                    continue
                if t not in ALLOWED_COLUMNS:
                    raise ValueError(f"Column or token '{t}' is not allowed.")
    return sql_query

def sql_query_tool(sql_query: str) -> Dict[str, Any]:
    clean_sql = _validate_select(sql_query)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(clean_sql))
            rows = [dict(r) for r in result.mappings().all()]
            return {"status": "success", "rows": rows}
    except Exception as e:
        logging.error("SQL execution failed: %s", e)
        return {"status": "error", "error": str(e)}
