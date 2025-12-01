# agent.py
import logging
import sys

from google.adk.plugins.logging_plugin import LoggingPlugin
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner

from agents.tools.sql_query import sql_query_tool
from agents.tools.vector_search import search_transactions

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    stream=sys.stdout
)

load_dotenv()


SYSTEM_INSTRUCTION = """
You are a personal finance assistant.
Use tools to fetch and compute real transaction data.

Tools available:
1) vector_search(query: str, k: int) -> returns semantic matches from transaction_embeddings (metadata includes transaction_id)
2) sql_query(sql: str) -> executes a safe SELECT on 'transactions' table and returns rows

Database schema for 'transactions' table:
id (int), transaction_date (date), posted_date (date), merchant_name (text),
merchant_raw (text), category (text), amount (numeric), transaction_type (text),
city (text), state (text), country (text), balance (numeric), created_at (timestamp)

When generating SQL:
- Output only a single SELECT statement (no semicolons).
- Only use columns in the schema above.
- If unsure about category names, call vector_search first to find matching transactions.
Return tool invocation parameters as JSON where the planner expects them.
"""

logging_plugin = LoggingPlugin()
logger.info("Initializing Financial Assistant Agent with LoggingPlugin.")

vector_tool = FunctionTool(search_transactions)
sql_tool = FunctionTool(sql_query_tool)

root_agent = Agent(
    name="finance_agent",
    model="gemini-2.5-flash-lite",
    instruction=SYSTEM_INSTRUCTION,
    tools=[vector_tool, sql_tool],
)

runner = InMemoryRunner(
    agent=root_agent,
    plugins=[
        LoggingPlugin()
    ],
)

logger.info("Financial Assistant Agent initialized.")

if __name__ == "__main__":
    print("Agent ready. Use `adk run` from the parent directory per ADK quickstart.")