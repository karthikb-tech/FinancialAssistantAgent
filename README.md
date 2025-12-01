Problem Statement
Managing personal finances is complex and often fragmented. Users struggle to understand their spending patterns, track balances, and extract insights from raw transaction data. Traditional dashboards offer charts and filters, but they lack semantic understanding and conversational flexibility.

The problem:
How can we make financial data conversationally accessible, blending structured analytics with semantic search — so users can ask questions like “What were my childcare expenses last quarter?” or “Show me transactions similar to rent payments” and get meaningful answers?

This is important because it democratizes financial literacy, reduces cognitive load, and empowers users to make informed decisions without needing SQL or spreadsheet skills.

** Why agents**
Agents are ideal for this problem because they:
Reason across tools: They can decide whether to use semantic search or structured SQL based on the user's intent.
Handle ambiguity: Users don’t always know the exact filters — agents can interpret fuzzy queries like “low balance months.”
Orchestrate multi-step logic: Agents can combine vector search with SQL queries to answer hybrid questions.
Scale with new capabilities: As we add forecasting, anomaly detection, or external APIs, agents can integrate them seamlessly.
Agents turn static data into dynamic, intelligent conversations.

What you created
`User → Agent (Gemini via Agents Toolkit)
├── Vector Search Tool (pgvector + LangChain)
├── SQL Query Tool (validated SQL via SQLAlchemy)
└── Future Tools (budgeting, anomaly detection)

Data Layer:

transactions (structured)
transaction_embeddings (semantic, pgvector)
Ingestion:

uploadDataInDB.py → parses CSV, inserts transactions, generates embeddings
Embedding Model:

SentenceTransformer("all-MiniLM-L6-v2") → 384-dim vectors`
Demo
Query: “Show me rent payments and my balance at the end of March.”
Agent Response:
Uses vector search to find transactions semantically similar to “rent payments.”
Uses SQL tool to fetch March’s end-of-month balance.
Combines both into a natural language summary:
“You had 3 rent-related transactions in March totaling $3,500. Your balance at the end of the month was $2,689.74.”

The Build
PostgreSQL with pgvector extension for semantic search
SQLAlchemy for ORM and safe SQL execution
LangChain for vector store integration
SentenceTransformer (all-MiniLM-L6-v2) for embedding generation
Google Agents Toolkit for agent orchestration
FastAPI for ingestion and optional REST endpoints
Python for all backend logic
HNSW/IVFFlat indexing for fast similarity search

If I had more time, this is what I'd do

Add budget forecasting and goal tracking tools
Build a chat UI with memory and multi-turn reasoning
