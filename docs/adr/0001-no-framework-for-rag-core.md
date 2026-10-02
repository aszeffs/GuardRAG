# No LangChain or LlamaIndex for the RAG core

Retrieval (vector search, full-text search, Reciprocal Rank Fusion), prompting, structured output and citation checking are written by hand. The embedder, retriever and LLM each sit behind a small interface so they can be swapped and compared. A framework would hide exactly the parts this project exists to measure and explain, and would make it harder to swap one component without changing the others. Libraries are still used at the edges (FastAPI, psycopg, fastembed, the Groq SDK, Ragas, promptfoo).
