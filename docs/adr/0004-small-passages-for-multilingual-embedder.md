# Passages are capped at 128 embedder tokens

We embed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, about 0.2 GB, runs on CPU through fastembed), because questions arrive in English, Filipino and Taglish while the documents are mostly English. That model truncates its input at 128 tokens, so anything past the first 128 tokens of a passage is never embedded. Passages are therefore capped at 128 tokens, counted with the embedder's own tokenizer, instead of the 500–800 tokens common in RAG tutorials.

## Considered Options

- **multilingual-e5-large** (512-token window): better multilingual quality, but 2.2 GB, 1024-dim and slow on CPU. CI would have to cache the model and the embeddings.
- **Small-to-big** (embed small passages, send their parent section to the LLM): kept as a later upgrade if answers lack context.
