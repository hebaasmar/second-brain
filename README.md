# Second Brain

Voice-powered semantic search over personal notes. Ask a question out loud, get the right note back in under 2 seconds.

## Problem

I keep structured notes in Notion - project retrospectives, technical decisions, things I've learned, stories worth remembering. The problem is retrieval. When I need a specific note mid-conversation or mid-thought, scrolling doesn't work. Keyword search fails because the way you ask a question rarely matches the way you wrote the note.

Semantic search fixes this. "Tell me about improving accuracy" matches a note about raising RAG precision from 78% to 96%, even though the word "accuracy" never appears in that note.

## Architecture
```
Voice Input → Whisper (local) → Text Query
                                    ↓
                          sentence-transformers
                          (all-MiniLM-L6-v2)
                                    ↓
                          384-dim query vector
                                    ↓
                    Cosine similarity vs. stored note embeddings
                                    ↓
                          Ranked results + scores
```

### Pipeline

1. **Ingestion**: Notion API pulls notes, parses each into beat-level chunks with metadata (project, topic, tags)
2. **Embedding**: all-MiniLM-L6-v2 encodes each chunk into a 384-dimensional vector, stored as JSON
3. **Query**: Whisper transcribes voice input, same model embeds the transcript, cosine similarity ranks all chunks
4. **Retrieval**: Top-k results returned with similarity scores and source metadata

### Why These Tools

**all-MiniLM-L6-v2 over OpenAI ada-002**: Local inference, no API cost, no key management. At ~100 chunks, the quality gap between a 384-dim and 1536-dim model is negligible. Latency matters more here than marginal recall improvement.

**Whisper base over Whisper large or an API**: Base model transcription adds ~1s latency on CPU. For short queries (5-15 words), base is accurate enough. Keeping it local means no network dependency and no voice data leaving the machine.

**Cosine similarity over FAISS/ANN**: With <200 vectors, brute-force cosine search completes in <1ms. Adding an approximate nearest neighbor index would be overhead with zero benefit at this scale. FAISS becomes relevant around 10K+ vectors.

**Chunking by semantic unit, not by token window**: Notes are pre-structured in Notion with named sections. Chunk boundaries carry semantic meaning rather than being arbitrary token splits. A "metric" section contains the quantitative result. An "action" section contains what happened. Search results map directly to the piece you actually need. This mirrors a core lesson from building RAG pipelines at Kunik: chunk strategy drives retrieval quality more than model selection.

## File Structure
```
├── app.py                          # Flask web interface
├── main.py                         # Notion sync + embedding generation
├── embeddings.py                   # Model loading, vector search
├── chunks.json                     # Generated locally, gitignored
├── chunks_with_embeddings.json     # Generated locally, gitignored
└── pyproject.toml                  # Dependencies (managed with uv)
```

## Setup
```bash
git clone https://github.com/hebaasmar/second-brain.git
cd second-brain

echo "NOTION_TOKEN=your_token_here" > .env

# Install dependencies
uv sync

# Pull notes from Notion and generate embeddings
python main.py

# Launch
python app.py
```

## What I'd Do Differently at Scale

**Vector store**: JSON works at 100 chunks. At 1K+, I'd move to a proper vector DB (pgvector for simplicity, Pinecone if multi-user). The embedding and search interfaces are already abstracted enough that swapping storage is a one-file change.

**Chunk enrichment**: Right now each chunk is embedded in isolation. Prepending the note title and project name before embedding would improve retrieval when the query references context ("that Kunik thing about...") rather than content.

**Full-note retrieval**: Current search returns individual chunks. The next version should return all chunks from a matched note in sequence, so you get the complete context instead of just the fragment that scored highest.

**Streaming transcription**: Whisper currently processes after you stop speaking. A streaming model (whisper.cpp or Deepgram) would let results appear as you talk.

## Current State

End-to-end pipeline works: speak a question, get ranked note chunks with similarity scores. Tested across different query phrasings with consistent retrieval at 0.51+ similarity for relevant matches.

Next: expanding the note corpus and adding full-note retrieval.
