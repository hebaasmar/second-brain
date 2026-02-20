import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (free, runs locally)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_chunks(filepath='chunks.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_chunks(chunks, filepath='chunks.json'):
    with open(filepath, 'w') as f:
        json.dump(chunks, f, indent=2)

def create_embeddings(chunks):
    """Create embeddings for all chunks."""
    print(f"Creating embeddings for {len(chunks)} chunks...")

    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()

    print("Done creating embeddings.")
    return chunks

def search(query, chunks, top_k=5):
    """Search for most relevant chunks."""
    query_embedding = model.encode([query])[0]

    scores = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk['embedding'])
        score = np.dot(query_embedding, chunk_embedding)
        scores.append(score)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            'score': float(scores[i]),
            'path': ' > '.join(chunks[i]['path']),
            'text': chunks[i]['text'][:500]
        })

    return results


def search_full(query, chunks, top_k=3):
    """Search for most relevant chunks, return full chunk objects (no truncation)."""
    query_embedding = model.encode([query])[0]

    scores = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk['embedding'])
        score = np.dot(query_embedding, chunk_embedding)
        scores.append(score)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        result = {k: v for k, v in chunks[i].items() if k != 'embedding'}
        result['score'] = float(scores[i])
        results.append(result)

    return results


def get_story_beats(company, story, all_chunks):
    """Return all beats for a story in order, stripping embeddings."""
    beats = [
        {k: v for k, v in c.items() if k != 'embedding'}
        for c in all_chunks
        if c.get('company') == company and c.get('story') == story
    ]

    def beat_num(chunk):
        m = re.match(r'Beat (\d+)', chunk.get('beat', ''))
        return int(m.group(1)) if m else 999

    beats.sort(key=beat_num)
    return beats

if __name__ == "__main__":
    # Test: load chunks and create embeddings
    chunks = load_chunks()
    chunks = create_embeddings(chunks)
    save_chunks(chunks, 'chunks_with_embeddings.json')

    # Test search
    print("\n--- Testing search ---")
    query = "product sense interview"
    results = search(query, chunks)

    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']:.3f}")
        print(f"    Path: {r['path']}")
        print(f"    {r['text'][:200]}...")