import sys
import os
from pathlib import Path
from typing import List, Dict
from src.provider.embedding.get_embedding import get_embeddings_from_text
import asyncio
import hashlib
import json
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_raw_docs(folder_path: str, recursive: bool = False) -> List[str]:
    p = Path(folder_path)
    pattern = "**/*.txt" if recursive else "*.txt"
    docs: List[str] = []
    for path in p.glob(pattern):
        if path.is_file():
            try:
                docs.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return docs

def chunk_text_tiktoken(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
    min_chunk_tokens: int = 20,
    model: str = "cl100k_base"
) -> List[Dict]:
    """
    Chia text thành chunks dựa trên tokens (tiktoken).
    """
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    total_tokens = len(tokens)

    chunks = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + chunk_size_tokens, total_tokens)
        if total_tokens - start <= min_chunk_tokens:
            end = total_tokens

        chunk_text = encoding.decode(tokens[start:end])
        chunks.append({
            "text": chunk_text,
            "chunk_index": chunk_index,
            "token_start": start,
            "token_end": end,
        })

        next_start = end - overlap_tokens
        if next_start <= start or total_tokens - next_start <= min_chunk_tokens:
            next_start = end

        start = next_start
        chunk_index += 1

    return chunks

def save_chunk_embedding(base_dir: str, doc_id: str, chunk, embedding):
    """Lưu mỗi chunk vào 1 file riêng"""
    os.makedirs(base_dir, exist_ok=True)
    chunk_file = os.path.join(base_dir, f"{doc_id}_chunk{chunk['chunk_index']}.json")
    record = {
        "chunk_index": chunk["chunk_index"],
        "token_start": chunk["token_start"],
        "token_end": chunk["token_end"],
        "text": chunk["text"],
        "embedding": embedding["embedding"],
    }
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

def hash_doc(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def main():
    texts = load_raw_docs("./data/raw")
    STACK_CHUNK = 10
    base_dir = "./data/embeddings"

    for text in texts:
        doc_id = hash_doc(text)
        chunks = chunk_text_tiktoken(
            text=text,
            chunk_size_tokens=500,
            overlap_tokens=50
        )
        print(f"[{doc_id}] Total chunks: {len(chunks)}")

        for i in range(0, len(chunks), STACK_CHUNK):
            batch_chunks = chunks[i:i+STACK_CHUNK]
            batch_texts = [c["text"] for c in batch_chunks]
            results = await get_embeddings_from_text(batch_texts, provider="siliconflow")
            for chunk, emb in zip(batch_chunks, results):
                save_chunk_embedding(base_dir, doc_id, chunk, emb)

        print(f"[{doc_id}] Saved {len(chunks)} chunk files → {base_dir}")

if __name__ == "__main__":
    asyncio.run(main())
