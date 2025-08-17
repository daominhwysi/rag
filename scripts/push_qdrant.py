import numpy as np
import os
import json
from qdrant_client.models import PointStruct
from src.core.qdrant_client import client, COLLECTION_NAME
def load_chunk_files(base_dir: str):
    chunks = []
    for fname in os.listdir(base_dir):
        if fname.endswith(".json"):
            path = os.path.join(base_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks.append(data)
    return chunks

base_dir = "./data/embeddings"
chunks = load_chunk_files(base_dir)
print("Embedding length:", len(chunks[0]["embedding"]))
print("Collection size:", client.get_collection(COLLECTION_NAME).config.params.vectors.size)
print("Count before insert:", client.count(collection_name=COLLECTION_NAME))
points = []
for i, chunk in enumerate(chunks):
    vec = np.array(chunk["embedding"], dtype=np.float32).tolist()
    points.append(
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                
            }
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)
print("Count after insert:", client.count(collection_name=COLLECTION_NAME))
print(client.get_collection(COLLECTION_NAME))


# # Test lại
# print(client.retrieve(
#     collection_name=COLLECTION_NAME,
#     ids=[0, 1],
#     with_vectors=True
# ))
