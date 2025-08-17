import asyncio
from src.provider.embedding.get_embedding import get_embeddings_from_text
from src.core.qdrant_client import client, COLLECTION_NAME

async def query():
    query_text = "Tình tiết giảm nhẹ trách nhiệm hình"
    query_emb = await get_embeddings_from_text([query_text], provider="siliconflow")
    query_vec = query_emb[0]["embedding"]

    # collection_info = client.get_collection(COLLECTION_NAME)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=5,
        with_vectors=False
    )

    # Ghi kết quả vào file txt
    with open("query_results.txt", "w", encoding="utf-8") as f:
        f.write("Top hits:\n")
        for hit in hits:
            line = f"{hit.id}\t{hit.score}\t{hit.payload.get('chunk_index')}\t{hit.payload.get('text')}\n------------------------\n"
            f.write(line)

    print("✅ Query results saved to query_results.txt")

if __name__ == "__main__":
    asyncio.run(query())
