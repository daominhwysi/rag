import os
import asyncio
import httpx
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("SILICONFLOW_API_KEY")
URL = "https://api.siliconflow.com/v1/embeddings"

async def get_embeddings_from_input_siliconflow(text_input):
    payload = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": text_input,
        "dimensions": 512
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['data']
# [{'embedding': [0.00016868368,....., 0.01037359], 'index': 0, 'object': 'embedding'},]
async def main():
    results = await get_embeddings_from_input_siliconflow(['hi','hello'])
    print(results)
    result = await get_embeddings_from_input_siliconflow('hi')
    print(result)

if __name__ == "__main__":
    asyncio.run(main())