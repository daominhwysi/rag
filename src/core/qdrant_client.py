from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("QDRANT_CLOUD")
URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "documents"
client = QdrantClient(
    url=URL,
    api_key=TOKEN,
)