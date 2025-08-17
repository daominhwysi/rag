from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient

COLLECTION_NAME = "documents"

client = QdrantClient(
    url="https://93de1f00-1bd9-4b8e-8d7c-7192daefd62c.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.fjsgHaltEwmg7FyCXlVbt75E6hjAMMUTGoS0Hg95j2k",
)