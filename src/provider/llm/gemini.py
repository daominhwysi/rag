from typing import Union
import os

from google import genai
from google.genai import types
from utils.logger_config import get_logger
from dotenv import load_dotenv

logger = get_logger()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

async def gemini_agent(
    model: str,
    contents: Union[types.ContentListUnion, types.ContentListUnionDict],
    config: types.GenerateContentConfigOrDict,
):

    while True:
        client = genai.Client(api_key=api_key)

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            if not isinstance(response.text, str) or not response.text.strip():
                raise ValueError("GeminiAgent: response.text is không hợp lệ")

            return response

        except Exception as e:
            logger.error("[GeminiAgent] Lỗi không xác định: %s – %s", type(e).__name__, e)
            raise
