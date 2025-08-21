from src.provider.embedding.siliconflow import get_embeddings_from_input_siliconflow
from typing import Union, List, Literal

async def get_embeddings_from_text(
    text: Union[str, List[str]],
    provider: Literal["siliconflow", "sample"]
):
    if isinstance(text, str):
        texts = [text]  
    elif isinstance(text, list):
        if not all(isinstance(t, str) for t in text):
            raise TypeError("All elements in `text` list must be str")
        texts = text
    else:
        raise TypeError("`text` must be str or list[str]")

    # --- Route provider ---
    match provider:
        case "siliconflow":
            embeddings = await get_embeddings_from_input_siliconflow(texts)
        case "sample":
            embeddings = [{"embedding": [0.0]*10} for _ in texts]  # dummy example
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    return embeddings
