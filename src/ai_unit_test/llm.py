import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


async def update_test_with_llm(
    source_code: str,
    test_code: str,
    file_name: str,
    coverage_lines: list[int],
    other_tests_content: str,
    test_style: str,
) -> str:
    """Calls the chat model to generate the new test file."""
    logger.info(f"Updating test for {file_name} with LLM.")
    logger.debug(
        f"Source code length: {len(source_code)}, Test code length: {len(test_code)}, Uncovered lines: {coverage_lines}"
    )

    api_key_val: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key_val:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise RuntimeError("Set the OPENAI_API_KEY environment variable with your OpenAI key.")

    api_url: str | None = os.environ.get("OPENAI_API_URL")

    client: AsyncOpenAI = AsyncOpenAI(api_key=api_key_val, base_url=api_url)

    system_msg_base = (
        "You are an expert Python test developer. Your task is to write new unit tests to cover missing lines "
        "in a given code chunk. You must follow the style of existing tests provided as reference. "
        "Your response must be only the new test code, without any explanations, comments, or markdown formatting. "
        "Your response must be only valid Python code."
    )

    if test_style == "unittest_class":
        system_msg_specific = (
            "Generate a new test method to be added inside a `unittest.TestCase` class. "
            "The method name should start with `test_` and it should accept `self` as its first argument."
        )
    elif test_style == "pytest_function":
        system_msg_specific = "Generate a new test function. The function name should start with `test_`."
    else:
        system_msg_specific = "Generate a new test function or method, adapting to the existing test file's style."

    system_msg = f"{system_msg_base} {system_msg_specific}"

    user_msg: str = f"""Here is the information for the test generation:

<file_to_be_tested>
{file_name}
</file_to_be_tested>

<uncovered_lines>
{coverage_lines}
</uncovered_lines>

<source_code_chunk>
{source_code}
</source_code_chunk>

<existing_tests>
{test_code}
</existing_tests>

<style_reference_tests>
{other_tests_content}
</style_reference_tests>
"""

    logger.debug(f"User message for LLM: {user_msg}")
    try:
        rsp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        response_content: str | None = rsp.choices[0].message.content
        if response_content is None:
            response_content = ""
        logger.debug(f"LLM response received: {response_content[:100]}...")
        return response_content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _create_cache_filepath(source_file_path: str, model_name: str, normalize: bool) -> str | None:
    """
    Creates a cache filepath based on source file content and parameters.

    Returns None if the source file cannot be read.
    """
    try:
        with open(source_file_path, "rb") as f:
            file_content = f.read()
        content_hash = hashlib.sha256(file_content).hexdigest()
        model_name_safe = model_name.replace("/", "_")
        norm_str = "norm" if normalize else "unorm"
        cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
        cache_dir = os.path.join(".cache", "embeddings")
        return os.path.join(cache_dir, cache_filename)
    except OSError as e:
        logger.warning(f"Could not read source file {source_file_path} for caching. Error: {e}")
        return None


def _load_embeddings_from_cache(cache_filepath: str) -> list[list[float]] | None:
    """
    Loads embeddings from cache file.

    Returns None if cache cannot be loaded.
    """
    try:
        logger.info(f"Cache hit. Loading embeddings from {cache_filepath}")
        with np.load(cache_filepath, allow_pickle=True) as data:
            embeddings: list[list[float]] = data["embeddings"].tolist()
            return embeddings
    except Exception as e:
        logger.warning(f"Could not load cache file {cache_filepath}: {e}")
        return None


def _generate_embeddings_with_model(texts: list[str], model_name: str, normalize: bool) -> np.ndarray | None:
    """
    Generates embeddings using SentenceTransformer model.

    Returns None if model fails to generate embeddings.
    """
    try:
        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=normalize)  # type: ignore
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None


def _save_embeddings_to_cache(
    cache_filepath: str, embeddings_np: np.ndarray, source_file_path: str, model_name: str, normalize: bool
) -> None:
    """Saves embeddings to cache file."""
    try:
        cache_dir = os.path.dirname(cache_filepath)
        os.makedirs(cache_dir, exist_ok=True)
        metadata: dict[str, Any] = {
            "source_file_path": source_file_path,
            "model_name": model_name,
            "normalized": normalize,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        np.savez_compressed(
            cache_filepath,
            embeddings=embeddings_np,
            metadata=np.array(metadata, dtype=object),
        )
        logger.info(f"Embeddings saved to cache at {cache_filepath}")
    except Exception as e:
        logger.error(f"Failed to save embeddings to cache: {e}")


def generate_embeddings(
    texts: list[str],
    source_file_path: str,
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = False,
) -> list[list[float]]:
    """
    Generates embeddings for a list of texts with intelligent caching.

    Args:
        texts (list[str]): A list of texts to embed.
        source_file_path (str): Path to the source file to generate a hash for caching.
        model_name (str): The name of the SentenceTransformer model to use.
        normalize (bool): Whether to normalize embeddings to unit length.

    Returns:
        list[list[float]]: A list of embeddings.
    """
    # Try to create cache filepath
    cache_filepath = _create_cache_filepath(source_file_path, model_name, normalize)
    if cache_filepath is None:
        logger.warning("Returning empty embeddings due to cache filepath creation failure.")
        return []

    # Check cache
    if os.path.exists(cache_filepath):
        cached_embeddings = _load_embeddings_from_cache(cache_filepath)
        if cached_embeddings is not None:
            return cached_embeddings

    # Generate embeddings
    logger.info(f"Cache miss. Generating embeddings for {len(texts)} texts.")
    embeddings_np = _generate_embeddings_with_model(texts, model_name, normalize)
    if embeddings_np is None:
        return []

    # Save to cache
    _save_embeddings_to_cache(cache_filepath, embeddings_np, source_file_path, model_name, normalize)

    logger.info("Embeddings generated successfully.")
    embeddings_list: list[list[float]] = embeddings_np.tolist()
    return embeddings_list
