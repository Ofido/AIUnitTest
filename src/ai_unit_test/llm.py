"""Legacy LLM module - deprecated constants and functions.

This module contains legacy constants that may still be referenced by old code.
All new implementations should use the LLM connector interfaces in core.interfaces.llm_connector.
"""

import logging

logger = logging.getLogger(__name__)

# Legacy constants - kept for backward compatibility
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# All functions in this module have been deprecated and moved to the new architecture.
# Use LLMConnectorFactory to create LLM connectors instead of direct function calls.


def generate_embeddings(
    texts: list[str],
    source_file_path: str,
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = False,
) -> list[list[float]]:
    """Legacy function - deprecated. Use LLMConnectorFactory instead."""
    logger.warning("generate_embeddings is deprecated. Use LLMConnectorFactory instead.")
    return []


async def update_test_with_llm(
    source_code: str,
    test_code: str,
    file_name: str,
    coverage_lines: list[int],
    other_tests_content: str,
    test_style: str,
) -> str:
    """Legacy function - deprecated. Use LLMConnectorFactory instead."""
    logger.warning("update_test_with_llm is deprecated. Use LLMConnectorFactory instead.")
    return ""
