"""Legacy LLM module.

This module is deprecated and will be removed in a future version.
All LLM-related functionality has been moved to the new architecture under `ai_unit_test.core`.

- For LLM connectors, see `ai_unit_test.core.implementations.llm`.
- For the LLM interface, see `ai_unit_test.core.interfaces.llm_connector`.
- To create LLM connectors, use `ai_unit_test.core.factories.llm_factory`.
"""

import warnings

warnings.warn(
    "The 'ai_unit_test.llm' module is deprecated and will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
