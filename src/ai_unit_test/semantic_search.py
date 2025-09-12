"""Legacy semantic search module.

This module is deprecated and will be removed in a future version.
All search-related functionality has been moved to the new architecture under `ai_unit_test.core`.

- For search functionality, see the `search` method in `ai_unit_test.core.interfaces.index_organizer`.
"""

import warnings

warnings.warn(
    "The 'ai_unit_test.semantic_search' module is deprecated and will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
