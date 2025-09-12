"""Legacy indexing module.

This module is deprecated and will be removed in a future version.
All indexing-related functionality has been moved to the new architecture under `ai_unit_test.core`.

- For index organizers, see `ai_unit_test.core.implementations.indexing`.
- For the index organizer interface, see `ai_unit_test.core.interfaces.index_organizer`.
- To create index organizers, use `ai_unit_test.core.factories.index_factory`.
"""

import warnings

warnings.warn(
    "The 'ai_unit_test.indexing' module is deprecated and will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
