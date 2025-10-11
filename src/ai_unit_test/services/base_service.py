"""Base service class for common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    """Base class for all service implementations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize service with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_config(self, required_keys: list[str]) -> None:
        """Validate that required configuration keys are present."""
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    @abstractmethod
    def get_service_name(self) -> str:
        """Return the name of this service for logging purposes."""
        pass
