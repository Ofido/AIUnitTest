"""Factory for creating index organizers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_unit_test.core.interfaces.index_organizer import IndexOrganizer

from ai_unit_test.core.exceptions import ConfigurationError


class IndexOrganizerFactory:
    """Factory for creating index organizer instances."""

    _organizers: dict[str, type["IndexOrganizer"]] = {}
    _availability_cache: dict[str, bool] = {}

    @classmethod
    def register_organizer(
        cls: type["IndexOrganizerFactory"],
        name: str,
        organizer_class: type["IndexOrganizer"],
    ) -> None:
        """Register a new organizer type."""
        cls._organizers[name.lower()] = organizer_class

    @classmethod
    def get_available_organizers(cls: type["IndexOrganizerFactory"]) -> list[str]:
        """Get list of available organizer names."""
        available = []
        for name, organizer_class in cls._organizers.items():
            if cls._check_availability(name, organizer_class):
                available.append(name)
        return available

    @classmethod
    def create_organizer(
        cls: type["IndexOrganizerFactory"],
        backend: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> "IndexOrganizer":
        """Create an organizer instance."""
        if config is None:
            config = {}

        # Auto-detect backend if not specified
        if backend is None:
            backend = cls._auto_detect_backend()

        backend_lower = backend.lower()

        if backend_lower not in cls._organizers:
            available = ", ".join(cls.get_available_organizers())
            raise ConfigurationError(f"Unknown index backend: {backend}. " f"Available backends: {available}")

        organizer_class = cls._organizers[backend_lower]

        # Check if backend is actually available
        if not cls._check_availability(backend_lower, organizer_class):
            raise ConfigurationError(
                f"Backend {backend} is registered but not available. " f"Check dependencies and installation."
            )

        # Merge with default config
        merged_config = cls._merge_default_config(backend_lower, config)

        return organizer_class(merged_config)

    @classmethod
    def create_from_config_file(cls: type["IndexOrganizerFactory"], config: dict[str, Any]) -> "IndexOrganizer":
        """Create organizer from pyproject.toml configuration."""
        indexing_config = config.get("tool", {}).get("ai-unit-test", {}).get("indexing", {})

        backend = indexing_config.get("backend", "auto")
        backend_config = indexing_config.get(backend, {}) if backend != "auto" else {}

        # Merge general indexing config with backend-specific config
        merged_config = {**indexing_config, **backend_config}

        if backend == "auto":
            backend = None  # Let auto-detection handle it

        return cls.create_organizer(backend, merged_config)

    @classmethod
    def _auto_detect_backend(cls: type["IndexOrganizerFactory"]) -> str:
        """Auto-detect the best available backend."""
        # Priority order: faiss (fastest) -> sklearn (fallback) -> memory (testing)
        priority_order = ["faiss", "sklearn", "memory"]

        for backend in priority_order:
            if backend in cls._organizers:
                organizer_class = cls._organizers[backend]
                if cls._check_availability(backend, organizer_class):
                    return backend

        raise ConfigurationError("No index backends available. Please install faiss-cpu or scikit-learn.")

    @classmethod
    def _check_availability(
        cls: type["IndexOrganizerFactory"],
        name: str,
        organizer_class: type["IndexOrganizer"],
    ) -> bool:
        """Check if an organizer backend is available."""
        if name in cls._availability_cache:
            return cls._availability_cache[name]

        try:
            # Try to import required dependencies for this backend
            if name == "faiss":
                import faiss  # noqa: F401

                available = True
            elif name == "sklearn":
                import joblib  # type: ignore[import-untyped] # noqa: F401
                import sklearn.neighbors  # type: ignore[import-untyped] # noqa: F401

                available = True
            elif name == "memory":
                # In-memory backend is always available
                available = True
            else:
                # For custom backends, assume available if registered
                available = True

        except ImportError:
            available = False

        cls._availability_cache[name] = available
        return available

    @classmethod
    def _merge_default_config(
        cls: type["IndexOrganizerFactory"], backend: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge configuration with backend-specific defaults."""
        defaults: dict[str, dict[str, Any]] = {
            "faiss": {
                "index_type": "IndexFlatIP",
                "normalize_embeddings": True,
                "nlist": 100,
            },  # for IVF indices
            "sklearn": {
                "algorithm": "ball_tree",
                "metric": "cosine",
                "n_neighbors": 10,
                "n_jobs": -1,
            },
            "memory": {"max_documents": 10000, "enable_persistence": False},
        }

        backend_defaults: dict[str, Any] = defaults.get(backend, {})
        return {**backend_defaults, **config}


# Auto-register organizers when they're imported
def _register_default_organizers() -> None:
    """Register default organizers."""
    try:
        from ai_unit_test.core.implementations.indexing.faiss_organizer import FaissIndexOrganizer

        IndexOrganizerFactory.register_organizer("faiss", FaissIndexOrganizer)
    except ImportError:
        pass

    try:
        from ai_unit_test.core.implementations.indexing.sklearn_organizer import SklearnIndexOrganizer

        IndexOrganizerFactory.register_organizer("sklearn", SklearnIndexOrganizer)
    except ImportError:
        pass

    try:
        from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer

        IndexOrganizerFactory.register_organizer("memory", InMemoryIndexOrganizer)
    except ImportError:
        pass


# Register default organizers on module import
_register_default_organizers()
