import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from ai_unit_test.indexing import (
    FAISS_AVAILABLE,
    FAISS_INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    SCHEMA_VERSION,
    SKLEARN_INDEX_FILENAME,
    load_faiss_index,
    save_faiss_index,
)
from ai_unit_test.llm import EMBEDDING_MODEL

if FAISS_AVAILABLE:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")
        import faiss


def test_save_faiss_index_validation_error() -> None:
    """Tests that save_faiss_index raises validation errors for bad inputs."""
    with pytest.raises(ValueError, match="The embeddings list is empty"):
        save_faiss_index([], [], "any_dir")

    with pytest.raises(ValueError, match="The number of embeddings .* does not match"):
        save_faiss_index([np.random.rand(128).tolist()], [], "any_dir")

    with pytest.raises(ValueError, match="Embeddings have inconsistent dimensions"):
        save_faiss_index([[0.1], [0.2, 0.3]], [{"a": 1}, {"b": 2}], "any_dir")

    with pytest.raises(ValueError, match="Not all items in the metadata list are valid"):
        save_faiss_index([[0.1]], ["not-a-dict"], "any_dir")


def test_save_faiss_index_mocked(tmp_path: Path, mocker: MockerFixture) -> None:
    """Tests the behavior of save_faiss_index by mocking external libraries."""
    # 1. Setup
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    # 2. Mocking
    mock_faiss_index = MagicMock()
    mock_write_index = mocker.patch("ai_unit_test.indexing.faiss.write_index")
    mocker.patch("ai_unit_test.indexing.faiss.IndexFlatIP", return_value=mock_faiss_index)
    mocker.patch("ai_unit_test.indexing.faiss.normalize_L2")

    # 3. Execution
    save_faiss_index(embeddings, metadata, str(index_dir))

    # 4. Asserts
    mock_faiss_index.add.assert_called_once()
    mock_write_index.assert_called_once_with(mock_faiss_index, str(index_dir / FAISS_INDEX_FILENAME))

    manifest_path = index_dir / MANIFEST_FILENAME
    metadata_path = index_dir / METADATA_FILENAME
    assert manifest_path.exists()
    assert metadata_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)
        assert manifest["chunk_count"] == 2
        assert manifest["vector_dimension"] == 3

    with open(metadata_path) as f:
        loaded_metadata = json.load(f)
        assert loaded_metadata == metadata


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu is not installed")
def test_save_and_load_faiss_index_integration(tmp_path: Path) -> None:
    """Performs an integration test of saving and loading with real FAISS."""
    # Create dummy embeddings and metadata
    embeddings = np.random.rand(10, 128).astype(np.float32)
    metadata: list[dict[str, Any]] = [{"test": f"meta{i}"} for i in range(10)]

    # Define index path
    index_dir = tmp_path / "test_index"

    # Save the index
    save_faiss_index(embeddings.tolist(), metadata, str(index_dir))

    # Check if the index file was created
    assert (index_dir / FAISS_INDEX_FILENAME).exists()
    assert (index_dir / METADATA_FILENAME).exists()
    assert (index_dir / MANIFEST_FILENAME).exists()

    # Load the index
    loaded_index, loaded_metadata, manifest = load_faiss_index(str(index_dir))

    # Check if the loaded index has the correct number of embeddings
    assert isinstance(loaded_index, faiss.Index)  # pyright: ignore[reportPossiblyUnboundVariable]
    assert loaded_index.ntotal == 10
    assert len(loaded_metadata) == 10
    assert loaded_metadata[0]["test"] == "meta0"
    assert manifest["chunk_count"] == 10


def test_load_non_existent_index(tmp_path: Path) -> None:
    """Tests that loading a non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_faiss_index(str(tmp_path / "non_existent_index"))


def test_save_and_load_sklearn_index(tmp_path: Path, mocker: MockerFixture) -> None:
    """Tests saving and loading using scikit-learn when FAISS is unavailable."""
    mocker.patch("ai_unit_test.indexing.FAISS_AVAILABLE", False)
    mock_joblib_dump = mocker.patch("joblib.dump")
    mock_joblib_load = mocker.patch("joblib.load")

    embeddings = np.random.rand(10, 128).tolist()
    metadata: list[dict[str, Any]] = [{"id": i} for i in range(10)]
    index_dir = tmp_path / "sklearn_index"

    # Save
    save_faiss_index(embeddings, metadata, str(index_dir))
    mock_joblib_dump.assert_called_once()

    # Create the mock index file that would be created by joblib.dump
    sklearn_index_path = index_dir / SKLEARN_INDEX_FILENAME
    sklearn_index_path.touch()  # Create empty file

    # Create the metadata file as well
    metadata_path = index_dir / METADATA_FILENAME
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    assert (index_dir / MANIFEST_FILENAME).exists()

    # Load
    mock_joblib_load.return_value = "fake-sklearn-index"
    loaded_index, _, manifest = load_faiss_index(str(index_dir))
    mock_joblib_load.assert_called_once()
    assert loaded_index == "fake-sklearn-index"
    assert manifest["index_type"] == "sklearn"


def test_faiss_availability() -> None:
    """Tests the FAISS availability flag."""
    assert FAISS_AVAILABLE == (not isinstance(faiss, MagicMock))  # pyright: ignore[reportPossiblyUnboundVariable]


def test_save_faiss_index_valid_inputs() -> None:
    """Tests that save_faiss_index successfully saves valid inputs."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = Path("valid_index_dir")

    # Create the directory
    index_dir.mkdir(exist_ok=True)

    # Execute the function
    save_faiss_index(embeddings, metadata, str(index_dir))

    # Check if the index file and metadata file were created
    assert (index_dir / FAISS_INDEX_FILENAME).exists()
    assert (index_dir / METADATA_FILENAME).exists()
    assert (index_dir / MANIFEST_FILENAME).exists()


def test_save_faiss_index_manifest_content(tmp_path: Path) -> None:
    """Tests that the manifest file contains the correct content."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    save_faiss_index(embeddings, metadata, str(index_dir))

    manifest_path = index_dir / MANIFEST_FILENAME
    with open(manifest_path) as f:
        manifest = json.load(f)
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["embedding_model"] == EMBEDDING_MODEL
        assert manifest["vector_dimension"] == 3
        assert manifest["chunk_count"] == 2
        assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_with_valid_data(tmp_path: Path) -> None:
    """Tests loading a FAISS index with valid data."""
    embeddings = np.random.rand(10, 128).astype(np.float32)
    metadata: list[dict[str, Any]] = [{"id": i} for i in range(10)]
    index_dir = tmp_path / "valid_index"

    # Save the index first
    save_faiss_index(embeddings.tolist(), metadata, str(index_dir))

    # Now load the index
    loaded_index, loaded_metadata, manifest = load_faiss_index(str(index_dir))

    # Assertions
    assert isinstance(loaded_index, faiss.Index)  # pyright: ignore[reportPossiblyUnboundVariable]
    assert loaded_index.ntotal == 10
    assert len(loaded_metadata) == 10
    assert loaded_metadata[0]["id"] == 0
    assert manifest["chunk_count"] == 10
    assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_manifest_version_mismatch(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises a ValueError for schema version mismatch."""
    index_dir = tmp_path / "mismatched_version_index"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with a mismatched schema version
    manifest = {
        "schema_version": "1.0",  # Assuming SCHEMA_VERSION is not "1.0"
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Schema version mismatch. Expected"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_index_file_not_found(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises FileNotFoundError if the index file is missing."""
    index_dir = tmp_path / "missing_index_file"
    index_dir.mkdir(exist_ok=True)

    # Create a valid manifest but do not create the index file
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(FileNotFoundError, match="Index file not found at"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_metadata_file_not_found(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises FileNotFoundError if the metadata file is missing."""
    index_dir = tmp_path / "missing_metadata_file"
    index_dir.mkdir(exist_ok=True)

    # Create a valid FAISS index file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")
        import faiss
    import numpy as np

    embeddings = [np.random.rand(128).astype("float32") for _ in range(10)]
    embeddings_array = np.array(embeddings)

    # Create and save a valid FAISS index
    index = faiss.IndexFlatL2(128)
    index.add(embeddings_array)  # pyright: ignore[reportCallIssue]
    faiss.write_index(index, str(index_dir / FAISS_INDEX_FILENAME))

    # Create a valid manifest but do not create the metadata file
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(FileNotFoundError, match="Metadata file not found at"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_inconsistent_metadata_length(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for inconsistent metadata length."""
    index_dir = tmp_path / "inconsistent_metadata_length"
    index_dir.mkdir(exist_ok=True)

    # Create a valid manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    # Create a dummy index and metadata
    faiss_index = faiss.IndexFlatL2(128)  # pyright: ignore[reportPossiblyUnboundVariable]
    faiss_index.add(np.random.rand(5, 128).astype(np.float32))  # 5 vectors # pyright: ignore[reportCallIssue]
    faiss.write_index(  # pyright: ignore[reportPossiblyUnboundVariable]
        faiss_index, str(index_dir / FAISS_INDEX_FILENAME)
    )

    # Create metadata with inconsistent length
    metadata = [{"id": i} for i in range(3)]  # 3 entries
    with open(index_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(ValueError, match="Inconsistency detected: FAISS index has"):
        load_faiss_index(str(index_dir))


def test_schema_version_constant() -> None:
    """Tests that the schema version constant is correctly set."""
    assert SCHEMA_VERSION == "1.1.0"


def test_faiss_index_filename_constant() -> None:
    """Tests that the FAISS index filename constant is correctly set."""
    assert FAISS_INDEX_FILENAME == "index.faiss"


def test_sklearn_index_filename_constant() -> None:
    """Tests that the scikit-learn index filename constant is correctly set."""
    assert SKLEARN_INDEX_FILENAME == "index.joblib"


def test_load_faiss_index_manifest_file_not_found(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises FileNotFoundError if the manifest file is missing."""
    index_dir = tmp_path / "missing_manifest_file"
    index_dir.mkdir(exist_ok=True)
    with open(index_dir / FAISS_INDEX_FILENAME, "w") as f:
        f.write("dummy index data")  # Create a dummy index file

    with pytest.raises(FileNotFoundError, match="Manifest file not found at"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_index_type(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type in the manifest."""
    index_dir = tmp_path / "invalid_index_type"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "invalid_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_index_type_faiss(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type in the
    manifest when FAISS is available.
    """
    index_dir = tmp_path / "invalid_index_type_faiss"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "invalid_faiss_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_index_type_sklearn(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type
    in the manifest when sklearn is used."""
    index_dir = tmp_path / "invalid_index_type_sklearn"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": SKLEARN_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "invalid_sklearn_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_schema_version_constant_is_incremented() -> None:
    """Tests that the schema version constant is incremented for new structures."""
    assert SCHEMA_VERSION != "1.0.0"  # Ensure it's not the old version


def test_save_faiss_index_manifest_file_creation(tmp_path: Path) -> None:
    """Tests that the manifest file is created with the correct content."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    save_faiss_index(embeddings, metadata, str(index_dir))

    manifest_path = index_dir / MANIFEST_FILENAME
    with open(manifest_path) as f:
        manifest = json.load(f)
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["embedding_model"] == EMBEDDING_MODEL
        assert manifest["vector_dimension"] == 3
        assert manifest["chunk_count"] == 2
        assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_invalid_index_type_faiss_with_valid_data(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type
    in the manifest when FAISS is available and valid data is present."""
    index_dir = tmp_path / "invalid_index_type_faiss_with_data"
    index_dir.mkdir(exist_ok=True)

    # Create a valid FAISS index file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")
        import faiss
    import numpy as np

    embeddings = np.random.rand(10, 128).astype(np.float32)
    index = faiss.IndexFlatL2(128)
    index.add(embeddings)  # pyright: ignore[reportCallIssue]
    faiss.write_index(index, str(index_dir / FAISS_INDEX_FILENAME))

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "invalid_faiss_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_save_faiss_index_manifest_file_creation_with_additional_fields(tmp_path: Path) -> None:
    """Tests that the manifest file is created with additional fields."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    save_faiss_index(embeddings, metadata, str(index_dir))

    manifest_path = index_dir / MANIFEST_FILENAME
    with open(manifest_path) as f:
        manifest = json.load(f)
        assert "created_at" in manifest
        assert "index_file" in manifest
        assert "metadata_file" in manifest
        assert "index_type" in manifest
        assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_invalid_index_type_faiss_with_no_data(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type in
    the manifest when no data is present."""
    index_dir = tmp_path / "invalid_index_type_faiss_no_data"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 0,
        "vector_dimension": 128,
        "index_type": "invalid_faiss_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_schema_version(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid schema version in the manifest."""
    index_dir = tmp_path / "invalid_schema_version"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid schema version
    manifest = {
        "schema_version": "0.0",  # Invalid version
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Schema version mismatch. Expected"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_with_missing_index_file(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises FileNotFoundError if the index file is missing."""
    index_dir = tmp_path / "missing_index_file"
    index_dir.mkdir(exist_ok=True)

    # Create a valid manifest but do not create the index file
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(FileNotFoundError, match="Index file not found at"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_index_type_faiss_with_no_data_and_invalid_type(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type in the manifest
    when no data is present."""
    index_dir = tmp_path / "invalid_index_type_faiss_no_data_invalid"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 0,
        "vector_dimension": 128,
        "index_type": "invalid_faiss_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_save_faiss_index_manifest_creation_with_all_fields(tmp_path: Path) -> None:
    """Tests that the manifest file is created with all expected fields."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    save_faiss_index(embeddings, metadata, str(index_dir))

    manifest_path = index_dir / MANIFEST_FILENAME
    with open(manifest_path) as f:
        manifest = json.load(f)
        assert "schema_version" in manifest
        assert "embedding_model" in manifest
        assert "vector_dimension" in manifest
        assert "index_file" in manifest
        assert "metadata_file" in manifest
        assert "created_at" in manifest
        assert "chunk_count" in manifest
        assert "index_type" in manifest
        assert manifest["chunk_count"] == 2
        assert manifest["vector_dimension"] == 3
        assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_invalid_index_type_sklearn_with_no_data(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type in the
    manifest when no data is present for sklearn."""
    index_dir = tmp_path / "invalid_index_type_sklearn_no_data"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": SKLEARN_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 0,
        "vector_dimension": 128,
        "index_type": "invalid_sklearn_type",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_schema_version_with_invalid_type(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid schema version
    in the manifest with an invalid type."""
    index_dir = tmp_path / "invalid_schema_version_invalid_type"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid schema version
    manifest = {
        "schema_version": "0.0",  # Invalid version
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Schema version mismatch. Expected"):
        load_faiss_index(str(index_dir))


def test_save_faiss_index_manifest_creation_with_all_fields_and_additional_info(tmp_path: Path) -> None:
    """Tests that the manifest file is created with all expected fields and additional info."""
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    index_dir = tmp_path / "test_index"

    save_faiss_index(embeddings, metadata, str(index_dir))

    manifest_path = index_dir / MANIFEST_FILENAME
    with open(manifest_path) as f:
        manifest = json.load(f)
        assert "created_at" in manifest
        assert "index_file" in manifest
        assert "metadata_file" in manifest
        assert "index_type" in manifest
        assert manifest["chunk_count"] == 2
        assert manifest["vector_dimension"] == 3
        assert manifest["index_type"] == "faiss" if FAISS_AVAILABLE else "sklearn"


def test_load_faiss_index_invalid_index_type_sklearn_with_no_data_and_invalid_type(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid index type
    in the manifest when no data is present for sklearn."""
    index_dir = tmp_path / "invalid_index_type_sklearn_no_data_invalid"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid index type
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_file": SKLEARN_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 0,
        "vector_dimension": 128,
        "index_type": "invalid_sklearn_type_invalid",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Invalid index type"):
        load_faiss_index(str(index_dir))


def test_load_faiss_index_invalid_schema_version_with_invalid_type_and_no_data(tmp_path: Path) -> None:
    """Tests that loading a FAISS index raises ValueError for an invalid schema version
    in the manifest with an invalid type and no data."""
    index_dir = tmp_path / "invalid_schema_version_invalid_type_no_data"
    index_dir.mkdir(exist_ok=True)

    # Create a manifest with an invalid schema version
    manifest = {
        "schema_version": "0.0",  # Invalid version
        "index_file": FAISS_INDEX_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "chunk_count": 10,
        "vector_dimension": 128,
        "index_type": "faiss",
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="Schema version mismatch. Expected"):
        load_faiss_index(str(index_dir))
