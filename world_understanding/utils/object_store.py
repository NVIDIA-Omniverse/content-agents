# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ObjectStore abstraction for workflow artifact storage."""

import json
import pickle
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ObjectStore(ABC):
    """Abstract key-value store for workflow artifacts."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key, return default if not found."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value for key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key-value pair."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        """Get all keys in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        pass


class InMemoryObjectStore(ObjectStore):
    """Dictionary-based object store for small objects."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key, return default if not found."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value for key."""
        self._store[key] = value

    def delete(self, key: str) -> None:
        """Delete key-value pair."""
        if key in self._store:
            del self._store[key]

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._store

    def keys(self) -> list[str]:
        """Get all keys in the store."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all data from the store."""
        self._store.clear()


class TempDirObjectStore(ObjectStore):
    """File-based object store with serialization for large objects."""

    def __init__(self, temp_dir: Path | None = None, serializer: str = "pickle"):
        """
        Initialize TempDirObjectStore.

        Args:
            temp_dir: Directory for storage (creates temp dir if None)
            serializer: Serialization format ("pickle" or "json")
        """
        if temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="object_store_"))
            self._cleanup_on_exit = True
        else:
            self._temp_dir = Path(temp_dir)
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_on_exit = False

        self._serializer = serializer
        self._metadata: dict[str, str] = {}  # key -> filename mapping

    def _get_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        if self._serializer == "json":
            return self._temp_dir / f"{safe_key}.json"
        else:
            return self._temp_dir / f"{safe_key}.pkl"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        if self._serializer == "json":
            return json.dumps(value).encode("utf-8")
        else:
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        if self._serializer == "json":
            return json.loads(data.decode("utf-8"))
        else:
            return pickle.loads(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key, return default if not found."""
        path = self._get_path(key)
        if not path.exists():
            return default

        try:
            with open(path, "rb") as f:
                data = f.read()
            return self._deserialize(data)
        except Exception as e:
            print(f"Error reading {key}: {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set value for key."""
        path = self._get_path(key)
        try:
            data = self._serialize(value)
            with open(path, "wb") as f:
                f.write(data)
            self._metadata[key] = path.name
        except Exception as e:
            raise RuntimeError(f"Failed to store {key}: {e}") from e

    def delete(self, key: str) -> None:
        """Delete key-value pair."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        if key in self._metadata:
            del self._metadata[key]

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_path(key).exists()

    def keys(self) -> list[str]:
        """Get all keys in the store."""
        # Scan directory for files
        keys = []
        for path in self._temp_dir.iterdir():
            if path.is_file():
                # Reconstruct key from filename
                if self._serializer == "json" and path.suffix == ".json":
                    key = path.stem.replace("_", "/")
                    keys.append(key)
                elif self._serializer == "pickle" and path.suffix == ".pkl":
                    key = path.stem.replace("_", "/")
                    keys.append(key)
        return keys

    def clear(self) -> None:
        """Clear all data from the store."""
        for path in self._temp_dir.iterdir():
            if path.is_file():
                path.unlink()
        self._metadata.clear()

    def __del__(self):
        """Clean up temp directory on exit if needed."""
        if hasattr(self, "_cleanup_on_exit") and self._cleanup_on_exit:
            try:
                import shutil

                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Best effort cleanup
