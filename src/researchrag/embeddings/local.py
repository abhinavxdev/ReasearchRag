"""Lazy, CPU-local BGE embeddings with long-input window pooling."""

import threading
from typing import Any

import numpy as np


class SentenceEmbedder:
    """Keep model downloads and loading out of application startup."""

    def __init__(self, model: str) -> None:
        self.name = model
        self._model: Any = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Average normalized overlapping model windows instead of silently truncating."""
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device="cpu")
            model = self._model
            prefix = (
                "Represent this sentence for searching relevant passages: "
                if query and "bge" in self.name.lower()
                else ""
            )
            windows: list[str] = []
            ranges: list[tuple[int, int]] = []
            budget = max(32, int(model.max_seq_length) - 32)
            for text in texts:
                ids = model.tokenizer.encode(text, add_special_tokens=False)
                start = len(windows)
                for offset in range(0, max(1, len(ids)), budget - 32):
                    windows.append(
                        prefix
                        + model.tokenizer.decode(
                            ids[offset : offset + budget], skip_special_tokens=True
                        )
                    )
                    if offset + budget >= len(ids):
                        break
                ranges.append((start, len(windows)))
            vectors = model.encode(
                windows, normalize_embeddings=True, show_progress_bar=False, batch_size=16
            )
            result: list[list[float]] = []
            for start, end in ranges:
                vector = np.mean(vectors[start:end], axis=0)
                vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                result.append([float(v) for v in vector])
            return result
