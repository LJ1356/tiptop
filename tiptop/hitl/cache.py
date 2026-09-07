"""On-disk cache for proposal queries, so iterating on prompts does not re-bill every run.

Modelled on ``prpl_llm_utils.cache.SQLite3PretrainedLargeModelCache`` and its ``Query.get_id()``:
the key is the model, the prompt, and a hash of the image that is robust to sensor noise, so two
captures of an unchanged scene hit the same entry. That library is not on PyPI and is synchronous,
and its image hashing needs ``imagehash`` (and PyWavelets under it), so the few lines that matter are
reproduced here instead of adding a dependency to the tiptop environment.

ONE DELIBERATE DIVERGENCE, and it is the whole reason this is not applied everywhere: only PROPOSAL
queries are cached. The reference caches every model call because its images are deterministic
renderings of a simulated state. Here they come from a ZED pointed at a table a human is about to
reach into, and a noise-robust hash is exactly the wrong key for "is the cloth folded?" -- two frames
either side of the fold can be close enough to collide, which would answer the post-teleop check
with the pre-teleop verdict. Grounding and verification always ask the model.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

# Edge length the image is reduced to before hashing. Small enough that sensor noise and a degree of
# lighting drift wash out, large enough that a moved object changes the hash.
_HASH_EDGE = 16


def image_fingerprint(image: Image.Image) -> str:
    """A hash of an image that survives sensor noise but not a change in the scene.

    Combines a difference hash (adjacent-pixel ordering, robust to brightness) with a coarse mean
    hash, the same pairing ``prpl_llm_utils`` uses, minus the wavelet and colour hashes that need an
    extra dependency to compute.
    """
    grey = np.asarray(image.convert("L").resize((_HASH_EDGE + 1, _HASH_EDGE), Image.Resampling.LANCZOS), dtype=np.int16)
    difference = (grey[:, 1:] > grey[:, :-1]).flatten()
    mean = (grey[:, :-1] > grey[:, :-1].mean()).flatten()
    bits = np.packbits(np.concatenate([difference, mean])).tobytes()
    return hashlib.sha256(bits).hexdigest()


class ProposalCache:
    """SQLite-backed cache of proposal responses, keyed on (model, prompt, image fingerprint)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, created REAL DEFAULT (unixepoch()))"
            )

    def _key(self, model: str, prompt: str, image: Image.Image | None) -> str:
        parts = [model, prompt, image_fingerprint(image) if image is not None else ""]
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    def get(self, model: str, prompt: str, image: Image.Image | None) -> str | None:
        """The cached response text, or None."""
        try:
            with sqlite3.connect(self._path) as conn:
                row = conn.execute(
                    "SELECT response FROM responses WHERE key = ?", (self._key(model, prompt, image),)
                ).fetchone()
        except sqlite3.Error as exc:  # a cache is never worth failing a run over
            _log.warning(f"HITL proposal cache read failed: {exc}")
            return None
        return row[0] if row else None

    def put(self, model: str, prompt: str, image: Image.Image | None, response: str) -> None:
        """Record a response. Only ever called with one that parsed and validated."""
        try:
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO responses (key, model, prompt, response) VALUES (?, ?, ?, ?)",
                    (self._key(model, prompt, image), model, prompt, response),
                )
        except sqlite3.Error as exc:
            _log.warning(f"HITL proposal cache write failed: {exc}")
