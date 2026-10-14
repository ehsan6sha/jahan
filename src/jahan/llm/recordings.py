"""Request-keyed recordings of every non-deterministic input (LLM calls, web fetches, human actions).

``request_key`` is the sha256 of canonical JSON describing the request.  In replay mode a missing key
fails closed (``MissingRecording``) instead of silently calling the network.  The manifest is an
append-only JSONL file; response bodies are stored as content-addressed JSON blobs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


class MissingRecording(RuntimeError):
    pass


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(request).encode("utf-8")).hexdigest()


class Recordings:
    def __init__(self, root: Path, replay_only: bool = False) -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.manifest = self.root / "manifest.jsonl"
        self.replay_only = replay_only
        self._index: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            self._index = {}
            if self.manifest.exists():
                for line in self.manifest.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._index[row["request_key"]] = row
        return self._index

    def get(self, request: dict[str, Any]) -> Any | None:
        key = request_key(request)
        row = self._load().get(key)
        if row is None:
            if self.replay_only:
                raise MissingRecording(
                    f"no recording for request {key[:12]} ({request.get('kind')}, "
                    f"{request.get('purpose')}); refusing to call the network in replay mode"
                )
            return None
        blob = self.blobs / f"{row['response_hash']}.json"
        return json.loads(blob.read_text(encoding="utf-8"))

    def put(self, request: dict[str, Any], response: Any, meta: dict[str, Any] | None = None) -> str:
        key = request_key(request)
        body = canonical(response)
        rh = hashlib.sha256(body.encode("utf-8")).hexdigest()
        self.blobs.mkdir(parents=True, exist_ok=True)
        blob = self.blobs / f"{rh}.json"
        if not blob.exists():
            blob.write_text(body, encoding="utf-8")
        row = {
            "request_key": key,
            "response_hash": rh,
            "kind": request.get("kind"),
            "purpose": request.get("purpose"),
            "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            **(meta or {}),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(canonical(row) + "\n")
        self._load()[key] = row
        return key
