from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RateLimiter:
    limit_per_minute: int
    state_path: Path

    def __post_init__(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, List[float]]:
        if not self.state_path.exists():
            return {"timestamps": []}
        with self.state_path.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"timestamps": []}

    def _save(self, payload: Dict[str, List[float]]) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def acquire(self, label: str = "run") -> None:
        now = time.time()
        data = self._load()
        ts = [t for t in data.get("timestamps", []) if now - t < 60.0]
        while len(ts) >= self.limit_per_minute:
            oldest = min(ts)
            wait_s = max(0.01, 60.0 - (now - oldest) + 0.01)
            time.sleep(wait_s)
            now = time.time()
            ts = [t for t in ts if now - t < 60.0]
        ts.append(now)
        data["timestamps"] = ts
        data["last_label"] = label
        data["last_rate"] = len(ts)
        self._save(data)

    def current_rate(self) -> int:
        now = time.time()
        data = self._load()
        ts = [t for t in data.get("timestamps", []) if now - t < 60.0]
        return len(ts)
