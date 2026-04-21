from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AlphaScheduler:
    schedule: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.schedule:
            self.schedule = [(0.0, 1.0), (1.0, 1.0)]
        self.schedule = sorted(self.schedule, key=lambda x: x[0])

    def alpha_at(self, progress: float) -> float:
        p = max(0.0, min(1.0, progress))
        if p <= self.schedule[0][0]:
            return float(self.schedule[0][1])
        if p >= self.schedule[-1][0]:
            return float(self.schedule[-1][1])
        for i in range(1, len(self.schedule)):
            p0, a0 = self.schedule[i - 1]
            p1, a1 = self.schedule[i]
            if p0 <= p <= p1:
                if p1 - p0 < 1e-12:
                    return float(a1)
                r = (p - p0) / (p1 - p0)
                return float(a0 + r * (a1 - a0))
        return float(self.schedule[-1][1])
