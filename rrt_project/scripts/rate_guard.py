#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rrt_project.benchmark.rate_limiter import RateLimiter


def main() -> None:
    p = argparse.ArgumentParser(description="Simple rate guard utility")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--label", type=str, default="manual")
    p.add_argument("--show-rate", action="store_true")
    args = p.parse_args()

    limiter = RateLimiter(limit_per_minute=args.limit, state_path=args.state)
    limiter.acquire(label=args.label)
    if args.show_rate:
        print(limiter.current_rate())


if __name__ == "__main__":
    main()
