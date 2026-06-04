"""Local validator for formal competition submissions.

Usage:
    python -m submission_interface.validator path/to/team_dir
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .api import (
    AgentDecision,
    CompetitionSubmission,
    KLine,
    MarketObservation,
    MatchResult,
    OrderRequest,
)


def load_submission(submission_dir: str, config: Mapping[str, Any] | None = None) -> CompetitionSubmission:
    path = Path(submission_dir).resolve() / "submission.py"
    if not path.exists():
        raise FileNotFoundError(f"missing submission.py: {path}")
    spec = importlib.util.spec_from_file_location("team_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["team_submission"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "create_submission"):
        raise AttributeError("submission.py must expose create_submission(config)")
    submission = module.create_submission(dict(config or {}))
    missing = [name for name in ("reset", "decide", "match_orders") if not callable(getattr(submission, name, None))]
    if missing:
        raise TypeError(f"submission object is missing methods: {missing}")
    return submission


def validate_submission(submission_dir: str) -> dict:
    submission = load_submission(submission_dir, {"agent_count": 12})
    submission.reset(seed=123, config={"agent_count": 12})

    klines = [
        KLine("SIM", f"2023-01-{day:02d}", 100 + day, 101 + day, 99 + day, 100 + day, 10_000)
        for day in range(1, 8)
    ]
    observation = MarketObservation(
        agent_id="A0001",
        tick=1,
        symbol="SIM",
        klines=klines,
        news=["growth upgrade breakout buy"],
        social_posts=[{"text": "bull buy", "influence": 5}],
        cash=100_000,
        position=200,
        avg_cost=101.0,
    )
    decision = submission.decide(observation)
    if not isinstance(decision, AgentDecision):
        raise TypeError("decide(...) must return AgentDecision")
    if decision.action not in {"buy", "sell", "hold"}:
        raise ValueError("decision.action must be buy/sell/hold")

    orders = [
        OrderRequest("O1", "seller", "SIM", "sell", 101.0, 100, 1, "E1"),
        OrderRequest("O2", "buyer", "SIM", "buy", 102.0, 100, 2, "E2"),
    ]
    match_result = submission.match_orders(orders, {"SIM": 100.0}, tick=2)
    if not isinstance(match_result, MatchResult):
        raise TypeError("match_orders(...) must return MatchResult")

    return {
        "status": "ok",
        "decision": decision.to_dict(),
        "match_result": match_result.to_dict(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m submission_interface.validator path/to/team_dir")
    result = validate_submission(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
