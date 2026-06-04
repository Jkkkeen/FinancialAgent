"""Run a small end-to-end challenge demo.

Usage:
    python -m competition_solution.demo
"""

from __future__ import annotations

import json

from .exchange import ExchangeAgent
from .investment_agent import InvestmentAgent, Position
from .metrics import disposition_effect, f1_score, spearman_belief_action, wasserstein_1d


def run_single_agent_demo() -> dict:
    agent = InvestmentAgent("demo_agent", personality="aggressive", cash=50_000, seed=11)
    agent.positions["SIM"] = Position(quantity=300, avg_cost=104.0)
    klines = [
        {"open": 100 + idx * 0.4, "high": 101 + idx * 0.4, "low": 99 + idx * 0.4, "close": 100 + idx * 0.55, "volume": 10_000}
        for idx in range(30)
    ]
    agent.ingest_market("SIM", klines)
    agent.ingest_news("SIM", ["growth beat upgrade", "social users call a breakout buy"])
    agent.ingest_social("SIM", [{"text": "bull breakout buy", "influence": 12}])
    decision = agent.decide("SIM")
    return decision.as_order_dict()


def run_exchange_demo() -> dict:
    exchange = ExchangeAgent()
    accepted = exchange.submit_order("maker_1", "SIM", "sell", 101.0, 200, timestamp=1, entity_id="same_owner")
    wash = exchange.submit_order("maker_2", "SIM", "buy", 102.0, 100, timestamp=2, entity_id="same_owner")
    exchange.submit_order("bid_1", "SIM", "buy", 100.0, 100, timestamp=3)
    exchange.submit_order("bid_2", "SIM", "buy", 99.8, 120, timestamp=4)
    spoof = exchange.submit_order("spoof_1", "SIM", "sell", 130.0, 5000, timestamp=5)
    return {"accepted": accepted, "wash": wash, "spoof": spoof}


def run_metrics_demo() -> dict:
    de = disposition_effect(
        [
            {"side": "sell", "price": 110, "avg_cost": 100},
            {"side": "hold", "price": 90, "avg_cost": 100},
            {"side": "sell", "price": 112, "avg_cost": 100},
            {"side": "hold", "price": 88, "avg_cost": 100},
        ]
    )
    rho = spearman_belief_action([1, 0, -1, 1], [1, 0, -1, 1])
    wd = wasserstein_1d([0.12, 0.08, 0.05], [0.10, 0.07, 0.06])
    f1 = f1_score([1, 0, 1, 1], [1, 0, 0, 1])
    return {"disposition": de, "belief_action_rho": rho, "turnover_wd": wd, "f1": f1}


def main() -> None:
    payload = {
        "single_agent": run_single_agent_demo(),
        "exchange": run_exchange_demo(),
        "metrics": run_metrics_demo(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
