#!/usr/bin/env python
"""Dedicated local evaluator for preliminary submissions.

Usage:
    python evaluate_submission.py path/to/team_dir
    python evaluate_submission.py path/to/team_dir --json report.json

This script is stricter and more scenario-driven than self_score.py.  It is
still a local approximation: official hidden tests may use different data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from competition_solution.metrics import (
    disposition_effect,
    f1_score,
    response_latency_scores,
    spearman_belief_action,
    wasserstein_1d,
)
from submission_interface.api import AgentDecision, KLine, MarketObservation, OrderRequest
from submission_interface.validator import load_submission


ACTION_VALUE = {"sell": -1, "hold": 0, "buy": 1}
BASELINE_TURNOVER = [0.08, 0.10, 0.12, 0.07, 0.09, 0.11]


def sign(value: float) -> int:
    if value > 0.05:
        return 1
    if value < -0.05:
        return -1
    return 0


def make_klines(symbol: str, closes: List[float]) -> List[KLine]:
    klines: List[KLine] = []
    for idx, close in enumerate(closes, start=1):
        prev = closes[idx - 2] if idx > 1 else close
        high = max(prev, close) * 1.01
        low = min(prev, close) * 0.99
        klines.append(
            KLine(
                symbol=symbol,
                timestamp=f"2023-02-{idx:02d}",
                open=prev,
                high=round(high, 4),
                low=round(low, 4),
                close=close,
                volume=10_000 + idx * 100,
            )
        )
    return klines


def decision_checks(decision: AgentDecision, observation: MarketObservation) -> Dict[str, bool]:
    equity = observation.cash + observation.position * observation.klines[-1].close
    notional = decision.limit_price * max(decision.quantity, 0)
    return {
        "returns_agent_decision": isinstance(decision, AgentDecision),
        "action_valid": decision.action in ACTION_VALUE,
        "thought_present": len(str(decision.thought).strip()) >= 10,
        "belief_in_range": -1.0 <= float(decision.belief_score) <= 1.0,
        "sentiment_valid": decision.sentiment_class in {-1, 0, 1},
        "limit_price_positive": decision.limit_price > 0,
        "quantity_non_negative": decision.quantity >= 0,
        "sell_not_over_position": decision.action != "sell" or decision.quantity <= observation.position,
        "buy_not_far_over_cash": decision.action != "buy" or notional <= max(observation.cash, equity) * 1.20,
    }


def score_disposition(de: float) -> float:
    if 0.05 <= de <= 0.15:
        return 100.0
    if de <= 0:
        return 0.0
    if de < 0.05:
        return de / 0.05 * 100.0
    if de <= 0.30:
        return 100.0 - (de - 0.15) / 0.15 * 70.0
    return 20.0


def score_belief_correlation(rho: float) -> float:
    if rho >= 0.80:
        return 100.0
    if rho <= 0:
        return 0.0
    return rho / 0.80 * 100.0


def score_turnover(wd: float) -> float:
    if wd <= 0.04:
        return 100.0
    if wd >= 0.30:
        return 0.0
    return (0.30 - wd) / 0.26 * 100.0


def build_task1_observations() -> List[MarketObservation]:
    scenarios = [
        (
            "A_gain_bull",
            [100, 101, 103, 105, 108, 110, 112],
            ["earnings upgrade, strong growth, breakout buying"],
            [{"text": "bullish breakout buy", "influence": 8}],
            80_000,
            200,
            96.0,
        ),
        (
            "A_loss_bear",
            [112, 110, 107, 103, 99, 95, 92],
            ["lawsuit risk, downgrade, liquidity stress"],
            [{"text": "panic sell, avoid", "influence": 9}],
            80_000,
            200,
            108.0,
        ),
        (
            "A_cash_bull",
            [95, 96, 98, 100, 103, 106, 109],
            ["new product launch, revenue beat, analyst upgrade"],
            [{"text": "accumulate, upside surprise", "influence": 6}],
            120_000,
            0,
            0.0,
        ),
        (
            "A_cash_bear",
            [108, 106, 105, 102, 99, 97, 94],
            ["fraud rumor, forced selling, demand shock"],
            [{"text": "stay away, heavy selling", "influence": 7}],
            120_000,
            0,
            0.0,
        ),
        (
            "A_neutral",
            [100, 101, 100, 101, 100, 101, 100],
            ["mixed guidance, no clear catalyst"],
            [{"text": "wait and see", "influence": 3}],
            90_000,
            120,
            100.0,
        ),
        (
            "A_gain_pressure",
            [100, 103, 106, 109, 113, 116, 118],
            ["strong rally but valuation looks stretched"],
            [{"text": "take profit soon", "influence": 6}],
            70_000,
            300,
            91.0,
        ),
    ]
    observations: List[MarketObservation] = []
    for tick, (agent_id, closes, news, social, cash, position, avg_cost) in enumerate(scenarios, start=1):
        observations.append(
            MarketObservation(
                agent_id=agent_id,
                tick=tick,
                symbol="SIM",
                klines=make_klines("SIM", closes),
                news=news,
                social_posts=social,
                cash=cash,
                position=position,
                avg_cost=avg_cost,
            )
        )
    return observations


def evaluate_task1(submission_dir: str, seed: int) -> Dict[str, Any]:
    submission = load_submission(submission_dir, {"agent_count": 12})
    submission.reset(seed=seed, config={"agent_count": 12})

    decisions: List[Dict[str, Any]] = []
    all_checks: Dict[str, List[bool]] = {}
    sentiments: List[int] = []
    actions: List[int] = []
    de_samples: List[Dict[str, Any]] = []
    turnover: List[float] = []

    for observation in build_task1_observations():
        decision = submission.decide(observation)
        checks = decision_checks(decision, observation)
        for name, passed in checks.items():
            all_checks.setdefault(name, []).append(passed)

        last_price = observation.klines[-1].close
        action_value = ACTION_VALUE.get(decision.action, 0)
        signal = decision.sentiment_class if decision.sentiment_class in {-1, 0, 1} else sign(decision.belief_score)
        if signal == 0:
            signal = sign(decision.belief_score)
        sentiments.append(signal)
        actions.append(action_value)

        de_samples.append(
            {
                "action": decision.action,
                "price": last_price,
                "avg_cost": observation.avg_cost or last_price,
            }
        )
        equity = max(observation.cash + observation.position * last_price, 1.0)
        traded_notional = decision.limit_price * decision.quantity if decision.action != "hold" else 0.0
        turnover.append(max(0.0, traded_notional / equity))
        decisions.append(
            {
                "agent_id": decision.agent_id,
                "action": decision.action,
                "quantity": decision.quantity,
                "limit_price": round(decision.limit_price, 4),
                "belief_score": round(float(decision.belief_score), 4),
                "sentiment_class": decision.sentiment_class,
                "thought": decision.thought,
            }
        )

    flat_checks = [passed for values in all_checks.values() for passed in values]
    format_score = sum(1 for passed in flat_checks if passed) / max(len(flat_checks), 1) * 100.0
    de = disposition_effect(de_samples)
    rho = spearman_belief_action(sentiments, actions)
    wd = wasserstein_1d(turnover, BASELINE_TURNOVER)

    component_scores = {
        "format": format_score,
        "disposition_effect": score_disposition(de["DE"]),
        "belief_correlation": score_belief_correlation(rho),
        "turnover_similarity": score_turnover(wd),
    }
    total = (
        component_scores["format"] * 0.20
        + component_scores["disposition_effect"] * 0.25
        + component_scores["belief_correlation"] * 0.35
        + component_scores["turnover_similarity"] * 0.20
    )

    return {
        "score": round(total, 1),
        "component_scores": {k: round(v, 1) for k, v in component_scores.items()},
        "checks": {k: f"{sum(v)}/{len(v)}" for k, v in all_checks.items()},
        "metrics": {
            "DE": round(de["DE"], 4),
            "PGR": round(de["PGR"], 4),
            "PLR": round(de["PLR"], 4),
            "belief_action_spearman": round(rho, 4),
            "turnover_wasserstein": round(wd, 4),
            "avg_turnover": round(sum(turnover) / len(turnover), 4),
        },
        "decisions": decisions,
    }


def has_alert(result: Any, keywords: Tuple[str, ...]) -> bool:
    for alert in getattr(result, "alerts", []):
        alert_type = str(getattr(alert, "alert_type", "")).lower()
        if any(keyword in alert_type for keyword in keywords):
            return True
    return False


def first_alert_tick(result: Any, keywords: Tuple[str, ...]) -> Optional[int]:
    ticks: List[int] = []
    for alert in getattr(result, "alerts", []):
        alert_type = str(getattr(alert, "alert_type", "")).lower()
        if any(keyword in alert_type for keyword in keywords):
            ticks.append(int(getattr(alert, "timestamp", 0)))
    return min(ticks) if ticks else None


def new_submission(submission_dir: str, seed: int) -> Any:
    submission = load_submission(submission_dir, {"agent_count": 12})
    submission.reset(seed=seed, config={"agent_count": 12})
    return submission


def evaluate_price_time(submission_dir: str, seed: int) -> Tuple[float, Dict[str, Any]]:
    submission = new_submission(submission_dir, seed)
    orders = [
        OrderRequest("S1", "seller1", "SIM", "sell", 10.0, 100, 1, "E1"),
        OrderRequest("S2", "seller2", "SIM", "sell", 10.0, 100, 2, "E2"),
        OrderRequest("B1", "buyer", "SIM", "buy", 10.0, 150, 3, "E3"),
    ]
    result = submission.match_orders(orders, {"SIM": 10.0}, tick=3)
    trades = getattr(result, "trades", [])
    checks = {
        "two_trades": len(trades) == 2,
        "first_seller_priority": len(trades) >= 1 and trades[0].seller_id == "seller1",
        "second_seller_priority": len(trades) >= 2 and trades[1].seller_id == "seller2",
        "partial_fill_quantity": len(trades) >= 2 and trades[0].quantity == 100 and trades[1].quantity == 50,
        "no_false_alert": len(getattr(result, "alerts", [])) == 0,
    }
    score = sum(1 for passed in checks.values() if passed) / len(checks) * 100.0
    return score, {"checks": checks, "trade_count": len(trades), "alert_count": len(getattr(result, "alerts", []))}


def evaluate_wash(submission_dir: str, seed: int) -> Tuple[float, Dict[str, Any]]:
    submission = new_submission(submission_dir, seed)
    submission.match_orders(
        [OrderRequest("W1", "acct_a", "SIM", "sell", 10.0, 100, 1, "SAME_OWNER")],
        {"SIM": 10.0},
        tick=1,
    )
    result = submission.match_orders(
        [OrderRequest("W2", "acct_b", "SIM", "buy", 10.0, 100, 2, "SAME_OWNER")],
        {"SIM": 10.0},
        tick=2,
    )
    detected = has_alert(result, ("wash",))
    alert_tick = first_alert_tick(result, ("wash",))
    latency = None if alert_tick is None else alert_tick - 2
    score = 100.0 if detected and latency is not None and latency <= 3 else 0.0
    return score, {"detected": detected, "latency": latency, "alert_count": len(getattr(result, "alerts", []))}


def evaluate_spoof(submission_dir: str, seed: int) -> Tuple[float, Dict[str, Any]]:
    submission = new_submission(submission_dir, seed)
    submission.match_orders(
        [
            OrderRequest("D1", "maker1", "SIM", "sell", 100.0, 50, 1, "M1"),
            OrderRequest("D2", "maker2", "SIM", "sell", 101.0, 50, 1, "M2"),
        ],
        {"SIM": 100.0},
        tick=1,
    )
    result = submission.match_orders(
        [OrderRequest("SP1", "spoofer", "SIM", "buy", 80.0, 1_000, 2, "SPOOF")],
        {"SIM": 100.0},
        tick=2,
    )
    detected = has_alert(result, ("spoof", "layer"))
    alert_tick = first_alert_tick(result, ("spoof", "layer"))
    latency = None if alert_tick is None else alert_tick - 2
    score = 100.0 if detected and latency is not None and latency <= 3 else 0.0
    return score, {"detected": detected, "latency": latency, "alert_count": len(getattr(result, "alerts", []))}


def evaluate_pump(submission_dir: str, seed: int) -> Tuple[float, Dict[str, Any]]:
    submission = new_submission(submission_dir, seed)
    detected = False
    latency: Optional[int] = None
    alert_count = 0
    for idx in range(8):
        tick = idx + 1
        result = submission.match_orders(
            [
                OrderRequest(f"P{idx}S", "dump_seller", "SIM", "sell", 100.0 + idx, 10, tick, "DUMP"),
                OrderRequest(f"P{idx}B", f"buyer_{idx % 4}", "SIM", "buy", 150.0, 10, tick, f"B{idx % 4}"),
            ],
            {"SIM": 100.0 + idx},
            tick=tick,
        )
        alert_count += len(getattr(result, "alerts", []))
        if has_alert(result, ("pump", "dump")):
            detected = True
            alert_tick = first_alert_tick(result, ("pump", "dump"))
            latency = None if alert_tick is None else alert_tick - tick
            break
    score = 100.0 if detected and latency is not None and latency <= 3 else 0.0
    return score, {"detected": detected, "latency": latency, "alert_count": alert_count}


def evaluate_task2(submission_dir: str, seed: int) -> Dict[str, Any]:
    price_time_score, price_time = evaluate_price_time(submission_dir, seed)
    wash_score, wash = evaluate_wash(submission_dir, seed + 1)
    spoof_score, spoof = evaluate_spoof(submission_dir, seed + 2)
    pump_score, pump = evaluate_pump(submission_dir, seed + 3)

    y_true = [1, 1, 1]
    y_pred = [int(wash["detected"]), int(spoof["detected"]), int(pump["detected"])]
    f1 = f1_score(y_true, y_pred)
    latency_scores = response_latency_scores(
        {"wash": 2, "spoof": 2, "pump": 8},
        {
            name: tick
            for name, tick in {
                "wash": None if wash["latency"] is None else 2 + wash["latency"],
                "spoof": None if spoof["latency"] is None else 2 + spoof["latency"],
                "pump": None if pump["latency"] is None else 8 + pump["latency"],
            }.items()
            if tick is not None
        },
    )
    avg_latency_score = sum(latency_scores.values()) / len(latency_scores) * 100.0
    surveillance_score = f1["f1"] * 70.0 + avg_latency_score * 0.30
    total = price_time_score * 0.30 + surveillance_score * 0.70

    return {
        "score": round(total, 1),
        "component_scores": {
            "price_time_priority": round(price_time_score, 1),
            "surveillance": round(surveillance_score, 1),
            "f1": round(f1["f1"] * 100.0, 1),
            "response_latency": round(avg_latency_score, 1),
        },
        "metrics": {
            "precision": round(f1["precision"], 4),
            "recall": round(f1["recall"], 4),
            "f1": round(f1["f1"], 4),
            "latency_scores": {k: round(v, 4) for k, v in latency_scores.items()},
        },
        "scenarios": {
            "price_time": price_time,
            "wash_trading": wash,
            "spoofing": spoof,
            "pump_and_dump": pump,
        },
    }


def interface_smoke(submission_dir: str, seed: int) -> Dict[str, Any]:
    try:
        submission = load_submission(submission_dir, {"agent_count": 8})
        submission.reset(seed=seed, config={"agent_count": 8})
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def print_report(report: Mapping[str, Any]) -> None:
    print()
    print("=" * 72)
    print("  AI Agent 金融市场模拟大赛 专项评测报告（本地参考）")
    print(f"  提交目录: {report['submission_dir']}")
    print("=" * 72)
    print()

    smoke = report["interface_smoke"]
    smoke_icon = "OK" if smoke["status"] == "ok" else "FAIL"
    print(f"[接口冒烟] {smoke_icon}  status={smoke['status']}")

    task1 = report["task1"]
    print()
    print(f"[任务一] 投资 Agent 行为拟人性: {task1['score']:.1f} / 100")
    for name, value in task1["component_scores"].items():
        print(f"  - {name}: {value:.1f}")
    metrics1 = task1["metrics"]
    print(
        "  - metrics: "
        f"DE={metrics1['DE']:.4f}, rho={metrics1['belief_action_spearman']:.4f}, "
        f"turnover_WD={metrics1['turnover_wasserstein']:.4f}"
    )

    task2 = report["task2"]
    print()
    print(f"[任务二] 交易所撮合与监管: {task2['score']:.1f} / 100")
    for name, value in task2["component_scores"].items():
        print(f"  - {name}: {value:.1f}")
    metrics2 = task2["metrics"]
    print(
        "  - metrics: "
        f"precision={metrics2['precision']:.4f}, recall={metrics2['recall']:.4f}, "
        f"f1={metrics2['f1']:.4f}"
    )
    for name, data in task2["scenarios"].items():
        print(f"  - {name}: {data}")

    overall = report["overall_score"]
    print()
    print("=" * 72)
    print(f"  综合专项参考分: {overall:.1f} / 100")
    print("=" * 72)
    print("  注：本脚本使用固定本地场景，不代表官方隐藏测试最终得分。")
    print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TwinMarket preliminary-stage local evaluator")
    parser.add_argument("submission_dir", help="team directory containing submission.py")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for deterministic checks")
    parser.add_argument("--json", dest="json_path", help="optional path for machine-readable report")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    submission_dir = str(Path(args.submission_dir).resolve())

    report = {
        "submission_dir": submission_dir,
        "interface_smoke": interface_smoke(submission_dir, args.seed),
        "task1": evaluate_task1(submission_dir, args.seed),
        "task2": evaluate_task2(submission_dir, args.seed),
    }
    report["overall_score"] = round(report["task1"]["score"] * 0.50 + report["task2"]["score"] * 0.50, 1)

    print_report(report)
    if args.json_path:
        output_path = Path(args.json_path)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
