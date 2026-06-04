#!/usr/bin/env python
"""Local evaluator for the anonymized 60-day market package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_solution.metrics import disposition_effect, spearman_belief_action, wasserstein_1d
from evaluate_submission import (
    ACTION_VALUE,
    BASELINE_TURNOVER,
    decision_checks,
    evaluate_task2,
    interface_smoke,
    score_belief_correlation,
    score_disposition,
    score_turnover,
)
from submission_interface.api import KLine, MarketObservation
from submission_interface.validator import load_submission


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Holding:
    quantity: int = 0
    avg_cost: float = 0.0


@dataclass
class Portfolio:
    cash: float
    holdings: Dict[str, Holding] = field(default_factory=dict)

    def value(self, prices: Mapping[str, float]) -> float:
        total = self.cash
        for symbol, holding in self.holdings.items():
            total += holding.quantity * prices.get(symbol, holding.avg_cost)
        return total


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_market_inputs(data_dir: Path) -> tuple[Dict[str, List[str]], List[str], List[Dict[str, Any]], Dict[str, str]]:
    days = [row["day"] for row in read_csv_dicts(data_dir / "calendar_60d.csv")]

    news_by_day: Dict[str, List[str]] = {}
    with (data_dir / "news_60d.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            news_by_day[row["day"]] = list(row.get("news", []))

    market_rows = []
    for row in read_csv_dicts(data_dir / "market_60d.csv"):
        market_rows.append(
            {
                "symbol": row["symbol"],
                "day": row["day"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "name": row.get("name", row["symbol"]),
            }
        )
    market_rows.sort(key=lambda item: (days.index(item["day"]), item["symbol"]))

    names = {
        row["stock_id"]: row.get("name", row["stock_id"])
        for row in read_csv_dicts(data_dir / "profile_60d.csv")
    }
    return news_by_day, days, market_rows, names


def execute_decision(
    portfolio: Portfolio,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    fee_rate: float,
) -> Optional[Dict[str, Any]]:
    holding = portfolio.holdings.get(symbol, Holding())
    quantity = int(max(0, quantity))
    if action == "buy" and quantity > 0:
        max_quantity = int(portfolio.cash / max(price * (1.0 + fee_rate), 1e-9))
        executed = min(quantity, max_quantity)
        if executed <= 0:
            return None
        notional = executed * price
        fee = notional * fee_rate
        avg_cost_before = holding.avg_cost or price
        total_qty = holding.quantity + executed
        holding.avg_cost = (holding.avg_cost * holding.quantity + notional + fee) / total_qty
        holding.quantity = total_qty
        portfolio.holdings[symbol] = holding
        portfolio.cash -= notional + fee
        return {
            "symbol": symbol,
            "side": "buy",
            "quantity": executed,
            "price": price,
            "notional": notional,
            "fee": fee,
            "avg_cost_before": avg_cost_before,
        }
    if action == "sell" and quantity > 0 and holding.quantity > 0:
        executed = min(quantity, holding.quantity)
        notional = executed * price
        fee = notional * fee_rate
        avg_cost_before = holding.avg_cost
        holding.quantity -= executed
        portfolio.cash += notional - fee
        if holding.quantity <= 0:
            portfolio.holdings.pop(symbol, None)
        else:
            portfolio.holdings[symbol] = holding
        return {
            "symbol": symbol,
            "side": "sell",
            "quantity": executed,
            "price": price,
            "notional": notional,
            "fee": fee,
            "avg_cost_before": avg_cost_before,
        }
    return None


def sign(value: float) -> int:
    if value > 0.05:
        return 1
    if value < -0.05:
        return -1
    return 0


def evaluate_task1(
    submission_dir: str,
    *,
    data_dir: str,
    seed: int,
    train_days_limit: Optional[int],
    lookback: int,
    max_news: int,
    initial_cash: float,
    fee_rate: float,
    use_llm: bool,
    llm_config_path: str,
) -> Dict[str, Any]:
    news_by_day, days, market_rows, name_by_symbol = load_market_inputs(Path(data_dir))
    if train_days_limit is not None:
        selected_days = days[: max(1, min(len(days), train_days_limit))]
    else:
        selected_days = days
    selected_day_set = set(selected_days)

    symbols = sorted({row["symbol"] for row in market_rows})
    rows_by_day: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in market_rows:
        if row["day"] in selected_day_set:
            rows_by_day[row["day"]][row["symbol"]] = row

    submission_config = {
        "agent_count": len(symbols),
        "use_llm": use_llm,
        "llm_config_path": llm_config_path,
    }
    submission = load_submission(submission_dir, submission_config)
    submission.reset(seed=seed, config=submission_config)

    portfolio = Portfolio(cash=initial_cash)
    histories = {symbol: [] for symbol in symbols}
    prices: Dict[str, float] = {}

    all_checks: Dict[str, List[bool]] = {}
    sentiments: List[int] = []
    actions: List[int] = []
    de_samples: List[Dict[str, Any]] = []
    turnover: List[float] = []
    action_counts = {"buy": 0, "sell": 0, "hold": 0}
    symbol_counts = {symbol: 0 for symbol in symbols}
    symbol_actions: Dict[str, List[int]] = defaultdict(list)
    symbol_sentiments: Dict[str, List[int]] = defaultdict(list)
    symbol_de_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    symbol_turnover: Dict[str, List[float]] = defaultdict(list)
    examples: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    tick = 0

    for day in selected_days:
        day_rows = rows_by_day.get(day, {})
        for symbol, row in day_rows.items():
            prices[symbol] = row["close"]
            history = histories[symbol]
            history.append(
                KLine(
                    symbol=symbol,
                    timestamp=day,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
            )
            if len(history) > lookback:
                del history[:-lookback]

        for symbol in symbols:
            if symbol not in day_rows:
                continue
            history = histories[symbol]
            if len(history) < max(5, min(lookback, 20)):
                continue

            holding = portfolio.holdings.get(symbol, Holding())
            tick += 1
            observation = MarketObservation(
                agent_id=f"agent_{symbol}",
                tick=tick,
                symbol=symbol,
                klines=list(history),
                news=list(news_by_day.get(day, []))[:max_news],
                social_posts=[],
                cash=portfolio.cash,
                position=holding.quantity,
                avg_cost=holding.avg_cost,
                extra={
                    "day": day,
                    "stock_name": name_by_symbol.get(symbol, symbol),
                    "news_count": len(news_by_day.get(day, [])),
                },
            )
            decision = submission.decide(observation)
            checks = decision_checks(decision, observation)
            for name, passed in checks.items():
                all_checks.setdefault(name, []).append(passed)

            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            signal = decision.sentiment_class if decision.sentiment_class in {-1, 0, 1} else sign(decision.belief_score)
            if signal == 0:
                signal = sign(decision.belief_score)
            sentiments.append(signal)
            action_value = ACTION_VALUE.get(decision.action, 0)
            actions.append(action_value)
            symbol_sentiments[symbol].append(signal)
            symbol_actions[symbol].append(action_value)

            equity_before = max(portfolio.value(prices), 1.0)
            trade = execute_decision(
                portfolio,
                symbol,
                decision.action,
                decision.quantity,
                day_rows[symbol]["close"],
                fee_rate,
            )
            traded_notional = 0.0
            if trade:
                trades.append({"day": day, **trade})
                traded_notional = abs(trade["notional"])

            if observation.position > 0:
                de_sample = {
                    "action": "sell" if trade and trade["side"] == "sell" else "hold",
                    "price": observation.klines[-1].close,
                    "avg_cost": observation.avg_cost,
                }
                de_samples.append(de_sample)
                symbol_de_samples[symbol].append(de_sample)
            turnover_value = traded_notional / equity_before
            turnover.append(turnover_value)
            symbol_turnover[symbol].append(turnover_value)

            if len(examples) < 8:
                examples.append(
                    {
                        "day": day,
                        "symbol": symbol,
                        "stock_name": name_by_symbol.get(symbol, symbol),
                        "position": observation.position,
                        "avg_cost": round(observation.avg_cost, 4),
                        "news_count": observation.extra["news_count"],
                        "action": decision.action,
                        "quantity": decision.quantity,
                        "belief_score": decision.belief_score,
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

    robustness = strict_robustness_scores(
        action_counts=action_counts,
        total_actions=len(actions),
        symbol_de_samples=symbol_de_samples,
        symbol_sentiments=symbol_sentiments,
        symbol_actions=symbol_actions,
        symbol_turnover=symbol_turnover,
    )
    strict_component_scores = dict(component_scores)
    strict_component_scores.update(robustness["component_scores"])

    legacy_total = (
        component_scores["format"] * 0.20
        + component_scores["disposition_effect"] * 0.25
        + component_scores["belief_correlation"] * 0.35
        + component_scores["turnover_similarity"] * 0.20
    )
    strict_total = (
        component_scores["format"] * 0.10
        + component_scores["disposition_effect"] * 0.15
        + component_scores["belief_correlation"] * 0.20
        + component_scores["turnover_similarity"] * 0.10
        + robustness["component_scores"]["cross_symbol_stability"] * 0.35
        + robustness["component_scores"]["action_diversity"] * 0.10
    )

    return {
        "score": round(strict_total, 1),
        "legacy_score": round(legacy_total, 1),
        "component_scores": {key: round(value, 1) for key, value in component_scores.items()},
        "strict_component_scores": {key: round(value, 1) for key, value in strict_component_scores.items()},
        "strict_penalties": robustness["penalties"],
        "checks": {key: f"{sum(values)}/{len(values)}" for key, values in all_checks.items()},
        "metrics": {
            "DE": round(de["DE"], 4),
            "PGR": round(de["PGR"], 4),
            "PLR": round(de["PLR"], 4),
            "belief_action_spearman": round(rho, 4),
            "turnover_wasserstein": round(wd, 4),
            "avg_turnover": round(sum(turnover) / max(len(turnover), 1), 4),
        },
        "data": {
            "data_dir": str(Path(data_dir).resolve()),
            "train_start": selected_days[0] if selected_days else None,
            "train_end": selected_days[-1] if selected_days else None,
            "train_days": len(selected_days),
            "symbols": len(symbols),
            "observations": len(actions),
            "trades": len(trades),
        },
        "action_counts": action_counts,
        "symbol_counts": symbol_counts,
        "symbol_metrics": robustness["symbol_metrics"],
        "examples": examples,
    }


def strict_robustness_scores(
    *,
    action_counts: Mapping[str, int],
    total_actions: int,
    symbol_de_samples: Mapping[str, List[Dict[str, Any]]],
    symbol_sentiments: Mapping[str, List[int]],
    symbol_actions: Mapping[str, List[int]],
    symbol_turnover: Mapping[str, List[float]],
) -> Dict[str, Any]:
    symbol_metrics = {}
    de_scores = []
    rho_scores = []
    turnover_scores = []
    active_symbols = 0

    for symbol in sorted(symbol_actions):
        samples = symbol_de_samples.get(symbol, [])
        de_metric = disposition_effect(samples) if samples else {"DE": 0.0, "PGR": 0.0, "PLR": 0.0}
        rho = spearman_belief_action(symbol_sentiments.get(symbol, []), symbol_actions.get(symbol, []))
        wd = wasserstein_1d(symbol_turnover.get(symbol, []), BASELINE_TURNOVER)
        de_score = score_disposition(de_metric["DE"])
        rho_score = score_belief_correlation(rho)
        turnover_score = score_turnover(wd)
        if samples:
            de_scores.append(de_score)
        rho_scores.append(rho_score)
        turnover_scores.append(turnover_score)
        if any(value != 0 for value in symbol_actions.get(symbol, [])):
            active_symbols += 1
        symbol_metrics[symbol] = {
            "DE": round(de_metric["DE"], 4),
            "rho": round(rho, 4),
            "turnover_WD": round(wd, 4),
            "de_score": round(de_score, 1),
            "rho_score": round(rho_score, 1),
            "turnover_score": round(turnover_score, 1),
            "actions": {
                "buy": sum(1 for item in symbol_actions.get(symbol, []) if item == 1),
                "sell": sum(1 for item in symbol_actions.get(symbol, []) if item == -1),
                "hold": sum(1 for item in symbol_actions.get(symbol, []) if item == 0),
            },
        }

    stability_parts = [
        sum(de_scores) / len(de_scores) if de_scores else 0.0,
        sum(rho_scores) / len(rho_scores) if rho_scores else 0.0,
        sum(turnover_scores) / len(turnover_scores) if turnover_scores else 0.0,
    ]
    cross_symbol_stability = sum(stability_parts) / len(stability_parts)

    buy_rate = action_counts.get("buy", 0) / max(total_actions, 1)
    sell_rate = action_counts.get("sell", 0) / max(total_actions, 1)
    hold_rate = action_counts.get("hold", 0) / max(total_actions, 1)
    active_symbol_rate = active_symbols / max(len(symbol_actions), 1)

    action_diversity = 100.0
    penalties = {}
    if hold_rate > 0.90:
        penalties["too_many_holds"] = round((hold_rate - 0.90) / 0.10 * 35.0, 2)
    if buy_rate < 0.03:
        penalties["too_few_buys"] = round((0.03 - buy_rate) / 0.03 * 25.0, 2)
    if sell_rate < 0.03:
        penalties["too_few_sells"] = round((0.03 - sell_rate) / 0.03 * 25.0, 2)
    if active_symbol_rate < 0.70:
        penalties["too_few_active_symbols"] = round((0.70 - active_symbol_rate) / 0.70 * 25.0, 2)
    action_diversity = max(0.0, action_diversity - sum(penalties.values()))

    return {
        "component_scores": {
            "cross_symbol_stability": cross_symbol_stability,
            "action_diversity": action_diversity,
        },
        "penalties": penalties,
        "symbol_metrics": symbol_metrics,
    }


def print_report(report: Mapping[str, Any]) -> None:
    print()
    print("=" * 72)
    print("  60天匿名市场行为测评报告（本地参考）")
    print(f"  提交目录: {report['submission_dir']}")
    print("=" * 72)

    smoke = report["interface_smoke"]
    print(f"\n[接口冒烟] {'OK' if smoke['status'] == 'ok' else 'FAIL'}  status={smoke['status']}")

    task1 = report["task1"]
    data = task1["data"]
    print(
        f"\n[任务一] 散户行为: {task1['score']:.1f} / 100 "
        f"(legacy={task1['legacy_score']:.1f}) "
        f"({data['train_start']} -> {data['train_end']}, "
        f"{data['observations']} observations, {data['trades']} trades)"
    )
    for name, value in task1["strict_component_scores"].items():
        print(f"  - {name}: {value:.1f}")
    if task1["strict_penalties"]:
        print(f"  - penalties: {task1['strict_penalties']}")
    metrics1 = task1["metrics"]
    print(
        "  - metrics: "
        f"DE={metrics1['DE']:.4f}, rho={metrics1['belief_action_spearman']:.4f}, "
        f"turnover_WD={metrics1['turnover_wasserstein']:.4f}"
    )
    print(f"  - actions: {task1['action_counts']}")

    task2 = report["task2"]
    print(f"\n[任务二] 原固定撮合/监管: {task2['score']:.1f} / 100")
    for name, value in task2["component_scores"].items():
        print(f"  - {name}: {value:.1f}")

    print()
    print("=" * 72)
    print(f"  综合参考分: {report['overall_score']:.1f} / 100")
    print("=" * 72)
    print("  注：任务一使用60天匿名市场数据；任务二沿用原固定场景。")
    print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="60-day anonymized market evaluator")
    parser.add_argument("submission_dir", help="team directory containing submission.py")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--max-news", type=int, default=20)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0003)
    parser.add_argument("--use-llm", action="store_true", help="enable LLM decisions for task 1")
    parser.add_argument("--llm-config-path", default="config.yaml", help="path passed to create_llm_client")
    parser.add_argument("--json", dest="json_path", help="optional path for machine-readable report")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    submission_dir = str(Path(args.submission_dir).resolve())
    report = {
        "submission_dir": submission_dir,
        "interface_smoke": interface_smoke(submission_dir, args.seed),
        "task1": evaluate_task1(
            submission_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            train_days_limit=args.train_days,
            lookback=args.lookback,
            max_news=args.max_news,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            use_llm=args.use_llm,
            llm_config_path=args.llm_config_path,
        ),
        "task2": evaluate_task2(submission_dir, args.seed),
    }
    report["overall_score"] = round(report["task1"]["score"] * 0.50 + report["task2"]["score"] * 0.50, 1)
    print_report(report)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report written to: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
