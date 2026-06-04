# ============================================================
# 评分指标实现
# ============================================================
# 🔧 MODIFIABLE — 选手可以扩展这些评分函数，添加自定义权重或指标。
# 但核心指标的计算公式应保持不变以确保与评测系统一致。
#
# 包含的指标:
#   任务一: disposition_effect, spearman_belief_action, wasserstein_1d, daily_turnover
#   任务二: f1_score, response_latency_scores
# ============================================================

"""Scoring metrics for the market simulation challenge."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def disposition_effect(trades: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Compute PGR, PLR, and DE from sell decisions.

    Expected fields: action/side, price, avg_cost or cost_basis.  Optional
    ``possible_gain``/``possible_loss`` booleans can be supplied for held lots.
    """

    realized_gains = realized_losses = 0
    possible_gains = possible_losses = 0
    for trade in trades:
        price = float(trade.get("price", trade.get("executed_price", 0.0)))
        cost = float(trade.get("avg_cost", trade.get("cost_basis", price)))
        action = str(trade.get("action", trade.get("side", trade.get("direction", "")))).lower()
        is_gain = price >= cost
        if is_gain:
            possible_gains += 1
        else:
            possible_losses += 1
        if action == "sell":
            if is_gain:
                realized_gains += 1
            else:
                realized_losses += 1
        if trade.get("possible_gain"):
            possible_gains += 1
        if trade.get("possible_loss"):
            possible_losses += 1

    pgr = realized_gains / possible_gains if possible_gains else 0.0
    plr = realized_losses / possible_losses if possible_losses else 0.0
    return {"PGR": pgr, "PLR": plr, "DE": pgr - plr}


def spearman_belief_action(sentiments: Sequence[float], actions: Sequence[float]) -> float:
    if len(sentiments) != len(actions) or len(sentiments) < 2:
        return 0.0
    return pearson(_ranks(sentiments), _ranks(actions))


def wasserstein_1d(sample: Sequence[float], baseline: Sequence[float]) -> float:
    """First Wasserstein distance for equally weighted 1-D samples."""

    if not sample or not baseline:
        return 0.0
    x = sorted(float(v) for v in sample)
    y = sorted(float(v) for v in baseline)
    n = max(len(x), len(y))
    total = 0.0
    for idx in range(n):
        q = (idx + 0.5) / n
        total += abs(_quantile_sorted(x, q) - _quantile_sorted(y, q))
    return total / n


def daily_turnover(transactions: Iterable[Mapping[str, Any]], equity_by_day: Mapping[str, float]) -> Dict[str, float]:
    amounts: Dict[str, float] = defaultdict(float)
    for item in transactions:
        day = str(item.get("date", item.get("timestamp", "")))[:10]
        price = float(item.get("price", item.get("executed_price", 0.0)))
        qty = float(item.get("quantity", item.get("executed_quantity", 0.0)))
        amounts[day] += abs(price * qty)
    return {
        day: amount / max(float(equity_by_day.get(day, 0.0)), 1e-9)
        for day, amount in amounts.items()
    }


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def response_latency_scores(attack_ticks: Mapping[str, int], alert_ticks: Mapping[str, int]) -> Dict[str, float]:
    scores = {}
    for attack_id, tick in attack_ticks.items():
        if attack_id not in alert_ticks:
            scores[attack_id] = 0.0
            continue
        delay = alert_ticks[attack_id] - tick
        if delay <= 3:
            scores[attack_id] = 1.0
        elif delay >= 10:
            scores[attack_id] = 0.0
        else:
            scores[attack_id] = (10 - delay) / 7.0
    return scores


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def _ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][0] == indexed[cursor][0]:
            end += 1
        rank = (cursor + end + 2) / 2.0
        for pos in range(cursor, end + 1):
            ranks[indexed[pos][1]] = rank
        cursor = end + 1
    return ranks


def _quantile_sorted(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    q = max(0.0, min(1.0, q))
    position = q * (len(values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight
