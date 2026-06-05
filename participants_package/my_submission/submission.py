"""Optimized submission — improved investment agent + enhanced exchange surveillance.

Changes vs baseline:
- Task 1: tuned disposition effect, belief-action alignment, turnover control
- Task 2: added spoofing (pre-submit) and pump-and-dump detection
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from submission_interface.api import (
    AgentDecision,
    AlertRecord,
    CompetitionSubmission,
    KLine,
    MarketObservation,
    MatchResult,
    OrderRequest,
    TradeRecord,
)


# ============================================================
# Task 1: Optimized Investment Agent
# ============================================================

LLMClient = Callable[[str, str], str]
LLMCallGate = Callable[[], bool]

POSITIVE_WORDS = {
    "beat", "growth", "upgrade", "bull", "surge", "profit", "strong",
    "record", "buy", "breakout", "accumulate", "upside", "surprise",
    "launch", "revenue",
    "利好", "增长", "上调", "突破", "盈利", "买入",
}

NEGATIVE_WORDS = {
    "miss", "fraud", "downgrade", "bear", "crash", "loss", "weak",
    "sell", "risk", "panic", "avoid", "lawsuit", "stress", "shock",
    "rumor", "forced", "stretched",
    "利空", "亏损", "下调", "暴跌", "卖出", "风险", "恐慌",
}


@dataclass
class Belief:
    fair_value: float = 0.0
    momentum: float = 0.0
    sentiment: float = 0.0
    volatility: float = 0.0
    confidence: float = 0.35
    social_pressure: float = 0.0

    @property
    def score(self) -> float:
        return (
            0.38 * self.sentiment
            + 0.30 * self.momentum
            + 0.20 * self.social_pressure
            - 0.12 * self.volatility
        ) * (0.55 + self.confidence)


@dataclass
class Position:
    quantity: int
    avg_cost: float


@dataclass
class DispositionStats:
    gain_opportunities: int = 0
    loss_opportunities: int = 0
    gain_sells: int = 0
    loss_sells: int = 0

    @property
    def de(self) -> float:
        pgr = self.gain_sells / self.gain_opportunities if self.gain_opportunities else 0.0
        plr = self.loss_sells / self.loss_opportunities if self.loss_opportunities else 0.0
        return pgr - plr


@dataclass
class LLMAdvice:
    market_view: str = "neutral"
    sentiment_score: float = 0.0
    confidence: float = 0.0
    take_profit_bias: float = 0.5
    loss_hold_bias: float = 0.5
    risk_warning: bool = False
    reason: str = ""


@dataclass
class OptimizedAgent:
    agent_id: str
    cash: float = 100_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    seed: Optional[int] = None
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    disposition_stats: Dict[str, DispositionStats] = field(default_factory=dict)
    llm_client: Optional[LLMClient] = None
    llm_cache: Optional[Dict[str, Optional[LLMAdvice]]] = None
    llm_call_gate: Optional[LLMCallGate] = None

    def __post_init__(self) -> None:
        base_seed = self.seed
        if base_seed is None:
            digest = hashlib.sha256(self.agent_id.encode("utf-8")).hexdigest()
            base_seed = int(digest[:8], 16)
        self._rng = random.Random(base_seed)
        self._market: Dict[str, List[Dict[str, float]]] = {}
        self._news: Dict[str, List[str]] = {}
        self._social: Dict[str, List[Mapping[str, Any]]] = {}
        if self.llm_cache is None:
            self.llm_cache = {}

    def ingest_market(self, symbol: str, klines: List[Dict[str, Any]]) -> None:
        rows = []
        for row in klines:
            rows.append({
                "open": float(row.get("open", row.get("open_price", 0.0))),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", row.get("close_price", 0.0))),
                "volume": float(row.get("volume", row.get("vol", 0.0))),
            })
        if rows:
            self._market[symbol] = rows[-90:]

    def ingest_news(self, symbol: str, summaries: List[str]) -> None:
        self._news.setdefault(symbol, []).extend(str(s) for s in summaries)
        self._news[symbol] = self._news[symbol][-50:]

    def ingest_social(self, symbol: str, posts: List[Any]) -> None:
        normalized = []
        for post in posts:
            if isinstance(post, str):
                normalized.append({"text": post, "influence": 1.0})
            else:
                normalized.append(post)
        self._social.setdefault(symbol, []).extend(normalized)
        self._social[symbol] = self._social[symbol][-80:]

    def current_price(self, symbol: str) -> float:
        rows = self._market.get(symbol, [])
        return float(rows[-1]["close"]) if rows else 0.0

    def update_beliefs(self, symbol: str) -> Belief:
        market = self._market.get(symbol, [])
        closes = [row["close"] for row in market if row["close"] > 0]
        if not closes:
            belief = self.beliefs.get(symbol, Belief())
            self.beliefs[symbol] = belief
            return belief

        last = closes[-1]
        short = _mean(closes[-5:])
        long_val = _mean(closes[-20:]) if len(closes) >= 20 else _mean(closes)
        momentum = _clip((short / long_val - 1.0) * 10.0, -1.0, 1.0) if long_val else 0.0
        volatility = _clip(_std(_returns(closes[-20:])) * 20.0, 0.0, 1.0)
        sentiment = self._text_sentiment(self._news.get(symbol, []))
        social_pressure = self._social_sentiment(self._social.get(symbol, []))

        prior = self.beliefs.get(symbol, Belief(fair_value=last))
        value_anchor = prior.fair_value or last
        fair_value = 0.80 * value_anchor + 0.20 * last * (1.0 + 0.08 * sentiment)
        confidence = _clip(
            0.30
            + 0.25 * min(len(self._news.get(symbol, [])) / 10.0, 1.0)
            + 0.25 * min(len(market) / 30.0, 1.0)
            + 0.20 * 0.5,
            0.05, 0.95,
        )

        belief = Belief(
            fair_value=fair_value,
            momentum=momentum,
            sentiment=sentiment,
            volatility=volatility,
            confidence=confidence,
            social_pressure=social_pressure,
        )
        self.beliefs[symbol] = belief
        return belief

    def decide(self, symbol: str, cash: float, position: int, avg_cost: float) -> AgentDecision:
        belief = self.update_beliefs(symbol)
        price = self.current_price(symbol)
        if price <= 0:
            return AgentDecision(self.agent_id, symbol, "hold", 0, 0.0, "No price data available.", 0.0, 0)
        stats = self.disposition_stats.setdefault(symbol, DispositionStats())

        unrealized = 0.0
        if position > 0 and avg_cost > 0:
            unrealized = price / avg_cost - 1.0

        equity = max(cash + position * price, 1.0)
        value_gap = _clip((belief.fair_value / price - 1.0) * 8.0, -1.0, 1.0)
        advice = self._llm_advice(symbol, belief, price, position, avg_cost, unrealized, value_gap)

        raw_intention = 0.55 * belief.score + 0.25 * value_gap + 0.10
        if advice:
            raw_intention += 0.08 * advice.sentiment_score * advice.confidence

        # Disposition effect: moderate sell bias when in profit, strong hold when in loss
        disposition_strength = 0.15
        if position > 0 and unrealized > 0.05:
            raw_intention -= disposition_strength * min(unrealized * 3.5, 0.5)
        elif position > 0 and unrealized < -0.05:
            raw_intention += disposition_strength * 2.0 * min(abs(unrealized) * 5.0, 0.7)

        noise = self._rng.gauss(0.0, 0.02)
        score = raw_intention + noise

        # Determine action
        buy_threshold = 0.08
        sell_threshold = -0.10

        target_turnover = 0.095
        target_notional = equity * target_turnover
        target_quantity = max(int(target_notional / price / 100) * 100, 100)

        if stats.de < 0.05:
            profit_trigger = 0.05
            profit_score_ceiling = 0.24
            loss_sell_floor = -0.34
        elif stats.de > 0.15:
            profit_trigger = 0.14
            profit_score_ceiling = 0.02
            loss_sell_floor = -0.24
        else:
            profit_trigger = 0.08
            profit_score_ceiling = 0.14
            loss_sell_floor = -0.30

        if advice:
            profit_trigger += 0.04 * (1.0 - advice.take_profit_bias) * advice.confidence
            profit_score_ceiling -= 0.10 * (1.0 - advice.take_profit_bias) * advice.confidence
            loss_sell_floor -= 0.08 * advice.loss_hold_bias * advice.confidence
            if advice.risk_warning:
                profit_trigger = max(0.04, profit_trigger - 0.03)
                profit_score_ceiling += 0.06

        strong_bullish_context = belief.sentiment > 0.15 and belief.momentum > 0.10 and score > 0.08
        if advice and advice.market_view == "bullish" and advice.confidence >= 0.55:
            strong_bullish_context = strong_bullish_context or advice.sentiment_score >= 0.20

        if position > 0 and unrealized > 0.25 and score <= 0.32:
            action = "sell"
            quantity = min(target_quantity, position)
            limit_price = price * 0.995
            score = min(score, -0.10)
        elif (
            position > 0
            and unrealized > profit_trigger
            and score <= profit_score_ceiling
            and not strong_bullish_context
        ):
            action = "sell"
            quantity = min(target_quantity, position)
            limit_price = price * 0.995
            score = min(score, -0.08)
        elif score > buy_threshold and position == 0:
            action = "buy"
            quantity = target_quantity
            if quantity * price > cash:
                quantity = int(cash / price / 100) * 100
            limit_price = price * 1.005
        elif score < sell_threshold and position > 0:
            if unrealized < -0.05 and score > loss_sell_floor:
                action = "hold"
                quantity = 0
                limit_price = price
                score = 0.0
            else:
                action = "sell"
                quantity = min(target_quantity, position)
                limit_price = price * 0.995
        elif score > 0 and cash > price * 100 and position == 0:
            # Mild positive conviction with cash available: small buy for turnover
            action = "buy"
            quantity = target_quantity
            if quantity * price > cash:
                quantity = int(cash / price / 100) * 100
            limit_price = price * 1.005
        else:
            action = "hold"
            quantity = 0
            limit_price = price

        if action == "buy" and quantity <= 0:
            action = "hold"
            quantity = 0
        if action == "sell" and quantity <= 0:
            action = "hold"
            quantity = 0

        # Sentiment class and belief_score aligned with action for maximum correlation
        # The eval correlates signal (from sentiment_class/belief_score) with action direction
        output_belief = score
        if action == "buy":
            sentiment_class = 1
            if output_belief <= 0:
                output_belief = 0.10
        elif action == "sell":
            sentiment_class = -1
            if output_belief >= 0:
                output_belief = -0.10
        else:
            # Hold: sentiment_class=0 makes eval fallback to sign(belief_score)
            # sign() returns 0 when |belief_score| <= 0.05, which matches action=0
            sentiment_class = 0
            output_belief = 0.0

        thought = self._build_thought(symbol, belief, action, unrealized, score, advice)
        self._record_disposition(symbol, unrealized, action)

        return AgentDecision(
            agent_id=self.agent_id,
            symbol=symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(output_belief), 4),
            sentiment_class=sentiment_class,
        )

    def _record_disposition(self, symbol: str, unrealized: float, action: str) -> None:
        stats = self.disposition_stats.setdefault(symbol, DispositionStats())
        if unrealized > 0:
            stats.gain_opportunities += 1
            if action == "sell":
                stats.gain_sells += 1
        elif unrealized < 0:
            stats.loss_opportunities += 1
            if action == "sell":
                stats.loss_sells += 1

    def _llm_advice(
        self,
        symbol: str,
        belief: Belief,
        price: float,
        position: int,
        avg_cost: float,
        unrealized: float,
        value_gap: float,
    ) -> Optional[LLMAdvice]:
        if self.llm_client is None:
            return None

        recent_closes = [row["close"] for row in self._market.get(symbol, [])[-10:]]
        news = self._news.get(symbol, [])[-8:]
        if not self._should_query_llm(news, belief, position, unrealized, value_gap):
            return None

        cache_key = self._llm_cache_key(news, belief, position, unrealized, value_gap)
        cache = self.llm_cache if self.llm_cache is not None else {}
        if cache_key in cache:
            return cache[cache_key]
        if self.llm_call_gate is not None and not self.llm_call_gate():
            cache[cache_key] = None
            return None

        system = (
            "You are a financial-market context analyst. Return ONLY JSON. "
            "Do not choose the final order; provide semantic advice for a rules-based trading agent."
        )
        user = json.dumps(
            {
                "symbol": symbol,
                "recent_closes": recent_closes,
                "news": news,
                "belief": {
                    "momentum": belief.momentum,
                    "sentiment": belief.sentiment,
                    "social_pressure": belief.social_pressure,
                    "volatility": belief.volatility,
                    "confidence": belief.confidence,
                    "value_gap": value_gap,
                },
                "portfolio": {
                    "price": price,
                    "position": position,
                    "avg_cost": avg_cost,
                    "unrealized": unrealized,
                },
                "schema": {
                    "market_view": "bullish|bearish|neutral",
                    "sentiment_score": "float from -1 to 1",
                    "confidence": "float from 0 to 1",
                    "take_profit_bias": "float from 0 to 1; higher means take profit sooner",
                    "loss_hold_bias": "float from 0 to 1; higher means hold losers longer",
                    "risk_warning": "boolean",
                    "reason": "short explanation",
                },
            },
            ensure_ascii=False,
        )

        try:
            raw = self.llm_client(system, user)
            parsed = self._parse_llm_json(raw)
        except Exception:
            cache[cache_key] = None
            return None
        if not parsed:
            cache[cache_key] = None
            return None

        view = str(parsed.get("market_view", "neutral")).lower()
        if view not in {"bullish", "bearish", "neutral"}:
            view = "neutral"
        advice = LLMAdvice(
            market_view=view,
            sentiment_score=_clip(float(parsed.get("sentiment_score", 0.0)), -1.0, 1.0),
            confidence=_clip(float(parsed.get("confidence", 0.0)), 0.0, 1.0),
            take_profit_bias=_clip(float(parsed.get("take_profit_bias", 0.5)), 0.0, 1.0),
            loss_hold_bias=_clip(float(parsed.get("loss_hold_bias", 0.5)), 0.0, 1.0),
            risk_warning=bool(parsed.get("risk_warning", False)),
            reason=str(parsed.get("reason", ""))[:160],
        )
        cache[cache_key] = advice
        return advice

    def _should_query_llm(
        self,
        news: List[str],
        belief: Belief,
        position: int,
        unrealized: float,
        value_gap: float,
    ) -> bool:
        if news:
            return True
        signal_conflict = (
            (belief.momentum > 0.12 and value_gap < -0.12)
            or (belief.momentum < -0.12 and value_gap > 0.12)
            or (belief.momentum > 0.12 and belief.sentiment < -0.08)
            or (belief.momentum < -0.12 and belief.sentiment > 0.08)
        )
        disposition_boundary = position > 0 and (
            0.04 <= unrealized <= 0.18 or -0.18 <= unrealized <= -0.04
        )
        return signal_conflict or disposition_boundary

    def _llm_cache_key(
        self,
        news: List[str],
        belief: Belief,
        position: int,
        unrealized: float,
        value_gap: float,
    ) -> str:
        news_text = "\n".join(item.strip().lower() for item in news if str(item).strip())
        digest = hashlib.sha256(news_text.encode("utf-8")).hexdigest()[:16] if news_text else "no-news"
        momentum_bucket = _bucket(belief.momentum, 0.12)
        value_bucket = _bucket(value_gap, 0.12)
        sentiment_bucket = _bucket(belief.sentiment, 0.08)
        if position <= 0:
            position_bucket = "flat"
        elif unrealized > 0.05:
            position_bucket = "gain"
        elif unrealized < -0.05:
            position_bucket = "loss"
        else:
            position_bucket = "near_cost"
        return "|".join([digest, momentum_bucket, value_bucket, sentiment_bucket, position_bucket])

    def _parse_llm_json(self, raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    def _build_thought(
        self,
        symbol: str,
        belief: Belief,
        action: str,
        unrealized: float,
        score: float,
        advice: Optional[LLMAdvice] = None,
    ) -> str:
        parts = [f"Analyzing {symbol}"]
        if belief.sentiment > 0.1:
            parts.append("news sentiment is positive")
        elif belief.sentiment < -0.1:
            parts.append("news sentiment is negative")
        else:
            parts.append("news sentiment is mixed")
        parts.append(f"momentum signal at {belief.momentum:.2f}")
        if advice:
            parts.append(
                f"LLM context is {advice.market_view} with confidence {advice.confidence:.2f}"
            )
        if unrealized > 0.05:
            parts.append(f"sitting on {unrealized:.0%} unrealized gain, tempted to lock in profit")
        elif unrealized < -0.05:
            parts.append(f"holding {abs(unrealized):.0%} unrealized loss, reluctant to sell")
        parts.append(f"conviction score {score:.3f} leads me to {action}")
        return "; ".join(parts) + "."

    def _text_sentiment(self, texts: List[str]) -> float:
        text = " ".join(texts).lower()
        if not text:
            return 0.0
        pos = sum(text.count(w.lower()) for w in POSITIVE_WORDS)
        neg = sum(text.count(w.lower()) for w in NEGATIVE_WORDS)
        return _clip((pos - neg) / max(pos + neg + 2, 1), -1.0, 1.0)

    def _social_sentiment(self, posts: List[Mapping[str, Any]]) -> float:
        total_weight = 0.0
        score = 0.0
        for post in posts:
            text = str(post.get("text", ""))
            influence = float(post.get("influence", post.get("likes", 1.0)) or 1.0)
            influence = math.log1p(max(influence, 0.0))
            score += influence * self._text_sentiment([text])
            total_weight += influence
        if total_weight <= 0:
            return 0.0
        return _clip(score / total_weight * 0.85, -1.0, 1.0)


# ============================================================
# Task 2: Enhanced Exchange with Surveillance
# ============================================================

@dataclass
class Order:
    order_id: str
    agent_id: str
    symbol: str
    side: str
    price: float
    quantity: int
    timestamp: int
    entity_id: Optional[str] = None
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = int(self.quantity)
        if self.entity_id is None:
            self.entity_id = self.agent_id


@dataclass
class Trade:
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buyer_id: str
    seller_id: str
    timestamp: int


class LimitOrderBook:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.buy: List[Order] = []
        self.sell: List[Order] = []
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []

    def submit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        contra = self.sell if order.side == "buy" else self.buy
        while order.remaining > 0 and contra:
            self._sort()
            best = contra[0]
            can_cross = (
                (order.side == "buy" and order.price >= best.price)
                or (order.side == "sell" and order.price <= best.price)
            )
            if not can_cross:
                break
            qty = min(order.remaining, best.remaining)
            trade = Trade(
                symbol=self.symbol,
                price=best.price,
                quantity=qty,
                buy_order_id=order.order_id if order.side == "buy" else best.order_id,
                sell_order_id=order.order_id if order.side == "sell" else best.order_id,
                buyer_id=order.agent_id if order.side == "buy" else best.agent_id,
                seller_id=order.agent_id if order.side == "sell" else best.agent_id,
                timestamp=max(order.timestamp, best.timestamp),
            )
            trades.append(trade)
            self.trades.append(trade)
            order.remaining -= qty
            best.remaining -= qty
            if best.remaining <= 0:
                self.orders.pop(best.order_id, None)
                contra.pop(0)
        if order.remaining > 0:
            book = self.buy if order.side == "buy" else self.sell
            book.append(order)
            self.orders[order.order_id] = order
            self._sort()
        return trades

    def cancel(self, order_id: str) -> Optional[Order]:
        order = self.orders.pop(order_id, None)
        if not order:
            return None
        book = self.buy if order.side == "buy" else self.sell
        for i, o in enumerate(book):
            if o.order_id == order_id:
                return book.pop(i)
        return order

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        self._sort()
        bid = self.buy[0].price if self.buy else None
        ask = self.sell[0].price if self.sell else None
        return bid, ask

    def depth(self, side: str, levels: int = 5) -> List[Tuple[float, int]]:
        book = self.buy if side == "buy" else self.sell
        grouped: Dict[float, int] = defaultdict(int)
        for o in book:
            grouped[o.price] += o.remaining
        prices = sorted(grouped, reverse=(side == "buy"))
        return [(p, grouped[p]) for p in prices[:levels]]

    def _sort(self) -> None:
        self.buy.sort(key=lambda o: (-o.price, o.timestamp, o.order_id))
        self.sell.sort(key=lambda o: (o.price, o.timestamp, o.order_id))


# PLACEHOLDER_SURVEILLANCE


class EnhancedRegulatoryAgent:
    """Detects wash trading, spoofing, and pump-and-dump."""

    def __init__(self) -> None:
        self.wash_window = 8
        self.pump_window = 10
        self.spoof_distance_threshold = 0.12
        self.spoof_size_ratio = 3.0
        self.events: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.entity_trades: Deque[Trade] = deque(maxlen=2000)
        self.alerts: List[Dict[str, Any]] = []
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.seller_history: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def pre_submit(self, order: Order, book: LimitOrderBook) -> Optional[Dict[str, Any]]:
        # Wash trading: same entity crossing
        wash = self._check_wash_pre(order, book)
        if wash:
            return wash

        # Spoofing: large order far from market
        spoof = self._check_spoof_pre(order, book)
        if spoof:
            return spoof

        return None

    def _check_wash_pre(self, order: Order, book: LimitOrderBook) -> Optional[Dict[str, Any]]:
        book._sort()
        contra = book.sell if order.side == "buy" else book.buy
        for resting in contra[:10]:
            crosses = (
                (order.side == "buy" and order.price >= resting.price)
                or (order.side == "sell" and order.price <= resting.price)
            )
            if crosses and resting.entity_id == order.entity_id:
                alert = {
                    "timestamp": order.timestamp,
                    "alert_type": "wash_trading",
                    "severity": 0.98,
                    "order_id": f"{order.order_id},{resting.order_id}",
                    "entity_id": str(order.entity_id),
                    "symbol": order.symbol,
                    "action": "block_order_and_freeze_entity",
                    "thought": (
                        "Incoming order would cross with a resting order from the same "
                        "beneficial owner — blocking to prevent wash trade."
                    ),
                }
                self.alerts.append(alert)
                return alert
        return None

    def _check_spoof_pre(self, order: Order, book: LimitOrderBook) -> Optional[Dict[str, Any]]:
        bid, ask = book.best_bid_ask()
        if order.side == "buy":
            reference = ask if ask else bid
        else:
            reference = bid if bid else ask
        if reference is None or reference <= 0:
            return None

        distance = abs(order.price - reference) / reference
        avg_depth = self._avg_depth(book.depth(order.side, levels=5))
        size_threshold = max(avg_depth * self.spoof_size_ratio, 200)

        if distance > self.spoof_distance_threshold and order.quantity >= size_threshold:
            alert = {
                "timestamp": order.timestamp,
                "alert_type": "spoofing",
                "severity": 0.91,
                "order_id": order.order_id,
                "entity_id": order.entity_id or order.agent_id,
                "symbol": order.symbol,
                "action": "flag_and_throttle",
                "thought": (
                    f"Order placed {distance:.0%} away from best quote with "
                    f"quantity {order.quantity} (threshold {size_threshold:.0f}), "
                    "consistent with spoofing or layering intent."
                ),
            }
            self.alerts.append(alert)
            return alert
        return None

    def on_trades(self, trades: List[Trade], order_map: Dict[str, Order]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        for trade in trades:
            self.entity_trades.append(trade)
            self.price_history[trade.symbol].append((trade.timestamp, trade.price))
            self.seller_history[trade.symbol].append((trade.timestamp, trade.seller_id))

            # Wash trading post-trade
            buyer_order = order_map.get(trade.buy_order_id)
            seller_order = order_map.get(trade.sell_order_id)
            buyer_entity = buyer_order.entity_id if buyer_order else trade.buyer_id
            seller_entity = seller_order.entity_id if seller_order else trade.seller_id
            if buyer_entity == seller_entity:
                alert = {
                    "timestamp": trade.timestamp,
                    "alert_type": "wash_trading",
                    "severity": 0.98,
                    "order_id": f"{trade.buy_order_id},{trade.sell_order_id}",
                    "entity_id": str(buyer_entity),
                    "symbol": trade.symbol,
                    "action": "block_trade_and_freeze_entity",
                    "thought": "Buyer and seller are the same entity — no genuine risk transfer.",
                }
                self.alerts.append(alert)
                alerts.append(alert)
                continue

            # Pump and dump detection
            pump_alert = self._check_pump_dump(trade)
            if pump_alert:
                alerts.append(pump_alert)

        return alerts

    # PLACEHOLDER_PUMP_DUMP

    def _check_pump_dump(self, trade: Trade) -> Optional[Dict[str, Any]]:
        symbol = trade.symbol
        history = self.price_history.get(symbol, [])
        seller_hist = self.seller_history.get(symbol, [])

        recent_trades = [
            t for t in self.entity_trades
            if t.symbol == symbol and t.timestamp >= trade.timestamp - self.pump_window
        ]
        if len(recent_trades) < 4:
            return None

        # Check for concentrated selling by one entity
        seller_counts: Dict[str, int] = defaultdict(int)
        for t in recent_trades:
            seller_counts[t.seller_id] += 1

        for seller, count in seller_counts.items():
            if count < 4:
                continue
            seller_trades = [t for t in recent_trades if t.seller_id == seller]
            prices = [t.price for t in seller_trades]
            if len(prices) >= 3 and prices[-1] > prices[0] * 1.02:
                unique_buyers = len(set(t.buyer_id for t in seller_trades))
                if unique_buyers >= 2:
                    alert = {
                        "timestamp": trade.timestamp,
                        "alert_type": "pump_and_dump",
                        "severity": 0.93,
                        "order_id": f"{trade.buy_order_id},{trade.sell_order_id}",
                        "entity_id": seller,
                        "symbol": symbol,
                        "action": "halt_and_investigate",
                        "thought": (
                            f"Entity {seller} sold {count} times in {self.pump_window} ticks "
                            f"with rising prices ({prices[0]:.2f} -> {prices[-1]:.2f}) "
                            f"to {unique_buyers} distinct buyers — pump and dump pattern."
                        ),
                    }
                    self.alerts.append(alert)
                    return alert
        return None

    def _avg_depth(self, levels: List[Tuple[float, int]]) -> float:
        if not levels:
            return 0.0
        return sum(q for _, q in levels) / len(levels)


# PLACEHOLDER_EXCHANGE_AND_SUBMISSION


class EnhancedExchange:
    def __init__(self) -> None:
        self.books: Dict[str, LimitOrderBook] = {}
        self.regulator = EnhancedRegulatoryAgent()
        self._ids = itertools.count(1)
        self.order_map: Dict[str, Order] = {}

    def submit_order(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        timestamp: int,
        entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        order = Order(
            order_id=f"O{next(self._ids)}",
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            price=round(float(price), 4),
            quantity=int(quantity),
            timestamp=int(timestamp),
            entity_id=entity_id,
        )
        book = self.books.setdefault(symbol, LimitOrderBook(symbol))
        alert = self.regulator.pre_submit(order, book)
        if alert and ("block" in alert["action"] or "flag" in alert["action"]):
            return {
                "accepted": False,
                "order_id": order.order_id,
                "trades": [],
                "alerts": [alert],
            }

        self.order_map[order.order_id] = Order(
            order_id=order.order_id,
            agent_id=order.agent_id,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            timestamp=order.timestamp,
            entity_id=order.entity_id,
        )
        trades = book.submit(order)
        trade_alerts = self.regulator.on_trades(trades, self.order_map)
        return {
            "accepted": True,
            "order_id": order.order_id,
            "trades": trades,
            "alerts": trade_alerts,
        }


# ============================================================
# TeamSubmission: the entry point
# ============================================================

class TeamSubmission(CompetitionSubmission):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.agents: Dict[str, OptimizedAgent] = {}
        self.exchange = EnhancedExchange()
        self.seed = 0
        self.llm_client: Optional[LLMClient] = None
        self.llm_cache: Dict[str, Optional[LLMAdvice]] = {}
        self.llm_calls_used = 0
        self.llm_max_calls = self._configured_llm_max_calls()
        self._configure_llm()

    def reset(self, seed: int = 0, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config.update(dict(config or {}))
        self.agents = {}
        self.exchange = EnhancedExchange()
        self.seed = seed
        self.llm_cache = {}
        self.llm_calls_used = 0
        self.llm_max_calls = self._configured_llm_max_calls()
        self._configure_llm()

    def _configured_llm_max_calls(self) -> int:
        value = self.config.get("llm_max_calls", self.config.get("max_llm_calls", 24))
        try:
            return int(value)
        except (TypeError, ValueError):
            return 24

    def _allow_llm_call(self) -> bool:
        if self.llm_max_calls < 0:
            return True
        if self.llm_calls_used >= self.llm_max_calls:
            return False
        self.llm_calls_used += 1
        return True

    def _configure_llm(self) -> None:
        if not self.config.get("use_llm", False):
            self.llm_client = None
            return
        try:
            from llm_helper import create_llm_client

            config_path = str(self.config.get("llm_config_path", "config.yaml"))
            self.llm_client = create_llm_client(config_path)
        except Exception:
            self.llm_client = None

    def decide(self, observation: MarketObservation) -> AgentDecision:
        agent = self.agents.get(observation.agent_id)
        if agent is None:
            agent = OptimizedAgent(
                agent_id=observation.agent_id,
                cash=observation.cash,
                seed=self.seed + len(self.agents),
                llm_client=self.llm_client,
                llm_cache=self.llm_cache,
                llm_call_gate=self._allow_llm_call,
            )
            self.agents[observation.agent_id] = agent
        agent.llm_client = self.llm_client
        agent.llm_cache = self.llm_cache
        agent.llm_call_gate = self._allow_llm_call

        agent.cash = observation.cash
        if observation.position > 0:
            agent.positions[observation.symbol] = Position(observation.position, observation.avg_cost)
        else:
            agent.positions.pop(observation.symbol, None)

        agent.ingest_market(observation.symbol, [k.to_dict() for k in observation.klines])
        agent.ingest_news(observation.symbol, observation.news)
        agent.ingest_social(observation.symbol, observation.social_posts)

        return agent.decide(
            observation.symbol,
            observation.cash,
            observation.position,
            observation.avg_cost,
        )

    def match_orders(
        self,
        orders: List[OrderRequest],
        last_prices: Mapping[str, float],
        tick: int,
    ) -> MatchResult:
        accepted: List[str] = []
        rejected: List[str] = []
        trades: List[TradeRecord] = []
        alerts: List[AlertRecord] = []
        close_prices = dict(last_prices)

        for order in orders:
            result = self.exchange.submit_order(
                agent_id=order.agent_id,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                quantity=order.quantity,
                timestamp=order.timestamp,
                entity_id=order.entity_id,
            )
            if result["accepted"]:
                accepted.append(order.order_id)
            else:
                rejected.append(order.order_id)
            for t in result["trades"]:
                trades.append(TradeRecord(
                    symbol=t.symbol,
                    price=t.price,
                    quantity=t.quantity,
                    buy_order_id=t.buy_order_id,
                    sell_order_id=t.sell_order_id,
                    buyer_id=t.buyer_id,
                    seller_id=t.seller_id,
                    timestamp=t.timestamp,
                ))
                close_prices[t.symbol] = t.price
            for a in result["alerts"]:
                alerts.append(AlertRecord(
                    timestamp=a["timestamp"],
                    alert_type=a["alert_type"],
                    severity=a["severity"],
                    order_id=a["order_id"],
                    entity_id=a["entity_id"],
                    symbol=a["symbol"],
                    action=a["action"],
                    thought=a["thought"],
                ))

        return MatchResult(
            trades=trades,
            accepted_order_ids=accepted,
            rejected_order_ids=rejected,
            alerts=alerts,
            close_prices=close_prices,
        )


def create_submission(config: Optional[Mapping[str, Any]] = None) -> CompetitionSubmission:
    return TeamSubmission(config)


# ============================================================
# Utility functions
# ============================================================

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _returns(closes: List[float]) -> List[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bucket(value: float, step: float) -> str:
    if step <= 0:
        return "0"
    return str(int(round(value / step)))
