# ============================================================
# 比赛任务二：交易所基础设施 Agent
# ============================================================
# 🔒 KEEP（不可修改）:
#   - Order / Trade 数据类结构
#   - LimitOrderBook 公共方法签名（submit, cancel, best_bid_ask, depth）
#   - ExchangeAgent 公共方法签名（submit_order, cancel_order, best_bid_ask）
#   - 价格-时间优先撮合规则
#
# 🔧 MODIFIABLE（可修改）:
#   - RegulatoryAgent 构造参数（检测阈值）
#   - _same_entity_cross_alert() / on_cancel()
#   - Alert.thought 文本内容
#
# 🚀 ADVANCED（高级玩法）:
#   - 用 LLM 生成监管 Alert 的 thought（分析订单流模式的上下文推理）
#   - 添加新的市场操纵检测类型（如 layering, quote stuffing）
#   - 实现自适应阈值（根据市场波动动态调整检测灵敏度）
#
# 💡 TIP（提示）:
#   - F1-Score 是监管检测的核心指标，需兼顾 Precision 和 Recall
#   - 响应延迟 ≤ 3 tick 得满分，> 10 tick 不得分
#   - 撮合引擎必须严格遵循价格-时间优先原则
# ============================================================

"""Limit order book and regulatory exchange agent."""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple


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
    visible: bool = True
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
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

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class LimitOrderBook:
    """Continuous LOB with strict price-time priority."""

    def __init__(self, symbol: str, tick_size: float = 0.01) -> None:
        self.symbol = symbol
        self.tick_size = tick_size
        self.buy: List[Order] = []
        self.sell: List[Order] = []
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []

    def submit(self, order: Order) -> List[Trade]:
        if order.symbol != self.symbol:
            raise ValueError("order symbol does not match book")
        trades: List[Trade] = []
        contra = self.sell if order.side == "buy" else self.buy
        while order.remaining > 0 and contra:
            self._sort_books()
            best = contra[0]
            can_cross = (
                order.side == "buy" and order.price >= best.price
            ) or (
                order.side == "sell" and order.price <= best.price
            )
            if not can_cross:
                break

            quantity = min(order.remaining, best.remaining)
            trade_price = best.price
            if order.side == "buy":
                buyer_id, seller_id = order.agent_id, best.agent_id
                buy_order_id, sell_order_id = order.order_id, best.order_id
            else:
                buyer_id, seller_id = best.agent_id, order.agent_id
                buy_order_id, sell_order_id = best.order_id, order.order_id
            trade = Trade(
                symbol=self.symbol,
                price=trade_price,
                quantity=quantity,
                buy_order_id=buy_order_id,
                sell_order_id=sell_order_id,
                buyer_id=buyer_id,
                seller_id=seller_id,
                timestamp=max(order.timestamp, best.timestamp),
            )
            trades.append(trade)
            self.trades.append(trade)
            order.remaining -= quantity
            best.remaining -= quantity
            if best.remaining <= 0:
                self.orders.pop(best.order_id, None)
                contra.pop(0)

        if order.remaining > 0:
            book = self.buy if order.side == "buy" else self.sell
            book.append(order)
            self.orders[order.order_id] = order
            self._sort_books()
        return trades

    def cancel(self, order_id: str) -> Optional[Order]:
        order = self.orders.pop(order_id, None)
        if not order:
            return None
        book = self.buy if order.side == "buy" else self.sell
        for idx, candidate in enumerate(book):
            if candidate.order_id == order_id:
                return book.pop(idx)
        return order

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        self._sort_books()
        bid = self.buy[0].price if self.buy else None
        ask = self.sell[0].price if self.sell else None
        return bid, ask

    def depth(self, side: str, levels: int = 5) -> List[Tuple[float, int]]:
        book = self.buy if side == "buy" else self.sell
        grouped: Dict[float, int] = defaultdict(int)
        for order in book:
            grouped[order.price] += order.remaining
        prices = sorted(grouped, reverse=(side == "buy"))
        return [(price, grouped[price]) for price in prices[:levels]]

    def _sort_books(self) -> None:
        self.buy.sort(key=lambda item: (-item.price, item.timestamp, item.order_id))
        self.sell.sort(key=lambda item: (item.price, item.timestamp, item.order_id))


@dataclass
class Alert:
    timestamp: int
    alert_type: str
    severity: float
    order_id: Optional[str]
    entity_id: str
    symbol: str
    action: str
    thought: str

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class RegulatoryAgent:
    """Hybrid rule/LLM-style market surveillance.

    The rules are explicit for score stability.  The produced ``thought`` field
    is the self-consistent explanation required by the challenge.

    🚀 ADVANCED: The ``thought`` field in each Alert can be replaced with LLM-generated
    analysis for richer, context-aware explanations of market manipulation patterns.
    """

    def __init__(
        self,
        wash_window: int = 8,            # 🔧 MODIFIABLE: 洗盘检测窗口（tick 数）
        spoof_cancel_window: int = 3,     # 🔧 MODIFIABLE: 虚假申报取消窗口（tick 数）
        large_order_ratio: float = 4.0,   # 🔧 MODIFIABLE: 大单判定比例（订单量 vs 平均深度）
        pump_window: int = 12,            # 🔧 MODIFIABLE: 拉高抛售检测窗口（tick 数）
    ) -> None:
        self.wash_window = wash_window
        self.spoof_cancel_window = spoof_cancel_window
        self.large_order_ratio = large_order_ratio
        self.pump_window = pump_window
        self.events: Deque[Dict[str, Any]] = deque(maxlen=2_000)
        self.open_orders: Dict[str, Order] = {}
        self.cancelled: List[Dict[str, Any]] = []
        self.alerts: List[Alert] = []
        self.entity_trades: Deque[Trade] = deque(maxlen=2_000)

    def pre_submit(self, order: Order, book: LimitOrderBook) -> Optional[Alert]:
        wash_alert = self._same_entity_cross_alert(order, book)
        if wash_alert:
            self.alerts.append(wash_alert)
            return wash_alert

        self.open_orders[order.order_id] = order
        self.events.append({"type": "submit", "timestamp": order.timestamp, "order": order})
        return None

    def _same_entity_cross_alert(self, order: Order, book: LimitOrderBook) -> Optional[Alert]:
        book._sort_books()
        contra = book.sell if order.side == "buy" else book.buy
        for resting in contra[:5]:
            crosses = (
                order.side == "buy" and order.price >= resting.price
            ) or (
                order.side == "sell" and order.price <= resting.price
            )
            if crosses and resting.entity_id == order.entity_id:
                return Alert(
                    timestamp=order.timestamp,
                    alert_type="wash_trading",
                    severity=0.98,
                    order_id=f"{order.order_id},{resting.order_id}",
                    entity_id=str(order.entity_id),
                    symbol=order.symbol,
                    action="block_order_and_freeze_entity",
                    # 🚀 ADVANCED: 可用 LLM 增强此监管理由的上下文分析
                    thought=(
                        "Incoming order would match with a resting order from the "
                        "same beneficial owner, so the exchange blocks it before execution."
                    ),
                )
        return None

    def on_cancel(self, order: Order, timestamp: int, book: LimitOrderBook) -> Optional[Alert]:
        self.open_orders.pop(order.order_id, None)
        age = timestamp - order.timestamp
        self.events.append({"type": "cancel", "timestamp": timestamp, "order": order})
        self.cancelled.append({"order": order, "timestamp": timestamp, "age": age})
        same_entity_recent = [
            item
            for item in self.cancelled[-20:]
            if (item["order"].entity_id == order.entity_id and item["timestamp"] >= timestamp - self.spoof_cancel_window)
        ]
        avg_depth = _average_depth(book.depth(order.side, levels=5))
        is_large = avg_depth == 0 or order.quantity >= max(1, avg_depth) * self.large_order_ratio
        if age <= self.spoof_cancel_window and is_large and len(same_entity_recent) >= 2:
            alert = Alert(
                timestamp=timestamp,
                alert_type="spoofing",
                severity=0.91,
                order_id=order.order_id,
                entity_id=order.entity_id or order.agent_id,
                symbol=order.symbol,
                action="intervene_cancel_and_throttle",
                # 🚀 ADVANCED: 可用 LLM 分析取消模式背后的意图
                thought=(
                    "Repeated oversized orders were cancelled within a few ticks, "
                    "which is consistent with quote stuffing or spoofing intent."
                ),
            )
            self.alerts.append(alert)
            return alert
        return None

    def on_trades(self, trades: Iterable[Trade], orders: Mapping[str, Order]) -> List[Alert]:
        alerts: List[Alert] = []
        for trade in trades:
            self.entity_trades.append(trade)
            buyer = orders.get(trade.buy_order_id)
            seller = orders.get(trade.sell_order_id)
            buyer_entity = buyer.entity_id if buyer else trade.buyer_id
            seller_entity = seller.entity_id if seller else trade.seller_id
            if buyer_entity == seller_entity:
                alert = Alert(
                    timestamp=trade.timestamp,
                    alert_type="wash_trading",
                    severity=0.98,
                    order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                    entity_id=str(buyer_entity),
                    symbol=trade.symbol,
                    action="block_trade_and_freeze_entity",
                    thought=(
                        "Buyer and seller resolve to the same beneficial owner, so "
                        "the matched trade has no economic risk transfer."
                    ),
                )
                self.alerts.append(alert)
                alerts.append(alert)
                continue

            recent = [
                item
                for item in self.entity_trades
                if item.symbol == trade.symbol and item.timestamp >= trade.timestamp - self.wash_window
            ]
            pair_count = 0
            for item in recent:
                if {item.buyer_id, item.seller_id} == {trade.buyer_id, trade.seller_id}:
                    pair_count += 1
            if pair_count >= 4:
                alert = Alert(
                    timestamp=trade.timestamp,
                    alert_type="wash_trading_ring",
                    severity=0.84,
                    order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                    entity_id=f"{trade.buyer_id}|{trade.seller_id}",
                    symbol=trade.symbol,
                    action="warn_and_sample_for_review",
                    thought="The same counterparties repeatedly reverse risk inside a short window.",
                )
                self.alerts.append(alert)
                alerts.append(alert)

        return alerts


class ExchangeAgent:
    """Multi-symbol exchange with surveillance and price-time matching."""

    def __init__(self, regulator: Optional[RegulatoryAgent] = None) -> None:
        self.books: Dict[str, LimitOrderBook] = {}
        self.regulator = regulator or RegulatoryAgent()
        self._ids = itertools.count(1)
        self.order_snapshots: Dict[str, Order] = {}

    def submit_order(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        timestamp: Optional[int] = None,
        entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = int(datetime.utcnow().timestamp())
        order = Order(
            order_id=f"O{next(self._ids)}",
            agent_id=agent_id,
            entity_id=entity_id,
            symbol=symbol,
            side=side,
            price=round(float(price), 4),
            quantity=int(quantity),
            timestamp=int(timestamp),
        )
        book = self.books.setdefault(symbol, LimitOrderBook(symbol))
        alert = self.regulator.pre_submit(order, book)
        if alert and (alert.action == "quarantine" or alert.action.startswith("block_")):
            return {"accepted": False, "order_id": order.order_id, "trades": [], "alerts": [alert.as_dict()]}

        self.order_snapshots[order.order_id] = Order(
            order_id=order.order_id,
            agent_id=order.agent_id,
            entity_id=order.entity_id,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            timestamp=order.timestamp,
        )
        trades = book.submit(order)
        alerts = self.regulator.on_trades(trades, self.order_snapshots)
        return {
            "accepted": True,
            "order_id": order.order_id,
            "trades": [trade.as_dict() for trade in trades],
            "alerts": [item.as_dict() for item in alerts],
        }

    def cancel_order(self, symbol: str, order_id: str, timestamp: int) -> Dict[str, Any]:
        book = self.books.get(symbol)
        if not book:
            return {"cancelled": False, "alerts": []}
        order = book.cancel(order_id)
        if not order:
            return {"cancelled": False, "alerts": []}
        alert = self.regulator.on_cancel(order, timestamp, book)
        return {"cancelled": True, "alerts": [alert.as_dict()] if alert else []}

    def best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        book = self.books.get(symbol)
        return book.best_bid_ask() if book else (None, None)


def _average_depth(levels: List[Tuple[float, int]]) -> float:
    if not levels:
        return 0.0
    return sum(quantity for _, quantity in levels) / len(levels)
