# ============================================================
# 🔒 DO NOT MODIFY THIS FILE! 🔒
# 此文件是官方竞赛接口契约，任何修改将导致校验失败和参赛资格取消。
# 所有自定义逻辑请放在 competition_solution/ 或你的提交目录中。
# ============================================================

"""Official Python interface for competition submissions.

Every team submits a directory containing ``submission.py``.  The judge imports
``create_submission(config)`` from that file and calls the methods defined by
``CompetitionSubmission``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class KLine:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketObservation:
    """All information visible to an investment agent at one decision point."""

    agent_id: str
    tick: int
    symbol: str
    klines: List[KLine]
    news: List[str] = field(default_factory=list)
    social_posts: List[Dict[str, Any]] = field(default_factory=list)
    cash: float = 0.0
    position: int = 0
    avg_cost: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["klines"] = [item.to_dict() for item in self.klines]
        return data


@dataclass
class AgentDecision:
    agent_id: str
    symbol: str
    action: str
    quantity: int
    limit_price: float
    thought: str
    belief_score: float = 0.0
    sentiment_class: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderRequest:
    order_id: str
    agent_id: str
    symbol: str
    side: str
    price: float
    quantity: int
    timestamp: int
    entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeRecord:
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buyer_id: str
    seller_id: str
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertRecord:
    timestamp: int
    alert_type: str
    severity: float
    order_id: Optional[str]
    entity_id: str
    symbol: str
    action: str
    thought: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    trades: List[TradeRecord] = field(default_factory=list)
    accepted_order_ids: List[str] = field(default_factory=list)
    rejected_order_ids: List[str] = field(default_factory=list)
    alerts: List[AlertRecord] = field(default_factory=list)
    close_prices: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades": [item.to_dict() for item in self.trades],
            "accepted_order_ids": list(self.accepted_order_ids),
            "rejected_order_ids": list(self.rejected_order_ids),
            "alerts": [item.to_dict() for item in self.alerts],
            "close_prices": dict(self.close_prices),
        }


class CompetitionSubmission(ABC):
    """Base class every formal submission must implement."""

    @abstractmethod
    def reset(self, seed: int = 0, config: Optional[Mapping[str, Any]] = None) -> None:
        """Reset all internal state before a judge run."""

    @abstractmethod
    def decide(self, observation: MarketObservation) -> AgentDecision:
        """Return one investment decision for one agent and one symbol."""

    @abstractmethod
    def match_orders(
        self,
        orders: List[OrderRequest],
        last_prices: Mapping[str, float],
        tick: int,
    ) -> MatchResult:
        """Run exchange matching and surveillance for a batch of orders."""
