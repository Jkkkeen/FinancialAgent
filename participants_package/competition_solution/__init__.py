"""Competition-ready BDI market simulation components.

🔧 MODIFIABLE — 选手可以替换或扩展这些模块。
"""

from .investment_agent import InvestmentAgent
from .exchange import ExchangeAgent, LimitOrderBook, Order, RegulatoryAgent
from . import metrics

__all__ = [
    "InvestmentAgent",
    "ExchangeAgent",
    "LimitOrderBook",
    "Order",
    "RegulatoryAgent",
    "metrics",
]
