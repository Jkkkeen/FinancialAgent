# ============================================================
# 正式提交模板 — 这是你的参赛入口！
# ============================================================
# 使用方法:
#   1. 复制整个 example_submission/ 目录到你的队伍目录
#   2. 修改 submission.py 中的三个核心方法
#   3. 校验: python -m submission_interface.validator your_team_dir
# ============================================================
# 🔒 KEEP（不可修改）:
#   - create_submission(config) 函数签名
#   - TeamSubmission 必须继承 CompetitionSubmission
#   - 三个方法的签名: reset / decide / match_orders
#   - submission_interface/api.py 中的所有数据类定义
#
# 🔧 MODIFIABLE（可修改）:
#   - decide() 中的 Agent 创建、个性分配、LLM 注入
#   - match_orders() 中的订单处理逻辑
#   - 导入你自己的策略模块替换 competition_solution
#
# 🚀 ADVANCED（高级玩法）:
#   - 注入 LLM 客户端到 InvestmentAgent:
#       from llm_helper import create_llm_client
#       agent = InvestmentAgent(..., llm_client=create_llm_client("config.yaml"))
#   - 多策略混合: 不同 Agent 使用不同决策逻辑
#   - 动态社交网络: 基于实时交易关系更新图结构
#
# 💡 TIP（提示）:
#   - 所有三个方法都会被评测系统独立调用，确保 reset() 正确清理状态
#   - 数值越界会扣分（如 quantity > position 的卖单）
# ============================================================

"""Example formal submission.

Teams may copy this file into their own submission directory and replace the
internal strategy while keeping the public method signatures unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from competition_solution.exchange import ExchangeAgent
from competition_solution.investment_agent import InvestmentAgent, Position
from submission_interface.api import (
    AgentDecision,
    AlertRecord,
    CompetitionSubmission,
    MarketObservation,
    MatchResult,
    OrderRequest,
    TradeRecord,
)


class TeamSubmission(CompetitionSubmission):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.agents: Dict[str, InvestmentAgent] = {}
        self.exchange = ExchangeAgent()

    def reset(self, seed: int = 0, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config.update(dict(config or {}))
        self.agents = {}
        self.exchange = ExchangeAgent()
        self.seed = seed

    def decide(self, observation: MarketObservation) -> AgentDecision:
        # 🔧 MODIFIABLE: 在这里实现你的单体 Agent 决策逻辑
        agent = self.agents.get(observation.agent_id)
        if agent is None:
            # 🔧 MODIFIABLE: 个性分配策略 — 可改为从 config 读取分布并按概率分配
            personality = self.config.get("default_personality", "trend")
            # 🚀 ADVANCED: 注入 LLM 客户端
            # from llm_helper import create_llm_client
            # llm = create_llm_client("config.yaml")
            agent = InvestmentAgent(
                agent_id=observation.agent_id,
                personality=personality,
                cash=observation.cash,
                seed=self.seed + len(self.agents),
            )
            self.agents[observation.agent_id] = agent
        agent.cash = observation.cash
        if observation.position > 0:
            agent.positions[observation.symbol] = Position(observation.position, observation.avg_cost)
        else:
            agent.positions.pop(observation.symbol, None)
        agent.ingest_market(observation.symbol, [item.to_dict() for item in observation.klines])
        agent.ingest_news(observation.symbol, observation.news)
        agent.ingest_social(observation.symbol, observation.social_posts)
        decision = agent.decide(observation.symbol)
        return AgentDecision(
            agent_id=decision.agent_id,
            symbol=decision.symbol,
            action=decision.action,
            quantity=decision.quantity,
            limit_price=decision.limit_price,
            thought=decision.thought,
            belief_score=decision.belief_score,
            sentiment_class=decision.sentiment_class,
        )

    def match_orders(
        self,
        orders: List[OrderRequest],
        last_prices: Mapping[str, float],
        tick: int,
    ) -> MatchResult:
        # 🔧 MODIFIABLE: 在这里实现你的撮合引擎和监管逻辑
        accepted: List[str] = []
        rejected: List[str] = []
        trades: List[TradeRecord] = []
        alerts: List[AlertRecord] = []
        close_prices = dict(last_prices)

        # 🔧 MODIFIABLE: 订单处理循环 — 可以添加预处理、风控检查等
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
            for item in result["trades"]:
                trades.append(TradeRecord(**item))
                close_prices[item["symbol"]] = item["price"]
            for item in result["alerts"]:
                alerts.append(AlertRecord(**item))

        return MatchResult(
            trades=trades,
            accepted_order_ids=accepted,
            rejected_order_ids=rejected,
            alerts=alerts,
            close_prices=close_prices,
        )


def create_submission(config: Optional[Mapping[str, Any]] = None) -> CompetitionSubmission:
    return TeamSubmission(config)
