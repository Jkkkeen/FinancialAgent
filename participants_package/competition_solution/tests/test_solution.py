import unittest

from competition_solution.exchange import ExchangeAgent, LimitOrderBook, Order
from competition_solution.investment_agent import InvestmentAgent, Position
from competition_solution.metrics import (
    disposition_effect,
    f1_score,
    spearman_belief_action,
    wasserstein_1d,
)


class InvestmentAgentTests(unittest.TestCase):
    def test_agent_ingests_and_decides(self):
        agent = InvestmentAgent("u1", personality="trend", cash=100_000, seed=1)
        agent.positions["AAA"] = Position(200, 100.0)
        agent.ingest_market(
            "AAA",
            [
                {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 1000}
                for i in range(30)
            ],
        )
        agent.ingest_news("AAA", ["growth upgrade breakout buy"])
        agent.ingest_social("AAA", [{"text": "bull buy", "influence": 10}])
        decision = agent.decide("AAA")
        self.assertIn(decision.action, {"buy", "sell", "hold"})
        self.assertTrue(decision.thought)


class ExchangeTests(unittest.TestCase):
    def test_price_time_priority(self):
        book = LimitOrderBook("AAA")
        book.submit(Order("s1", "seller1", "AAA", "sell", 10.0, 100, 1))
        book.submit(Order("s2", "seller2", "AAA", "sell", 10.0, 100, 2))
        trades = book.submit(Order("b1", "buyer", "AAA", "buy", 10.0, 150, 3))
        self.assertEqual(trades[0].seller_id, "seller1")
        self.assertEqual(trades[1].seller_id, "seller2")
        self.assertEqual(trades[0].quantity, 100)
        self.assertEqual(trades[1].quantity, 50)

    def test_regulator_flags_wash_trade(self):
        exchange = ExchangeAgent()
        exchange.submit_order("a1", "AAA", "sell", 10.0, 100, timestamp=1, entity_id="E")
        result = exchange.submit_order("a2", "AAA", "buy", 10.0, 100, timestamp=2, entity_id="E")
        self.assertTrue(any(alert["alert_type"] == "wash_trading" for alert in result["alerts"]))


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        de = disposition_effect(
            [
                {"side": "sell", "price": 11, "avg_cost": 10},
                {"side": "hold", "price": 9, "avg_cost": 10},
            ]
        )
        self.assertGreater(de["DE"], 0)
        self.assertAlmostEqual(spearman_belief_action([1, 0, -1], [1, 0, -1]), 1.0)
        self.assertGreaterEqual(wasserstein_1d([1, 2], [2, 3]), 0)
        self.assertEqual(f1_score([1, 0], [1, 0])["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
