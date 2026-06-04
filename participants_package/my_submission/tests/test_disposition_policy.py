import unittest

from my_submission.submission import OptimizedAgent


def make_klines(closes):
    return [
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 10_000,
        }
        for close in closes
    ]


class DispositionPolicyTests(unittest.TestCase):
    def test_moderate_winner_takes_partial_profit_when_not_strongly_bullish(self):
        agent = OptimizedAgent("winner", cash=50_000, seed=7)
        agent.ingest_market("SIM", make_klines([109, 110, 110, 109, 110] * 6))
        agent.ingest_news("SIM", ["mixed guidance, no clear catalyst"])
        decision = agent.decide("SIM", cash=50_000, position=500, avg_cost=100.0)

        self.assertEqual(decision.action, "sell")
        self.assertGreater(decision.quantity, 0)
        self.assertLessEqual(decision.quantity, 500)
        self.assertLess(decision.belief_score, 0)

    def test_loser_holds_when_bearish_signal_is_not_extreme(self):
        agent = OptimizedAgent("loser", cash=50_000, seed=11)
        agent.ingest_market("SIM", make_klines([92, 91, 90, 91, 90] * 6))
        agent.ingest_news("SIM", ["risk downgrade, weak demand"])
        decision = agent.decide("SIM", cash=50_000, position=500, avg_cost=100.0)

        self.assertNotEqual(decision.action, "sell")


if __name__ == "__main__":
    unittest.main()
