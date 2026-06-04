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

    def test_llm_bullish_advice_protects_winner_from_mechanical_profit_taking(self):
        def fake_llm(system, user):
            return """
            {
              "market_view": "bullish",
              "sentiment_score": 0.8,
              "confidence": 0.9,
              "take_profit_bias": 0.0,
              "loss_hold_bias": 0.4,
              "risk_warning": false,
              "reason": "The rally is supported by broad positive context."
            }
            """

        agent = OptimizedAgent("llm_winner", cash=50_000, seed=7, llm_client=fake_llm)
        agent.ingest_market("SIM", make_klines([109, 110, 110, 109, 110] * 6))
        agent.ingest_news("SIM", ["management update and sector demand commentary"])
        decision = agent.decide("SIM", cash=50_000, position=500, avg_cost=100.0)

        self.assertEqual(decision.action, "hold")
        self.assertIn("LLM context", decision.thought)

    def test_llm_failure_falls_back_to_rule_decision(self):
        def failing_llm(system, user):
            raise RuntimeError("network unavailable")

        agent = OptimizedAgent("llm_fallback", cash=50_000, seed=13, llm_client=failing_llm)
        agent.ingest_market("SIM", make_klines([95, 96, 97, 98, 99] * 6))
        agent.ingest_news("SIM", ["growth upgrade"])
        decision = agent.decide("SIM", cash=50_000, position=0, avg_cost=0.0)

        self.assertIn(decision.action, {"buy", "sell", "hold"})


if __name__ == "__main__":
    unittest.main()
