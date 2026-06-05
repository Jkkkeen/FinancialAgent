import unittest

from my_submission.submission import OptimizedAgent, TeamSubmission
from submission_interface.api import KLine, MarketObservation


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


def make_observation(agent_id, symbol, closes, news, position=100, avg_cost=100.0):
    return MarketObservation(
        agent_id=agent_id,
        tick=1,
        symbol=symbol,
        klines=[
            KLine(
                symbol=symbol,
                timestamp=str(index),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for index, row in enumerate(make_klines(closes), start=1)
        ],
        news=list(news),
        social_posts=[],
        cash=50_000,
        position=position,
        avg_cost=avg_cost,
    )


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

    def test_llm_skips_low_value_context_without_news_or_position(self):
        calls = {"count": 0}

        def fake_llm(system, user):
            calls["count"] += 1
            return '{"market_view":"neutral","confidence":0.5}'

        agent = OptimizedAgent("llm_skip", cash=50_000, seed=17, llm_client=fake_llm)
        agent.ingest_market("SIM", make_klines([100, 101, 102, 103, 104] * 6))
        decision = agent.decide("SIM", cash=50_000, position=0, avg_cost=0.0)

        self.assertIn(decision.action, {"buy", "sell", "hold"})
        self.assertEqual(calls["count"], 0)
        self.assertNotIn("LLM context", decision.thought)

    def test_llm_reuses_same_news_context_from_shared_cache(self):
        calls = {"count": 0}

        def fake_llm(system, user):
            calls["count"] += 1
            return """
            {
              "market_view": "bullish",
              "sentiment_score": 0.7,
              "confidence": 0.8,
              "take_profit_bias": 0.2,
              "loss_hold_bias": 0.5,
              "risk_warning": false,
              "reason": "Shared news context is constructive."
            }
            """

        shared_cache = {}
        first = OptimizedAgent("llm_cache_1", cash=50_000, seed=19, llm_client=fake_llm, llm_cache=shared_cache)
        second = OptimizedAgent("llm_cache_2", cash=50_000, seed=23, llm_client=fake_llm, llm_cache=shared_cache)
        for agent, symbol in ((first, "AAA"), (second, "BBB")):
            agent.ingest_market(symbol, make_klines([101, 100, 101, 100, 101] * 6))
            agent.ingest_news(symbol, ["complex macro policy update with mixed implications"])
            agent.decide(symbol, cash=50_000, position=100, avg_cost=100.0)

        self.assertEqual(calls["count"], 1)

    def test_team_submission_shares_llm_cache_between_agents(self):
        calls = {"count": 0}

        def fake_llm(system, user):
            calls["count"] += 1
            return """
            {
              "market_view": "neutral",
              "sentiment_score": 0.1,
              "confidence": 0.7,
              "take_profit_bias": 0.5,
              "loss_hold_bias": 0.5,
              "risk_warning": false,
              "reason": "Same context should be cached."
            }
            """

        submission = TeamSubmission({"use_llm": False, "llm_max_calls": 8})
        submission.llm_client = fake_llm
        submission.reset(seed=3, config={"use_llm": False, "llm_max_calls": 8})
        submission.llm_client = fake_llm

        news = ["complex macro policy update with mixed implications"]
        closes = [101, 100, 101, 100, 101] * 6
        submission.decide(make_observation("agent_a", "AAA", closes, news))
        submission.decide(make_observation("agent_b", "BBB", closes, news))

        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
