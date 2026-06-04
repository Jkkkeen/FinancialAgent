---
noteId: "38496420585e11f1b0bfe5d6c0c9ce88"
tags: []

---

# Competition Solution

This package turns `题目.md` into a runnable baseline:

- `investment_agent.py`: BDI `InvestmentAgent` with market/news/social ingestion, personality traits, belief updates, disposition-effect bias, and decision thoughts.
- `exchange.py`: price-time-priority limit order book plus a regulatory agent for wash trading, spoofing, and pump-and-dump alerts.
- `metrics.py`: PGR/PLR/DE, Spearman belief-action correlation, Wasserstein turnover distance, F1, and latency scoring.
- `demo.py`: an end-to-end smoke run.

Run:

```bash
python -m competition_solution.demo
python -m unittest discover -s competition_solution/tests
```
