---
noteId: "6f2b7e605c4b11f184ae01b0c1f6091d"
tags: []

---

# 自评指南

本地自评只覆盖当前题目中的两项内容：投资 Agent 和交易所 Agent。

> 注意：本地自评结果仅用于开发参考，不代表官方隐藏测试最终评分。

## 使用方式

```bash
python evaluate_submission.py submission_interface/example_submission
```

兼容入口：

```bash
python self_score.py submission_interface/example_submission
```

导出 JSON：

```bash
python evaluate_submission.py my_team --json report.json
```

## 任务一指标

```python
from competition_solution.metrics import (
    disposition_effect,
    spearman_belief_action,
    wasserstein_1d,
)

trades = [
    {"side": "sell", "price": 110, "avg_cost": 100},
    {"side": "hold", "price": 90, "avg_cost": 100},
]
de = disposition_effect(trades)

rho = spearman_belief_action([1, 0, -1], [1, 0, -1])
wd = wasserstein_1d([0.12, 0.08, 0.05], [0.10, 0.07, 0.06])

print(de, rho, wd)
```

重点关注：

- 处置效应 `DE` 是否落在合理区间。
- `thought` / `belief_score` / `sentiment_class` 是否和交易动作一致。
- 换手率是否过高或过低。

## 任务二指标

```python
from competition_solution.metrics import f1_score, response_latency_scores

y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]
f1 = f1_score(y_true, y_pred)

attack_ticks = {"wash_1": 10, "spoof_1": 25}
alert_ticks = {"wash_1": 12, "spoof_1": 28}
latency = response_latency_scores(attack_ticks, alert_ticks)

print(f1, latency)
```

重点关注：

- 价格时间优先是否正确。
- Wash Trading、Spoofing、Pump and Dump 是否能被及时识别。
- 误报和漏报是否可控。
- 每条预警是否有清晰的 `thought`。

## 调试建议

1. 先跑 `python -m submission_interface.validator my_team`，确保接口合格。
2. 再跑 `python evaluate_submission.py my_team`，看分项指标。
3. 优先修复格式、越界数量和撮合规则，再调监管阈值。
4. 固定随机种子，确保每次调整可复现。
