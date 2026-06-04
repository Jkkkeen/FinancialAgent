---
noteId: "5baa0e105c4b11f184ae01b0c1f6091d"
tags: []

---

# AI Agent 金融市场模拟大赛初赛教程

本教程只覆盖当前题目中的两项能力：投资 Agent 决策，以及交易所撮合与监管。

---

## 1. 环境准备

安装依赖：

```bash
cd participants_package
pip install -r requirements.txt
```

配置 LLM（可选，基线规则模式无需 LLM）：

```bash
cp config.example.yaml config.yaml
export API_KEY="sk-your-key-here"
```

验证安装：

```bash
python -m competition_solution.demo
python -m unittest discover -s competition_solution/tests
python -m submission_interface.validator submission_interface/example_submission
python evaluate_submission.py submission_interface/example_submission
```

---

## 2. 项目架构

核心数据流：

```text
MarketObservation (K线 + 新闻 + 社交帖子)
    |
    v
decide(observation) -> AgentDecision
    |
    v
match_orders(orders, last_prices, tick) -> MatchResult
```

提交对象必须实现：

| 方法 | 用途 | 返回值 |
|---|---|---|
| `reset(seed, config)` | 清理内部状态 | None |
| `decide(observation)` | 单体投资 Agent 决策 | AgentDecision |
| `match_orders(orders, last_prices, tick)` | 撮合订单并输出监管结果 | MatchResult |

---

## 3. 任务一：投资 Agent 建模

基线方案使用 BDI（信念-欲望-意图）框架：

1. `ingest_market()` 读取 K 线。
2. `ingest_news()` 读取新闻。
3. `ingest_social()` 读取社交帖子。
4. `decide()` 生成动作、数量、限价、信念分数和内心独白。

四种预设性格：

| 性格 | 特点 |
|---|---|
| aggressive | 高风险偏好，高换手 |
| value | 低换手，更偏长期持有 |
| trend | 更重视趋势和社交信号 |
| anxious | 更强亏损厌恶和情绪反应 |

规则式决策的核心思路：

```python
raw_intention = (
    0.55 * belief.score
    + 0.25 * value_gap
    + 0.20 * risk_appetite
    - 0.14
)
```

也可以注入 LLM：

```python
from llm_helper import create_llm_client
from competition_solution.investment_agent import InvestmentAgent

llm = create_llm_client("config.yaml")
agent = InvestmentAgent("A0001", personality="trend", llm_client=llm)
```

任务一关注：

- `action` 是否有效：`buy` / `sell` / `hold`
- `thought` 是否和动作一致
- `belief_score` 是否落在 `[-1, 1]`
- 是否出现适度处置效应
- 换手率是否合理

---

## 4. 任务二：交易所与监管

`LimitOrderBook` 实现价格-时间优先：

- 买单价高优先，同价先到先得。
- 卖单价低优先，同价先到先得。
- 价格交叉时自动撮合。
- 部分成交后剩余数量留在订单簿。

`RegulatoryAgent` 检测：

| 类型 | 典型模式 |
|---|---|
| Wash Trading | 同一实体或关联账户自买自卖 |
| Spoofing | 远离盘口的大额订单诱导市场 |
| Pump and Dump | 多账户协同拉升后集中卖出 |

任务二关注：

- 撮合结果是否正确
- `trades`、`accepted_order_ids`、`rejected_order_ids`、`close_prices` 是否结构完整
- 异常检测的 Precision / Recall / F1
- 是否能在较少 tick 内预警或拦截
- `AlertRecord.thought` 是否解释清楚

---

## 5. 本地评测

接口校验：

```bash
python -m submission_interface.validator my_team
```

专项评测：

```bash
python evaluate_submission.py my_team
```

导出 JSON 报告：

```bash
python evaluate_submission.py my_team --json report.json
```

---

## 6. 常见陷阱

1. `reset()` 没有清理旧订单簿或旧 Agent，导致多轮评测互相污染。
2. Agent 的 `thought` 看涨但 `action` 却卖出，信念-行动相关性会下降。
3. 卖出数量超过持仓，或买入金额远超现金。
4. 撮合同价订单时没有按时间优先。
5. 监管阈值过松导致误报，或过严导致漏报。
6. 预警太晚，响应延迟分会下降。
