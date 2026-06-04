---
noteId: "834b51905c4b11f184ae01b0c1f6091d"
tags: []

---

# 常见问题排错

## 安装与导入

### ModuleNotFoundError: No module named 'competition_solution'

原因：没有在 `participants_package/` 目录下运行。

解决：

```bash
cd participants_package
python -m competition_solution.demo
```

### ImportError: No module named 'yaml'

原因：`pyyaml` 未安装。

解决：

```bash
pip install pyyaml
```

## LLM 调用

### LLM API 调用返回空响应

常见原因：

1. API key 未设置或错误。
2. `base_url` 不正确。
3. 模型名称错误。

排查：

```bash
echo $API_KEY
```

### LLM 返回的决策无效

常见原因：模型在 JSON 外包裹了 Markdown 代码块，或输出了额外解释。

处理方式：

1. 调整 prompt，明确要求只返回 JSON。
2. 降低 `temperature`。
3. 先用规则模式跑通，再接入 LLM。

## 校验与提交

### Validator 报错 missing submission.py

原因：提交目录不存在或缺少 `submission.py`。

正确结构：

```text
my_team/
├── submission.py
├── README.md
└── requirements.txt
```

### Validator 返回 missing methods

原因：提交对象没有实现全部必需方法。

必须实现：

```python
reset(seed, config=None)
decide(observation)
match_orders(orders, last_prices, tick)
```

### decide 返回类型错误

原因：`decide()` 没有返回 `AgentDecision`。

解决：使用 `submission_interface.api.AgentDecision` 构造返回值。

### match_orders 返回类型错误

原因：`match_orders()` 没有返回 `MatchResult`。

解决：使用 `submission_interface.api.MatchResult`，并确保内部字段为对应的数据类列表。

## 评测指标异常

### 处置效应 DE 为负数

Agent 表现出“快速止损、持有盈利”的反散户行为。可以适当提高处置效应相关参数，或调整盈利/亏损持仓下的卖出倾向。

### 信念-行动相关性低

检查：

- `thought` 是否表达了真实决策理由。
- `belief_score` 是否和市场信息方向一致。
- `sentiment_class` 是否和 `action` 大体一致。

### F1-Score 为 0

监管 Agent 未检测到异常，或误报过多。建议先用固定订单流分别调试：

- 同实体自买自卖。
- 远离盘口的大额挂单。
- 多账户拉升后集中卖出。

### 响应延迟过高

预警逻辑触发太晚。尽量在 `match_orders()` 处理当前 batch 时就完成检测和干预。

## 其他

### 如何安全存储 API key

推荐使用环境变量：

```bash
export API_KEY="sk-..."
```

不要把 API key 写进代码或提交目录。
