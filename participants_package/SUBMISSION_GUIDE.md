---
noteId: "5e2af6d058cd11f1b0bfe5d6c0c9ce88"
tags: []

---

# AI Agent 金融市场模拟大赛比赛提交说明

为保证正式评测一致，所有队伍使用统一提交接口。

## 1. 提交目录格式

每队提交一个目录，例如：

```text
submissions/team_alpha/
├── submission.py
├── README.md
└── requirements.txt
```

其中 `submission.py` 是唯一强制入口。

## 2. 必须实现的入口

`submission.py` 必须提供：

```python
def create_submission(config: dict):
    return TeamSubmission(config)
```

返回对象必须实现：

```python
reset(seed, config=None)
decide(observation)
match_orders(orders, last_prices, tick)
```

具体数据结构见：

```text
submission_interface/api.py
```

## 3. 官方示例

示例提交在：

```text
submission_interface/example_submission/submission.py
```

同学可以复制它作为模板，然后替换内部策略。

## 4. 提交前校验

在 `participants_package/` 目录下运行：

```bash
python -m submission_interface.validator path/to/team_dir
```

例如：

```bash
python -m submission_interface.validator submission_interface/example_submission
```

看到 JSON 输出且 `"status": "ok"` 即接口合格。

## 5. 评测覆盖

初赛专项评测重点考核：

- `decide`：单体投资 Agent 的信念、内心独白和交易动作。
- `match_orders`：撮合引擎、价格时间优先、异常交易识别与干预。

本地运行：

```bash
python evaluate_submission.py path/to/team_dir
```

## 6. 禁止事项

- 不要修改官方评测接口文件。
- 不要读取测试集标签或硬编码测试答案。
- 不要依赖未声明的外部服务。
- 不要删除或改动仓库原始数据。
