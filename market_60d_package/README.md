---
noteId: "782bca305e0111f18d9319383af6a959"
tags: []

---

# 60天匿名市场数据包

本目录包含一份可直接用于本地测评的匿名数据和对应评测器。

## 文件

- `data/market_60d.csv`: 匿名行情数据
- `data/calendar_60d.csv`: 相对交易日
- `data/profile_60d.csv`: 匿名标的资料
- `data/news_60d.jsonl`: 匿名新闻数据
- `data/metadata.json`: 数据说明
- `evaluate_60d_submission.py`: 本地测评器

## 使用

在参赛包根目录运行：

```bash
python new_data/market_60d_package/evaluate_60d_submission.py submission_interface/example_submission
```

如需启用 API agent：

```bash
export API_KEY="your-key"
python new_data/market_60d_package/evaluate_60d_submission.py submission_interface/example_submission --use-llm --llm-config-path config.yaml
```
