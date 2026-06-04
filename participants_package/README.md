# AI Agent 金融市场模拟大赛

本包包含开发参赛方案所需的全部工具和数据。

## 目录结构

```
├── 题目.md                        # 比赛题目（初赛两题 + 详细评分标准）
├── SUBMISSION_GUIDE.md            # 提交指南（格式、校验、注意事项）
├── TUTORIAL.md                    # 📖 完整教程（从零开始的开发指南）
├── DATA_FORMAT.md                 # 📊 数据格式说明
├── TROUBLESHOOTING.md             # 🔧 常见问题排错
├── SELF_SCORING.md                # 📈 自评指南
├── requirements.txt               # Python 依赖声明
├── .gitignore                     # Git 忽略规则
├── config.example.yaml            # ⚙️ 配置模板（复制为 config.yaml 后填入密钥）
├── llm_helper.py                  # 🤖 LLM 调用辅助类（开箱即用）
├── self_score.py                  # 📊 自评脚本
├── evaluate_submission.py         # 📊 初赛专项评测脚本
├── submission_interface/          # 🔒 官方提交接口（不可修改）
│   ├── api.py                     # 数据类 + CompetitionSubmission 抽象基类
│   ├── validator.py               # 本地校验器，提交前自检
│   └── example_submission/        # 示例提交模板，复制后替换策略
├── competition_solution/          # 官方基线方案（参考实现）
│   ├── investment_agent.py        # BDI 投资 Agent
│   ├── exchange.py                # 限价订单簿 + 监管 Agent
│   ├── metrics.py                 # 所有评分指标实现
│   ├── demo.py                    # 端到端冒烟测试
│   └── tests/                     # 单元测试
└── data/                          # 数据文件
    ├── klines_dict.pkl            # 1h K线数据（前10币种）
    └── fetch_binance_kline.py     # 数据获取脚本
```

## 快速开始

### 1. 安装依赖

```bash
cd participants_package
pip install -r requirements.txt
```

### 2. 配置 LLM

```bash
# 复制配置模板
cp config.example.yaml config.yaml

# 编辑 config.yaml，修改 provider/model 等参数
# 设置 API key 环境变量
export API_KEY="your-api-key"
```

### 3. 校验示例提交

```bash
python -m submission_interface.validator submission_interface/example_submission
```

看到 `"status": "ok"` 即表示接口合格。

### 4. 跑通基线方案

```bash
python -m competition_solution.demo
```

### 5. 跑基线方案测试

```bash
python -m unittest discover -s competition_solution/tests
```

### 6. 跑初赛专项评测

```bash
python evaluate_submission.py submission_interface/example_submission
```

## 本地测评方式

本地测评脚本是：

```bash
python evaluate_submission.py path/to/team_dir
```

它的定位是开发阶段的稳定专项测评，用固定场景检查你的提交是否符合当前两道题的要求。它不是 A 股或 crypto 的完整历史回测，也没有读取真实舆论数据；任务一使用虚拟标的 `SIM`、模拟 K 线和手写新闻/社交文本，任务二使用固定订单流场景。

### 任务一：投资 Agent

脚本会构造 6 个 `MarketObservation` 场景：

- 盈利持仓 + 看涨新闻/社交
- 亏损持仓 + 看跌新闻/社交
- 空仓 + 看涨信息
- 空仓 + 看跌信息
- 中性震荡行情
- 大幅盈利 + 止盈压力

每个场景会调用：

```python
decide(observation)
```

评分项包括：

- 返回值是否为 `AgentDecision`
- `action` 是否为 `buy` / `sell` / `hold`
- `thought` 是否存在
- `belief_score` 是否在 `[-1, 1]`
- `sentiment_class` 是否为 `-1` / `0` / `1`
- 买入/卖出数量是否越界
- 处置效应是否合理
- 信念和交易动作是否一致
- 换手率是否接近参考分布

### 任务二：交易所 Agent

脚本会构造固定订单流并调用：

```python
match_orders(orders, last_prices, tick)
```

测试场景包括：

- 正常撮合：验证价格时间优先和部分成交
- Wash Trading：同一实体自买自卖，应预警或拦截
- Spoofing：远离盘口的大额订单，应识别风险
- Pump and Dump：多账户拉升后集中卖出，应识别异常

评分项包括：

- 成交顺序和成交数量是否正确
- 是否误报正常订单
- 异常检测的 Precision / Recall / F1
- 预警响应延迟
- `AlertRecord` 结构和解释是否完整

如需保存机器可读报告：

```bash
python evaluate_submission.py path/to/team_dir --json report.json
```

### 7. 开发自己的方案

```bash
# 复制示例提交到你的队伍目录
cp -r submission_interface/example_submission my_team

# 修改 my_team/submission.py 中的三个核心方法
# 用你的策略替换默认实现

# 本地校验
python -m submission_interface.validator my_team

# 初赛专项评测
python evaluate_submission.py my_team
```

## 代码修改指引

所有源文件使用以下标注帮助选手理解哪些可以改、哪些不能改：

| 标注 | 含义 | 示例 |
|---|---|---|
| 🔒 KEEP | 不可修改（接口契约） | 公共方法签名、数据类结构 |
| 🔧 MODIFIABLE | 可自由修改 | 性格参数、检测阈值、决策公式 |
| 🚀 ADVANCED | 高级扩展点 | LLM 替换规则式决策、自定义网络拓扑 |
| 💡 TIP | 提示和最佳实践 | 调优建议、得分区间参考 |

详细说明见 `TUTORIAL.md`。

## 提交格式

每个队伍提交一个目录，必须包含：

- `submission.py` — 暴露 `create_submission(config)` 函数
- `README.md` — 队伍说明
- `requirements.txt` — 如需额外依赖

详见 `SUBMISSION_GUIDE.md`。

## 文档索引

| 文档 | 用途 |
|---|---|
| `TUTORIAL.md` | 从零开始的全流程教程 |
| `DATA_FORMAT.md` | K线数据格式说明 |
| `TROUBLESHOOTING.md` | 常见错误与解决办法 |
| `SELF_SCORING.md` | 如何本地自测得分 |
| `SUBMISSION_GUIDE.md` | 正式提交格式与要求 |
| `题目.md` | 完整赛题与评分标准 |

---

模型：deepseek（也支持 OpenAI 兼容接口）
每个队伍：1 美元 API 额度
