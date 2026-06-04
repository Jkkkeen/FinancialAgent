# 数据格式说明

## K线数据 (`data/klines_dict.pkl`)

### 文件结构

```python
{
    "interval": "1h",          # K线周期
    "data": {
        "BTCUSDT": DataFrame,  # 每个币种一个 DataFrame
        "ETHUSDT": DataFrame,
        ...
    }
}
```

### 加载方式

```python
import pickle
import pandas as pd

with open("data/klines_dict.pkl", "rb") as f:
    obj = pickle.load(f)

interval = obj["interval"]     # "1h"
klines_dict = obj["data"]      # Dict[str, pd.DataFrame]

# 访问 BTCUSDT 的 K线
btc_df = klines_dict["BTCUSDT"]
print(btc_df.head())
```

### DataFrame 列说明

| 列名 | 类型 | 说明 |
|---|---|---|
| `open_time` | datetime64[ns, UTC] | K线开盘时间 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `volume` | float | 成交量（基础资产） |
| `close_time` | datetime64[ns, UTC] | K线收盘时间 |
| `quote_asset_volume` | float | 成交额（报价资产） |
| `number_of_trades` | float | 成交笔数 |
| `taker_buy_base_asset_volume` | float | 主动买入量（基础资产） |
| `taker_buy_quote_asset_volume` | float | 主动买入额（报价资产） |
| `ignore` | float | 忽略字段 |

### 数据在 Agent 管线中的流转

```
klines_dict.pkl
    │
    │ 加载为 DataFrame → 转换为 Dict[List[Dict]]
    ▼
┌─────────────────────────────┐
│ agent.ingest_market(symbol,  │
│   [{"open": ..., "close": ..., ...}, ...]) │
└─────────────────────────────┘
    │
    │ 存储最近 90 根 K 线
    ▼
┌─────────────────────────────┐
│ agent.update_beliefs(symbol) │
│   - 计算 momentum（短/长期价格比）  │
│   - 计算 volatility（收益率标准差） │
│   - 计算 fair_value（价值锚定）     │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ agent.decide(symbol)         │
│   返回 buy/sell/hold 决策    │
└─────────────────────────────┘
```

### 支持的交易对

默认包含 10 个 Binance 永续合约交易对：

`BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `TRXUSDT`, `AVAXUSDT`, `DOTUSDT`

时间范围：约 2021-01-01 至 2026-01-01 的 1h K线。

### 自定义数据

如需使用其他交易对或时间范围，运行：

```bash
python data/fetch_binance_kline.py \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --interval 1h \
    --out-dir data \
    --out-file my_klines.pkl
```

### 在 Agent 中使用

MarketObservation 中的 `klines` 字段是 `List[KLine]` 对象（由评测系统构造），基线方案在 `ingest_market()` 中将其转换为 `Dict`：

```python
def ingest_market(self, symbol: str, klines: Iterable[Mapping[str, Any]]) -> None:
    rows = []
    for row in klines:
        rows.append({
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)),
        })
    if rows:
        self._market[symbol] = rows[-90:]  # 保留最近90根K线
```
