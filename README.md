# worldcup-predictor — World Cup 2026 AI Match Prediction Engine

**Elo + XGBoost + Market Odds Ensemble**, 從小組賽到淘汰賽，方向正確率 **~80%**（經過 27 場實際比賽驗證）。

## 🏗️ 架構

```
                      ┌────────────────────┐
                      │    Data Layer      │
                      │  The Odds API      │
                      │  26worldcup.cn     │
                      │  競彩SP (澳客)      │
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │   Analysis Core    │
                      │  ┌──────────┐      │
                      │  │ XGBoost  │ 20%  │
                      │  │ Elo V3   │ 30%  │
                      │  │ Market   │ 50%  │
                      │  └──────────┘      │
                      │ ▲ v5.0 Qualitative │
                      │ │  Layer: Injury   │
                      │ │  Brand Premium  │
                      │ │  Defense Adjust │
                      │ │  Altitude/Battle│
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │   Output Layer     │
                      │  HTML Reports      │
                      │  競彩推薦格式       │
                      │  三通道推送         │
                      └────────────────────┘
```

## 🚀 快速開始

```bash
pip install -r requirements.txt
cd src
python predictor.py             # 預設 demo 模式
```

**預測特定比賽：**
```bash
python predictor.py --match "Brazil" "France"
```

**使用競彩格式：**
```python
from predictor import predict_jingcai
result = predict_jingcai(
    match_no="061", home="Norway", away="France",
    sp_odds={"home": 4.75, "draw": 4.27, "away": 1.46},
    is_knockout=True,
    away_injuries=["Mbappe"]  # v5.0 傷病參數
)
```

## 📊 模型版本演進

| 版本 | 日期 | 核心改進 | 方向正確率 |
|:---:|:----:|:---------|:---------:|
| v1.0 | 06/12 | 基礎 Elo + Poisson | — |
| v2.0 | 06/14 | Elo 動態更新 + 三大缺口修復 | — |
| v3.0 | 06/14 | XGBoost ML（32K 場訓練，draw recall 0.45） | — |
| v4.0 | 06/15 | Ensemble（ML+Elo+市場） | 50% → 66% |
| v4.5 | 06/22 | 新聞層 + 競彩賠率集成 | 85.7% (3日) |
| v4.7 | 06/25 | Elo FIFA 排名校準 | 83.3% (單日) |
| v4.9 | 06/28 | Dixon-Coles + 淘汰賽模式 | 76.9% (淘汰賽) |
| **v5.0** | **07/06** | **定性層（傷病/品牌/防守/海拔/戰意）** | — |

### v5.0 亮點
- `injury_factor()` — 三級傷病制度（-80/-50/-20 Elo，xG 0.75~0.93 折）
- `brand_premium_corrector()` — 品牌溢價修正（巴西 @1.79 → 實值 @2.10）
- `knockout_defense_adjuster()` — 淘汰賽防守隊 xG 折讓（維德角 -28%）
- `altitude_factor()` — 海拔高度懲罰（>1,500m → -80 Elo）
- `battle_intent_factor()` — 小組賽戰意調整（已出線 -60 Elo）

## 📂 項目結構

```
worldcup-predictor/
├── src/                    # 核心源碼（預測引擎+每日流水線）
│   ├── predictor.py        # ⭐ 主預測引擎（Elo + XGBoost + Ensemble）
│   ├── daily_predict.py    # 每日預測流水線
│   └── rebuild_elo.py      # Elo V3 重建腳本
├── docs/                   # 模型文檔
│   ├── model_v5.0_qualitative_layer.md
│   ├── model_v4.7_jingcai_layer.md
│   └── model_v4.7_elo_correction.md
├── examples/               # 範例輸出
├── requirements.txt        # Python 依賴
├── ATTRIBUTIONS.md         # 來源聲明（goaliqlab 等）
├── CHANGELOG.md            # 版本變更記錄
└── LICENSE                 # MIT License
```

## 🙏 致謝

- **goaliqlab** — 核心預測引擎基於 MIT 授權開源項目
- **The Odds API** — 即時市場賠率
- **26worldcup.cn** — 世界盃數據中心
- 詳見 `ATTRIBUTIONS.md`

## ⚠️ 免責聲明

本項目僅供學習與研究用途。不構成任何投注建議。預測模型有其固有局限性，實際比賽結果受大量隨機因素影響。

