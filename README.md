# worldcup-predictor — World Cup 2026 AI Match Prediction Engine

**Elo + XGBoost + Market Odds Ensemble**, 從小組賽到淘汰賽,方向正確率 **73.3%**（30 場實際比賽驗證）。

> **最新版本:v6.2** — 決賽教訓修正（崩潰效應、連續加時懲罰、嚴厲裁判、決賽 xG cap）
> 詳見 [CHANGELOG.md](CHANGELOG.md)

---

## 🏗️ 架構

```
                      ┌────────────────────┐
                      │    Data Layer      │
                      │  The Odds API      │
                      │  44k Dataset       │
                      │  競彩 SP (澳客)     │
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │   Analysis Core    │
                      │  ┌──────────┐      │
                      │  │ XGBoost  │ 15%  │ ← KO 模式
                      │  │ Elo V3   │ 20%  │
                      │  │ Market   │ 65%  │
                      │  └──────────┘      │
                      │ ┌── v6.0 H2H ────┐ │
                      │ │   44k 往績層   │ │
                      │ ├── v6.1 Clutch ─┤ │
                      │ │   絕殺基因     │ │
                      │ └── v6.2 修正 ───┘ │
                      │    崩潰效應       │
                      │    加時懲罰       │
                      │    嚴厲裁判       │
                      │    決賽 xG cap    │
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │   Output Layer     │
                      │  HTML Reports      │
                      │  競彩推薦格式       │
                      │  TEMP-QUIZ 欄位    │
                      └────────────────────┘
```

---

## 🔑 API 設置

本項目需要以下免費 API Key 先可以正常運行:

| API | 用途 | 申請連結 |
|:----|:----|:-------|
| **The Odds API** | 即時市場賠率 | https://the-odds-api.com/#get-access |

將 Key 填入 `.env` 檔案（可複製 `.env.example`）:

```bash
cp .env.example .env
# 然後編輯 .env 填入你的 API Key
```

---

## 🚀 快速開始

```bash
pip install -r requirements.txt
cd src
python predictor.py             # 預設 demo 模式
```

### 預測單場比賽

```python
from predictor import predict_ensemble, EloRatingSystem
import joblib

elo = joblib.load("models/elo_v3.joblib")
model = joblib.load("models/xgb_weighted.joblib")

result = predict_ensemble(
    "Spain", "Argentina",
    elo, model, feat_df,
    market_odds={"h2h_home": 2.35, "h2h_draw": 3.20, "h2h_away": 3.10},
    is_knockout=True,
    # v6.2 參數
    away_et_fatigue=3,      # 阿根廷連續 3 場 KO 加時
    away_health=2,          # 疲勞
    referee="Vinčić",       # 嚴厲裁判
    tournament_stage="final",
)

print(f"xG: {result['xg_home']} vs {result['xg_away']}")
print(f"晉級: H {result['ko']['advance_home']} A {result['ko']['advance_away']}")
print(f"預期黃牌: {result['exp_yellow']}")
print(f"崩潰風險: {result['v6_2']['collapse_risk_away']}")
```

---

## 📊 模型演進

| 版本 | 日期 | 核心改進 |
|:-----|:-----|:---------|
| **v6.2** | 2026-07-20 | 決賽教訓修正:崩潰效應、連續加時懲罰、嚴厲裁判、決賽 xG cap |
| **v6.1** | 2026-07-16 | Clutch gene（絕殺基因）、recency H2H |
| **v6.0** | 2026-07-15 | 對賽歷史層（44k dataset）、H2H override |
| **v5.4** | 2026-07-12 | Health factor、Elo 殘差校正、淨勝球偏移 |
| **v5.0** | 2026-07-06 | 定性分析層（傷病、品牌溢價、防守、海拔、戰意） |
| **v4.0** | 2026-06-15 | Ensemble 架構、Draw calibration |
| **v3.0** | 2026-06-14 | XGBoost 模型訓練 |

---

## 🧪 驗證結果

### 2026 世界盃實戰表現

| 階段 | 場次 | 命中 | 命中率 |
|:-----|:----:|:----:|:------:|
| 小組賽 | 19 | 15 | 78.9% |
| 淘汰賽 | 11 | 7 | 63.6% |
| **總計** | **30** | **22** | **73.3%** |

### v6.2 決賽重測（西班牙 vs 阿根廷）

| 指標 | v6.1 | v6.2 | 實際 |
|:-----|:----:|:----:|:----:|
| 阿根廷 xG | 2.01 | **0.73** | 0 射門 ✓ |
| 先開紀錄 | 阿根廷 | **西班牙** | 西班牙 ✓ |
| 預期黃牌 | 3.16 | **5.61** | 7+ ✓ |
| 紅牌概率 | 0.177 | **0.443** | 恩佐紅牌 ✓ |

---

## 📁 專案結構

```
worldcup-predictor/
├── src/
│   ├── predictor.py        ← 主模型（v6.2）
│   ├── daily_predict.py    ← 每日預測流水線
│   └── rebuild_elo.py      ← Elo 重建腳本
├── models/
│   ├── elo_v3.joblib       ← Elo 評分系統
│   └── xgb_weighted.joblib ← XGBoost 模型
├── data/
│   └── (資料集,需自行下載)
├── docs/                   ← 文檔
├── CHANGELOG.md            ← 版本變更記錄
└── README.md
```

---

## 🔬 關鍵函數

### `predict_ensemble()`

核心預測函數,整合 XGBoost、Elo、市場賠率三層。

**v6.2 新增參數:**
- `referee` — 裁判名稱（Vinčić / Barton）
- `tournament_stage` — 賽事階段（final / semi / quarter）
- `home_et_fatigue` / `away_et_fatigue` — 連續加時場次
- `home_health` / `away_health` — 球隊健康狀態（0-4）

**輸出欄位:**
- `ensemble_home/draw/away` — 90 分鐘概率
- `xg_home/xg_away` — 期望入球
- `ko` — 淘汰賽進程（ET/PK 概率）
- `exp_yellow` / `p_red` — 紅黃牌預測
- `v6_2` — v6.2 調試字段（崩潰風險、加時懲罰、裁判乘數）

---

## 📝 License

MIT License — 詳見 [LICENSE](LICENSE)

---

## 🙏 致謝

- **The Odds API** — 市場賠率數據
- **AndyLin31/International-Football-Results** — 44k 歷史賽果數據集
- **openfootball/worldcup.json** — 世界盃賽程數據

---

**Made with 🦞 by william-lamg**
