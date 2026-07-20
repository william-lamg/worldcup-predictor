# worldcup-predictor — World Cup 2026 AI Match Prediction Engine

**Elo + XGBoost + Market Odds Ensemble**, 从小组赛到淘汰赛，方向正确率 **73.3%**（30 场实际比赛验证）。

> **最新版本: v6.2** — 决赛教训修正（崩溃效应、连续加时惩罚、严厉裁判、决赛 xG cap）
> 详见 [CHANGELOG.md](CHANGELOG.md)

---

## 🏗️ 架构

```
                      ┌────────────────────┐
                      │    Data Layer      │
                      │  The Odds API      │
                      │  44k Dataset       │
                      │  竞彩 SP (澳客)     │
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
                      │ │   44k 往绩层   │ │
                      │ ├── v6.1 Clutch ─┤ │
                      │ │   绝杀基因     │ │
                      │ └── v6.2 修正 ───┘ │
                      │    崩溃效应       │
                      │    加时惩罚       │
                      │    严厉裁判       │
                      │    决赛 xG cap    │
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │   Output Layer     │
                      │  HTML Reports      │
                      │  竞彩推荐格式       │
                      │  TEMP-QUIZ 栏位    │
                      └────────────────────┘
```

---

## 🔑 API 设置

本项目需要以下免费 API Key 才可以正常运行:

| API | 用途 | 申请链接 |
|:----|:----|:-------|
| **The Odds API** | 即时市场赔率 | https://the-odds-api.com/#get-access |

将 Key 填入 `.env` 文件（可复制 `.env.example`）:

```bash
cp .env.example .env
# 然后编辑 .env 填入你的 API Key
```

---

## 🚀 快速开始

```bash
pip install -r requirements.txt
cd src
python predictor.py             # 预设 demo 模式
```

### 预测单场比赛

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
    # v6.2 参数
    away_et_fatigue=3,      # 阿根廷连续 3 场 KO 加时
    away_health=2,          # 疲劳
    referee="Vinčić",       # 严厉裁判
    tournament_stage="final",
)

print(f"xG: {result['xg_home']} vs {result['xg_away']}")
print(f"晋级: H {result['ko']['advance_home']} A {result['ko']['advance_away']}")
print(f"预期黄牌: {result['exp_yellow']}")
print(f"崩溃风险: {result['v6_2']['collapse_risk_away']}")
```

---

## 📊 模型演进

| 版本 | 日期 | 核心改进 |
|:-----|:-----|:---------|
| **v6.2** | 2026-07-20 | 决赛教训修正: 崩溃效应、连续加时惩罚、严厉裁判、决赛 xG cap |
| **v6.1** | 2026-07-16 | Clutch gene（绝杀基因）、recency H2H |
| **v6.0** | 2026-07-15 | 对赛历史层（44k dataset）、H2H override |
| **v5.4** | 2026-07-12 | Health factor、Elo 残差校正、净胜球偏移 |
| **v5.0** | 2026-07-06 | 定性分析层（伤病、品牌溢价、防守、海拔、战意） |
| **v4.0** | 2026-06-15 | Ensemble 架构、Draw calibration |
| **v3.0** | 2026-06-14 | XGBoost 模型训练 |

---

## 🧪 验证结果

### 2026 世界杯实战表现

| 阶段 | 场次 | 命中 | 命中率 |
|:-----|:----:|:----:|:------:|
| 小组赛 | 19 | 15 | 78.9% |
| 淘汰赛 | 11 | 7 | 63.6% |
| **总计** | **30** | **22** | **73.3%** |

### v6.2 决赛重测（西班牙 vs 阿根廷）

| 指标 | v6.1 | v6.2 | 实际 |
|:-----|:----:|:----:|:----:|
| 阿根廷 xG | 2.01 | **0.73** | 0 射门 ✓ |
| 先开纪录 | 阿根廷 | **西班牙** | 西班牙 ✓ |
| 预期黄牌 | 3.16 | **5.61** | 7+ ✓ |
| 红牌概率 | 0.177 | **0.443** | 恩佐红牌 ✓ |

---

## 📁 项目结构

```
worldcup-predictor/
├── src/
│   ├── predictor.py        ← 主模型（v6.2）
│   ├── daily_predict.py    ← 每日预测流水线
│   └── rebuild_elo.py      ← Elo 重建脚本
├── models/
│   ├── elo_v3.joblib       ← Elo 评分系统
│   └── xgb_weighted.joblib ← XGBoost 模型
├── data/
│   └── (数据集,需自行下载)
├── docs/                   ← 文档
├── CHANGELOG.md            ← 版本变更记录
└── README.md
```

---

## 🔬 关键函数

### `predict_ensemble()`

核心预测函数, 整合 XGBoost、Elo、市场赔率三层。

**v6.2 新增参数:**
- `referee` — 裁判名称（Vinčić / Barton）
- `tournament_stage` — 赛事阶段（final / semi / quarter）
- `home_et_fatigue` / `away_et_fatigue` — 连续加时场次
- `home_health` / `away_health` — 球队健康状态（0-4）

**输出栏位:**
- `ensemble_home/draw/away` — 90 分钟概率
- `xg_home/xg_away` — 期望进球
- `ko` — 淘汰赛进程（ET/PK 概率）
- `exp_yellow` / `p_red` — 红黄牌预测
- `v6_2` — v6.2 调试字段（崩溃风险、加时惩罚、裁判乘数）

---

## 📝 License

MIT License — 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- **The Odds API** — 市场赔率数据
- **AndyLin31/International-Football-Results** — 44k 历史赛果数据集
- **openfootball/worldcup.json** — 世界杯赛程数据

---

**Author: william-lamg**

**2030 年世界杯见**
