# ATTRIBUTIONS

## 核心引擎 — goaliqlab (MIT License)

本項目嘅核心預測引擎（`src/predictor.py`）係基於 **goaliqlab** 嘅開源項目大幅修改而成。

- 原始項目：FIFA World Cup 2026 — AI Match Prediction Model
- GitHub：[goaliqlab/world-cup-2026-predictor](https://github.com/goaliqlab/world-cup-2026-predictor)
- 授權：MIT License (Copyright (c) 2026 goaliqlab)
- 原始貢獻：ElRatingSystem 框架、XGBoost 訓練 pipeline、特徵工程 (`build_features`)、Poisson Monte Carlo 模擬 (`predict_match`)、Dixon-Coles tau 函數

### 本項目嘅修改與增值

喺 goaliqlab 原始代碼基礎上，本項目進行了以下主要修改與增補：

| 版本 | 修改內容 |
|:----|:---------|
| v1.0 → v1.2 | Elo 動態更新（原版凍結不更新）、主場優勢修正、攻防統計動態化 |
| v2.0 → v3.0 | XGBoost 模型訓練（32,291 場比賽）、Draw recall 從 0.04 提升至 0.45 |
| v4.0 | Ensemble 預測器（20% XGBoost + 30% Elo + 50% 市場賠率） |
| v4.5 | 競彩賠率集成、冷熱偏差分析、讓球盤預測 |
| v4.6 → v4.7 | 初始 Elo 基於 FIFA 排名校準、Elo 修正層（10+ 支球隊） |
| v4.9 | Dixon-Coles 泊松模擬（O/U、BTTS、AH、正確比分 Top5）、淘汰賽模式權重 |
| **v5.0** | **定性分析層**（本項目核心創新）：傷病因子、品牌溢價修正、淘汰賽防守調整、海拔高度因子、戰意/出線形勢因子 |

## 數據來源

- **The Odds API**（sportradar.com）：即時市場賠率數據
- **26worldcup.cn**（世界盃數據中心）：賽程、賽果、積分榜
- **澳客網 okooo.cn**：中國競彩 SP 數據
- **騰訊新聞/人民網**：賽前前瞻、傷停情報、冷熱分析

## 其他引用

- 基於 [XGBoost](https://github.com/dmlc/xgboost) 機器學習框架（Apache 2.0）
- 彩雲天氣 API（用於天氣預警模塊）
- Python 生態：pandas、numpy、scikit-learn、joblib

---

*本項目為開源學習用途，所有數據僅用於學術研究與個人參考。*
