# CHANGELOG — worldcup-predictor

所有重大版本變更將記錄在此文件。

格式基於 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。

---

## [v5.0] — 2026-07-06

### 新增
- **定性分析層（5 個模塊）**
  - `injury_factor()`：傷病 Elo/xG 調整（三級制：-80/-50/-20 Elo，xG 0.75~0.93 折）
  - `brand_premium_corrector()`：品牌溢價修正（巴西 -18%、阿根廷 -12%、德國 -15% 等）
  - `knockout_defense_adjuster()`：淘汰賽防守隊 xG 折讓（維德角 -28%、巴拉圭 -25%）
  - `altitude_factor()`：海拔高度客隊懲罰（>1,500m → -80 Elo）
  - `battle_intent_factor()`：小組賽戰意調整（已出線 -60 Elo，生死戰 +25 Elo）
- `predict_ensemble()` 新增 9 個可選參數：`home_injuries`、`away_injuries`、`home_qualified`、`away_qualified`、`home_must_win`、`away_must_win`、`opponent_is_defensive`、`disable_brand_correction`、`disable_altitude`
- `predict_jingcai()` 同步更新，向前傳遞 v5.0 參數
- `daily_predict.py` 新增 v5_params 字典傳遞，支援自動從 The Odds API 獲取今日場次

---

## [v4.9] — 2026-06-29

### 新增
- Dixon-Coles tau 函數
- Monte Carlo 泊松模擬（O/U 2.5、BTTS、亞洲讓球、正确比分 Top5）
- 淘汰賽模式權重（ML 15% / Elo 20% / 市場 65%）
- `predict_ensemble()` 回傳衍生市場輸出（over_15~over_35, btts_yes/no, ah_*）

### 文件
- `model_v49_dixon_coles_20260628.md`

---

## [v4.8] — 2026-06-28

### 新增
- Elo V3 重建（使用全部 144 場小組賽賽果，336 支球隊 Elo 即時更新）
- `rebuild_elo.py`：組賽結果餵入腳本
- 淘汰賽模式參數 `is_knockout` 集成到整個流水線
- 移除舊 Elo 修正層（所有修正已 bake into elo_v3）

---

## [v4.7] — 2026-06-25

### 新增
- 初始 Elo 基於 FIFA 排名校準
- Elo 修正層（10+ 支球隊：美國+150、厄瓜多爾-200、日本-150 等）
- `model_correction_v4.7_26062025.md`

### 文件
- `model_v4.7_elo_correction.md`

---

## [v4.6] — 2026-06-25

### 新增
- 競彩 SP fallback（當 The Odds API 無數據）
- 冷熱偏差分析（大眾高估主隊 >10% 即反推）
- 手動修正波黑 vs 卡塔爾（跟市場 @2.60）

---

## [v4.5] — 2026-06-22

### 新增
- 定性新聞層（傷病/紅牌/門將送禮吸收）
- 和局概率加權 +3~5%
- 大小球風險系數
- 每日賽程自動讀取
- 競彩賠率分歧 >10% 二次校準

### 修復
- `parse_commence_time()` pytz 缺失問題（替換為 zoneinfo）
- Market odds 鍵錯誤（h2h_home/h2h_draw/h2h_away）

---

## [v4.0] — 2026-06-15

### 新增
- **Ensemble 預測器**（權重：20% XGBoost + 30% Elo + 50% 市場賠率）
- Draw calibration（無市場賠率時）
- xG 上限 3.5 cap
- `predict_ensemble()` 函數
- HTML 報告格式（卡片式 7 項元素）

### 修復
- Draw recall 0.46（加權訓練修復）
- Elo V2：分層初始值、降低資格賽 K 值、Bayesian shrinkage

---

## [v3.0] — 2026-06-14

### 新增
- XGBoost 模型訓練（32,291 場比賽）
- CV accuracy 57.9% ± 0.1%
- Draw recall 提升至 0.45（class weighting + hyperparameter tuning）
- `xgb_weighted.joblib` 模型文件

---

## [v2.0] — 2026-06-14

### 新增
- Elo + Poisson + Monte Carlo 預測引擎
- 三大缺口修復：Elo 永不更新 → 動態更新
- 主場優勢修正
- 攻防統計動態化

---

## [v1.0 → v1.2] — 2026-06-12 ~ 06-14

### 新增
- 基於 goaliqlab 原始碼 fork
- 彩雲天氣 API 集成
- 三通道推送（郵箱、QQ、桌面 PDF）
- 16 個運動技能分類審查
- 首次世界盃預測：墨西哥 vs 南非、韓國 vs 捷克
