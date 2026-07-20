# CHANGELOG — worldcup-predictor

所有重大版本變更將記錄在此文件。

格式基於 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。

---

## [v6.2] — 2026-07-20（當前版本）

### 修正（決賽教訓）
- **`health_factor()` 重寫**：tier 2 由背水效應（+15 Elo / 1.05x xG）改為崩潰效應（-50 Elo / 0.85x xG）；新增 tier 4 半隊輪換（-120 Elo / 0.50x xG）。
  決賽證明：阿根廷連續 3 場加時後 90 分鐘 0 射門，疲勞係崩潰而非背水。
- **`consecutive_et_penalty()` 新增**：接駁原本未使用嘅 `home_et_fatigue` / `away_et_fatigue` 參數；
  ≥3 場 KO 加時 xG ×0.60，≥2 場 ×0.75。
- **`finals_xg_cap()` 新增**：決賽總 xG cap 1.8 球、半決賽 2.2、八強 2.5（`tournament_stage` 參數控制）。
- **`collapse_risk()` 新增**：疲勞 + 健康輸出「崩潰概率」標籤（阿根廷決賽 0.75），僅作輸出提示唔覆蓋預測。
- **`STRICT_REFS` + `referee_card_multiplier()` 新增**：Vinčić / Barton 等嚴厲裁判紅黃牌基數 5.15、紅牌 2x（`referee` 參數）。
- **`predict_ensemble()` 新增參數**：`referee`、`tournament_stage`；輸出 `v6_2` 調試字段（et_penalty / collapse_risk / referee 乘數）。

### 驗證
- 決賽重測（西班牙 vs 阿根廷，阿根廷 et_fatigue=3 / health=2）：
  阿根廷 xG 2.01 → 0.73；先開紀錄反轉為西班牙優先；紅黃牌 3.16 → 5.61（實際 7 張 + 恩佐紅牌）；西班牙晉級 0.556 → 0.595。
- smoke test 三場（決賽 / 季軍戰 / 對照）全 pass。

### 文件
- `model_final_review_2026_worldcup.md`（最終復盤 + v6.2 設計）
- `worldcup_v62_implementation_20260720.md`（實作記錄）

---

## [v6.1] — 2026-07-16

### 新增
- **`clutch_gene_factor()`**：近 5 場大賽 1-0 / 0-1 絕殺傾向（xG 0.94–1.08）。
- **`wc2026_clutch_factor()`**：WC-2026 專用 clutch 層（阿根廷 1.06 / 西班牙 1.01）。
- **`health_factor()` tier 2 背水效應**（+15 Elo / 1.05x xG）—— ⚠️ 此方向於 v6.2 被推翻（見上）。
- **`matchup_h2h_probs()` recency 加權**：近 5 場 2x、6–10 場 1.5x。
- **`predict_ensemble()` 新增參數**：`home_et_fatigue` / `away_et_fatigue` / `use_clutch_gene` / `h2h_recency_weighted`。
- 修正 `simulate_knockout_progression` 接收 clutch 調整後 xG。

### 驗證
- smoke test 4/4 淘汰賽方向全中。

---

## [v6.0] — 2026-07-15

### 新增
- **對賽歷史層（H2H Override）**：`matchup_h2h_probs()` 近 10 場 window，recency 加權；
  `H2H_OVERRIDE` 手動 curated 往績修正「剋星」pattern（如西班牙 vs 法國 7-1-2）。
- **數據源升級至 44k dataset**：AndyLin31/International-Football-Results（1872–2023，44,762 場），取代原本 32k feat_df。
- H2H 混合權重 ≤0.15（只輕推唔覆蓋 ensemble）。

### 驗證
- 法國 vs 西班牙：v5.4 法國 51.3% 晉級 → v6.0 西班牙 52.1% 晉級（方向與實際 2-0 一致）。
- 阿根廷 vs 英格蘭、巴西 vs 日本均正常運行。

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
