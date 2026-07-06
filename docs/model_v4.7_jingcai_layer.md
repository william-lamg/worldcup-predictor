# 2026-06-26 世界盃預測模型更新：競彩推薦層

## 任務
用戶要求將世界盃預測輸出格式改為「中國體彩競彩」格式，以競彩SP為準而非模型概率，同時保留 The Odds API 作核心分析。

## 修改內容

### predictor.py
- 新增函數: `predict_jingcai()` — 競彩格式預測，以SP反推市場概率為主方向
- 新增函數: `format_jingcai_picks()` — 格式化輸出
- 引入 `re_du_mismatch`（冷熱偏差）參數：>10%即反方向推薦
- 讓球盤按競彩讓球SP概率最大方向推薦（非模型主觀）
- 信心級別：🟢高(SP概率≥55%+模型認同) / 🟡中 / 🔥冷(冷熱偏差≥15%)

### predict_jingcai_2706.py
- 27/06 六場競彩格式預測完整腳本

### daily_predict.py
- 改為呼叫 predict_jingcai() 的框架

## 郵件推送
- HTML格式含6場卡牌（SP/讓球/概率/xG/推薦/冷熱/比分）
- 已成功發送至 lamzhongyan@126.com, maxcubes@yeah.net

## 核心理念
- The Odds API → Elo + XGBoost + 市場賠率 Ensemble（分析層）
- 競彩SP + 讓球盤（推薦層）
- 冷熱偏差判斷（修正層）
