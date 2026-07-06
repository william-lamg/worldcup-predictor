"""
Daily World Cup Prediction v5 — 中国体彩竞彩格式
================================================
改用 竞彩SP 代替 The Odds API，输出 竞彩场次格式推荐。
"""
import sys, os, json, math, glob, re
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Auto-load .env for API keys ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from zoneinfo import ZoneInfo

HKT = ZoneInfo('Asia/Hong_Kong')
BASE = Path(__file__).resolve().parent.parent  # repo root
DATA = BASE / 'data'
MODELS = BASE / 'models'
sys.path.insert(0, str(BASE / 'src'))
from predictor import predict_ensemble, EloRatingSystem, build_features, predict_jingcai, format_jingcai_picks

# ── Load artifacts ──
print("Loading model & Elo ...")
model_path = MODELS / 'xgb_weighted.joblib'
if not model_path.exists():
    model_path = MODELS / 'xgb_v2.joblib'
if model_path.exists():
    model = joblib.load(model_path)
else:
    print("⚠️  Model file not found. Run training pipeline first.")
    model = None

elo_path = MODELS / 'elo_v3.joblib'
if not elo_path.exists():
    elo_path = MODELS / 'elo_v2.joblib'
if elo_path.exists():
    elo = joblib.load(elo_path)
else:
    print("⚠️  Elo file not found, using default EloRatingSystem()")
    elo = EloRatingSystem()

feat_csv = DATA / 'results.csv'
if feat_csv.exists():
    feat_df = pd.read_csv(feat_csv).dropna(subset=['home_score', 'away_score'])
    feat_df['home_score'] = feat_df['home_score'].astype(int)
    feat_df['away_score'] = feat_df['away_score'].astype(int)
else:
    print("⚠️  results.csv not found in data/")
    feat_df = pd.DataFrame()
feat_df['home_score'] = feat_df['home_score'].astype(int)
feat_df['away_score'] = feat_df['away_score'].astype(int)
full_feat = build_features(feat_df, elo)
print(f"Loaded: model={model_path.name}, features={full_feat.shape}")

# ── 自动从26worldcup API获取今日场次 ──
# 竞彩SP需手动填写（从澳客/中国体彩官网获取）

def get_jingcai_sp() -> dict:
    """返回竞彩SP数据，key=场次编号"""
    return {}


def get_today_matches() -> list:
    """从 26worldcup.cn API 获取今日比赛并填充场次配置
    v5.0: 每个match新增 v5_params 字段，用于传递 injury/battle_intent 等参数
    """
    today_str = datetime.now(HKT).strftime("%Y-%m-%d")
    import urllib.request
    wc26_key = os.getenv("WORLDCUP26_API_KEY", "")
    if not wc26_key:
        print("⚠️  WORLDCUP26_API_KEY not set, skipping 26worldcup.cn API")
        raise Exception("No API key")
    api_url = f"https://www.26worldcup.cn/api/v1/cup/2026/schedule?date={today_str}"
    try:
        req = urllib.request.Request(api_url, headers={"Api-Key": wc26_key})
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        matches_raw = resp.get("data", {}).get("matches", [])
        matches = []
        for i, m in enumerate(matches_raw):
            home, away = m["home_team"], m["away_team"]
            matches.append({
                "match_no": f"{i+1:03d}",
                "home": home,
                "away": away,
                "neutral": True,
                "is_wc": True,
                "is_knockout": True,
                "sp": None,
                "rangqiu": None,
                "note": f"1/8决赛" if True else "小组赛",
                # v5.0 定性参数（需手动填写伤停名单）
                "v5_params": {
                    "home_injuries": [],
                    "away_injuries": [],
                    "home_qualified": False,
                    "away_qualified": False,
                    "home_must_win": False,
                    "away_must_win": False,
                    "opponent_is_defensive": None,
                }
            })
        return matches
    except Exception as e:
        print(f"⚠️ API获取失败: {e}，用 The Odds API 手动获取今日场次")
        # Fallback: use The Odds API
        try:
            odds_api_key = os.getenv("ODDS_API_KEY", "")
            if not odds_api_key:
                print("⚠️  ODDS_API_KEY not set, skipping Odds API fallback")
                raise Exception("No API key")
            odds_url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/?apiKey={odds_api_key}&regions=eu,us&markets=h2h&oddsFormat=decimal"
            req = urllib.request.Request(odds_url, headers={"User-Agent": "Mozilla/5.0"})
            all_odds = json.loads(urllib.request.urlopen(req, timeout=10).read())
            # Filter for upcoming matches (commence_time in future)
            from datetime import timezone
            now_ts = datetime.now(timezone.utc).timestamp()
            matches = []
            for i, m in enumerate(all_odds):
                ct = m.get("commence_time", "")
                if ct:
                    try:
                        from datetime import datetime as dt2
                        ct_ts = dt2.fromisoformat(ct.replace("Z", "+00:00")).timestamp()
                    except:
                        ct_ts = 0
                else:
                    ct_ts = 0
                if ct_ts < now_ts:
                    continue  # skip matches already started
                home, away = m["home_team"], m["away_team"]
                outcomes = m.get("bookmakers", [{}])[0].get("markets", [{}])[0].get("outcomes", [])
                pm = {o.get("name", ""): o.get("price", 2.0) for o in outcomes}
                matches.append({
                    "match_no": f"{i+1:03d}",
                    "home": home,
                    "away": away,
                    "neutral": True,
                    "is_wc": True,
                    "is_knockout": True,
                    "sp": {"home": pm.get(home), "draw": pm.get("Draw"), "away": pm.get(away)},
                    "rangqiu": None,
                    "note": "1/8决赛",
                    "v5_params": {
                        "home_injuries": [],
                        "away_injuries": [],
                        "home_qualified": False,
                        "away_qualified": False,
                        "home_must_win": False,
                        "away_must_win": False,
                        "opponent_is_defensive": None,
                    }
                })
            if matches:
                return matches
        except Exception as e2:
            print(f"  Odds API也失败: {e2}")
        return []


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    matches = get_today_matches()
    if not matches:
        print("⚠️ 未配置今日场次！请先在 get_today_matches() 中填写")
        return

    # 自动竞彩SP（如果有）
    jc_sp = get_jingcai_sp()

    picks = []
    for m in matches:
        mn = m["match_no"]
        row_jc = jc_sp.get(mn, {})
        sp = m.get("sp") or row_jc.get("sp")
        rq = m.get("rangqiu") or row_jc.get("rangqiu")
        v5 = m.get("v5_params", {})

        entry = predict_jingcai(
            match_no=mn,
            home=m["home"],
            away=m["away"],
            elo=elo,
            model=model,
            feat_df=full_feat,
            sp_odds=sp,
            rangqiu_odds=rq,
            neutral=m.get("neutral", True),
            is_wc=m.get("is_wc", True),
            is_knockout=m.get("is_knockout", False),
            note=m.get("note", ""),
            # v5.0 qualitative params
            home_injuries=v5.get("home_injuries", []),
            away_injuries=v5.get("away_injuries", []),
            home_qualified=v5.get("home_qualified", False),
            away_qualified=v5.get("away_qualified", False),
            home_must_win=v5.get("home_must_win", False),
            away_must_win=v5.get("away_must_win", False),
            opponent_is_defensive=v5.get("opponent_is_defensive", None),
        )
        picks.append(entry)

    # 输出
    now_str = datetime.now(HKT).strftime("%d/%m/%Y %H:%M")
    print(f"\n{'='*60}")
    print(f"  {now_str} — 世界盃競彩預測")
    print(f"{'='*60}\n")

    print(format_jingcai_picks(picks))

    # 输出JSON
    out = {
        "updated": now_str,
        "picks": picks,
    }
    out_path = BASE / 'examples' / 'daily_prediction_data.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存至 {out_path}")


if __name__ == "__main__":
    main()
