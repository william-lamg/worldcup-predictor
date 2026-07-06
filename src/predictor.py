from __future__ import annotations

# ── Auto-load .env for API keys ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

"""
FIFA World Cup 2026 – Match Prediction Model
=============================================
A complete, production-ready ML pipeline for predicting match outcomes.

Architecture:
  1. Data layer   – historical World Cup + international match data
  2. Feature eng  – Elo ratings, recent form, head-to-head, tournament context
  3. Model layer  – XGBoost classifier (home win / draw / away win)
  4. Prediction   – probabilities + expected goals for any upcoming match
  5. Visualisation– charts ready to drop into your YouTube thumbnails/videos

Usage:
  pip install -r requirements.txt
  python predictor.py                    # demo mode: predict several WC26 matches
  python predictor.py --match "Brazil" "France"
  python predictor.py --simulate         # full tournament Monte Carlo simulation
"""

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
VIS_DIR   = Path(__file__).parent / "visuals"
for d in [DATA_DIR, MODEL_DIR, VIS_DIR]:
    d.mkdir(exist_ok=True)

# Free dataset: international results 1872–present (Kaggle / GitHub mirror)
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# FIFA World Cup 2026 group stage schedule (openfootball public domain data)
WC2026_MATCHES = [
    # ── GROUP A: Mexico · South Africa · South Korea · Czechia ──
    ("Mexico",        "South Africa",          "2026-06-11"),
    ("South Korea",   "Czechia",               "2026-06-11"),
    ("Mexico",        "Czechia",               "2026-06-17"),
    ("South Korea",   "South Africa",          "2026-06-17"),
    ("Mexico",        "South Korea",           "2026-06-22"),
    ("Czechia",       "South Africa",          "2026-06-22"),

    # ── GROUP B: Canada · Switzerland · Qatar · Bosnia and Herzegovina ──
    ("Canada",                    "Bosnia and Herzegovina", "2026-06-12"),
    ("Qatar",                     "Switzerland",            "2026-06-13"),
    ("Canada",                    "Qatar",                  "2026-06-18"),
    ("Bosnia and Herzegovina",    "Switzerland",            "2026-06-18"),
    ("Canada",                    "Switzerland",            "2026-06-23"),
    ("Bosnia and Herzegovina",    "Qatar",                  "2026-06-23"),

    # ── GROUP C: Brazil · Morocco · Haiti · Scotland ──
    ("Brazil",        "Morocco",               "2026-06-13"),
    ("Haiti",         "Scotland",              "2026-06-13"),
    ("Brazil",        "Scotland",              "2026-06-18"),
    ("Morocco",       "Haiti",                 "2026-06-19"),
    ("Brazil",        "Haiti",                 "2026-06-23"),
    ("Scotland",      "Morocco",               "2026-06-24"),

    # ── GROUP D: USA · Paraguay · Australia · Turkey ──
    ("USA",           "Paraguay",              "2026-06-12"),
    ("Australia",     "Turkey",                "2026-06-13"),
    ("USA",           "Australia",             "2026-06-19"),
    ("Paraguay",      "Turkey",                "2026-06-19"),
    ("USA",           "Turkey",                "2026-06-25"),
    ("Paraguay",      "Australia",             "2026-06-25"),

    # ── GROUP E: Germany · Curacao · Ivory Coast · Ecuador ──
    ("Germany",       "Curacao",               "2026-06-14"),
    ("Ivory Coast",   "Ecuador",               "2026-06-14"),
    ("Germany",       "Ivory Coast",           "2026-06-20"),
    ("Ecuador",       "Curacao",               "2026-06-20"),
    ("Germany",       "Ecuador",               "2026-06-25"),
    ("Ivory Coast",   "Curacao",               "2026-06-25"),

    # ── GROUP F: Netherlands · Japan · Sweden · Tunisia ──
    ("Netherlands",   "Japan",                 "2026-06-14"),
    ("Sweden",        "Tunisia",               "2026-06-14"),
    ("Netherlands",   "Sweden",                "2026-06-20"),
    ("Tunisia",       "Japan",                 "2026-06-20"),
    ("Japan",         "Sweden",                "2026-06-25"),
    ("Tunisia",       "Netherlands",           "2026-06-25"),

    # ── GROUP G: Belgium · Egypt · Iran · New Zealand ──
    ("Belgium",       "Egypt",                 "2026-06-15"),
    ("Iran",          "New Zealand",           "2026-06-15"),
    ("Belgium",       "Iran",                  "2026-06-21"),
    ("New Zealand",   "Egypt",                 "2026-06-21"),
    ("Egypt",         "Iran",                  "2026-06-26"),
    ("New Zealand",   "Belgium",               "2026-06-26"),

    # ── GROUP H: Spain · Cape Verde · Saudi Arabia · Uruguay ──
    ("Spain",         "Cape Verde",            "2026-06-15"),
    ("Saudi Arabia",  "Uruguay",               "2026-06-15"),
    ("Spain",         "Saudi Arabia",          "2026-06-21"),
    ("Uruguay",       "Cape Verde",            "2026-06-21"),
    ("Spain",         "Uruguay",               "2026-06-26"),
    ("Cape Verde",    "Saudi Arabia",          "2026-06-26"),

    # ── GROUP I: France · Senegal · Iraq · Norway ──
    ("France",        "Senegal",               "2026-06-16"),
    ("Iraq",          "Norway",                "2026-06-16"),
    ("France",        "Iraq",                  "2026-06-22"),
    ("Norway",        "Senegal",               "2026-06-22"),
    ("France",        "Norway",                "2026-06-26"),
    ("Senegal",       "Iraq",                  "2026-06-26"),

    # ── GROUP J: Argentina · Algeria · Austria · Jordan ──
    ("Argentina",     "Algeria",               "2026-06-16"),
    ("Austria",       "Jordan",                "2026-06-16"),
    ("Argentina",     "Jordan",                "2026-06-22"),
    ("Algeria",       "Austria",               "2026-06-22"),
    ("Argentina",     "Austria",               "2026-06-26"),
    ("Jordan",        "Algeria",               "2026-06-26"),

    # ── GROUP K: Portugal · DR Congo · Uzbekistan · Colombia ──
    ("Portugal",      "DR Congo",              "2026-06-17"),
    ("Uzbekistan",    "Colombia",              "2026-06-17"),
    ("Portugal",      "Uzbekistan",            "2026-06-23"),
    ("Colombia",      "DR Congo",              "2026-06-23"),
    ("Portugal",      "Colombia",              "2026-06-27"),
    ("DR Congo",      "Uzbekistan",            "2026-06-27"),

    # ── GROUP L: England · Croatia · Ghana · Panama ──
    ("England",       "Croatia",               "2026-06-17"),
    ("Ghana",         "Panama",                "2026-06-17"),
    ("England",       "Ghana",                 "2026-06-23"),
    ("Croatia",       "Panama",                "2026-06-23"),
    ("England",       "Panama",                "2026-06-27"),
    ("Croatia",       "Ghana",                 "2026-06-27"),
]


# ──────────────────────────────────────────────
# STEP 1 – DATA LOADING
# ──────────────────────────────────────────────

def load_results(max_rows: int = 50_000) -> pd.DataFrame:
    """
    Load historical international match results.
    Falls back to synthetic data if the network is unavailable
    (useful for offline testing / sandboxed environments).
    """
    cache = DATA_DIR / "results.csv"
    if cache.exists():
        print("📂  Loading cached match history …")
        df = pd.read_csv(cache)
    else:
        print("📡  Downloading international results dataset …")
        try:
            df = pd.read_csv(RESULTS_URL)
            df.to_csv(cache, index=False)
            print(f"    Saved {len(df):,} matches to cache.")
        except Exception as e:
            print(f"    Network unavailable ({e}). Generating synthetic dataset …")
            df = _synthetic_results(n=max_rows)
            df.to_csv(cache, index=False)

    df["date"] = pd.to_datetime(df["date"])
    # Keep only matches from 1990 onward (modern football)
    df = df[df["date"] >= "1990-01-01"].copy()
    # Drop rows with missing scores (NaN) — real dataset has some future/cancelled matches
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        print(f"    Dropped {dropped} rows with missing scores.")
    print(f"    {len(df):,} matches loaded (1990–present).")
    return df


def _synthetic_results(n: int = 50_000) -> pd.DataFrame:
    """
    Generates realistic-looking synthetic match data so the full pipeline
    runs even in air-gapped / sandboxed environments.
    """
    rng = np.random.default_rng(42)
    teams = [
        "Brazil","Germany","France","Argentina","Spain","England","Italy",
        "Netherlands","Portugal","Belgium","Croatia","Uruguay","Mexico",
        "USA","Japan","South Korea","Colombia","Chile","Denmark","Sweden",
        "Switzerland","Poland","Senegal","Nigeria","Egypt","Morocco",
        "Saudi Arabia","Iran","Australia","Canada","South Africa","Panama",
    ]
    strengths = {t: rng.uniform(0.3, 1.0) for t in teams}
    # Make big teams stronger
    for t, v in [("Brazil",1.0),("Germany",0.95),("France",0.95),
                 ("Argentina",0.93),("Spain",0.90),("England",0.87)]:
        strengths[t] = v

    rows = []
    dates = pd.date_range("1990-01-01", "2026-05-01", periods=n)
    tournaments = ["FIFA World Cup","Friendly","Copa America","Euro","Gold Cup","Qualifier"]
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = strengths[h] * 1.5 + 0.2  # slight home advantage
        lam_a = strengths[a] * 1.3
        gh = int(rng.poisson(lam_h))
        ga = int(rng.poisson(lam_a))
        rows.append({
            "date":       dates[i].strftime("%Y-%m-%d"),
            "home_team":  h,
            "away_team":  a,
            "home_score": gh,
            "away_score": ga,
            "tournament": rng.choice(tournaments, p=[0.08,0.5,0.08,0.08,0.08,0.18]),
            "neutral":    rng.random() > 0.6,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# STEP 2 – ELO RATING ENGINE
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# POISSON + DIXON-COLES HELPER FUNCTIONS (v4.9)
# ──────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    """Pure Python Poisson PMF — P(X=k) for mean=lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def dixon_coles_tau(x: int, y: int, rho: float = -0.07) -> float:
    """
    Dixon-Coles low-scoring correlation adjustment.
    
    Corrects the well-known bias where independent Poisson models
    under-estimate 0-0 and 1-0/0-1 scorelines in football.
    
    rho ≈ -0.07 (empirically estimated from European league data).
    When rho < 0: 0-0 and 1-0/0-1 probabilities are adjusted upward.
    """
    if x == 0 and y == 0:
        return 1 - rho          # ~1.07 → 0-0 gets +7% boost
    if (x == 0 and y == 1) or (x == 1 and y == 0):
        return 1 + rho * 0.5    # ~0.965 → slight dampener on 1-0/0-1
    return 1.0


def match_probs_from_xg(xg_home: float, xg_away: float, max_goals: int = 15) -> dict:
    """
    Compute full match probs from xG using Poisson + Dixon-Coles tau.
    
    Returns dict with:
      1X2, O/U 1.5/2.5/3.5, BTTS, Asian Handicap -0.5/-0.25/-0.75,
      correct score top 5
    """
    # Precompute Poisson PMFs
    pmf_h = [poisson_pmf(g, xg_home) for g in range(max_goals)]
    pmf_a = [poisson_pmf(g, xg_away) for g in range(max_goals)]

    total = home_wins = draws = away_wins = btts_yes = 0.0
    goal_diff_probs: dict[int, float] = {}
    total_goal_probs: dict[int, float] = {}
    score_probs: dict[str, float] = {}

    for h in range(max_goals):
        for a in range(max_goals):
            prob = pmf_h[h] * pmf_a[a] * dixon_coles_tau(h, a)
            total += prob

            if h > a:      home_wins += prob
            elif h == a:   draws     += prob
            else:          away_wins += prob

            if h > 0 and a > 0:
                btts_yes += prob

            gd = h - a
            goal_diff_probs[gd] = goal_diff_probs.get(gd, 0.0) + prob

            tg = h + a
            total_goal_probs[tg] = total_goal_probs.get(tg, 0.0) + prob

            score_probs[f"{h}-{a}"] = prob

    # Normalise
    home_wins /= total; draws /= total; away_wins /= total
    btts_yes /= total

    # O/U
    def cum_goal(threshold: float) -> float:
        return sum(p for tg, p in total_goal_probs.items() if tg > threshold) / total

    # Asian Handicap
    def cum_diff(threshold: float) -> float:
        return sum(p for gd, p in goal_diff_probs.items() if gd > threshold) / total
    def cum_diff_under(threshold: float) -> float:
        return sum(p for gd, p in goal_diff_probs.items() if gd < threshold) / total

    # Correct score top 5
    sorted_scores = sorted(score_probs.items(), key=lambda x: -x[1])[:5]
    correct_score = [(s, round(p / total, 4)) for s, p in sorted_scores]

    return {
        "home_win":       round(home_wins, 4),
        "draw":           round(draws, 4),
        "away_win":       round(away_wins, 4),
        "over_15":        round(cum_goal(1.5), 4),
        "over_25":        round(cum_goal(2.5), 4),
        "over_35":        round(cum_goal(3.5), 4),
        "btts_yes":       round(btts_yes, 4),
        "btts_no":        round(1 - btts_yes, 4),
        "ah_home_05":     round(cum_diff(0.5), 4),
        "ah_home_025":    round(cum_diff(0.25), 4),
        "ah_home_075":    round(cum_diff(0.75), 4),
        "ah_away_05":     round(cum_diff_under(-0.5), 4),
        "ah_away_025":    round(cum_diff_under(-0.25), 4),
        "ah_away_075":    round(cum_diff_under(-0.75), 4),
        "correct_score":  correct_score,
    }


class EloRatingSystem:
    """
    Dynamic Elo rating system for national teams.
    Ratings update after every match and weight tournament matches
    more heavily than friendlies.

    V2 improvements:
      - Tiered initial Elo (stronger teams start higher)
      - Reduced WCQ K-factor (prevent violent ranking swings)
      - Bayesian shrinkage for low-match-count teams
    """

    # Tournament importance multipliers (k-factor weights)
    TOURNAMENT_WEIGHTS = {
        "FIFA World Cup":          60,
        "Copa America":            45,
        "UEFA Euro":               45,
        "Africa Cup of Nations":   45,
        "CONCACAF Gold Cup":       35,
        "Asian Cup":               35,
        "World Cup Qualification": 20,    # reduced from 30 → prevent extreme swings
        "Friendly":                20,
    }
    DEFAULT_K = 25
    INITIAL_ELO = 1500
    MEAN_ELO = 1700  # shrinkage target

    # Tiered starting Elo for historically strong teams
    TIER_ELO = {
        # Tier S (historically elite)
        "Brazil": 2000, "Argentina": 2000, "France": 2000,
        "Germany": 2000, "Italy": 2000, "England": 2000,
        "Spain": 2000, "Netherlands": 2000,
        # Tier A (strong regulars)
        "Portugal": 1900, "Belgium": 1900, "Croatia": 1900,
        "Uruguay": 1900, "Colombia": 1900,
        # Tier B (solid / occasional contenders)
        "Denmark": 1850, "Sweden": 1850, "Switzerland": 1850,
        "Japan": 1850, "South Korea": 1850, "Mexico": 1850,
        "USA": 1850, "Nigeria": 1850, "Senegal": 1850,
        "Morocco": 1850, "Egypt": 1850,
        "Serbia": 1800, "Poland": 1800, "Ukraine": 1800,
        "Turkey": 1800, "Austria": 1800, "Czechia": 1800,
        "Norway": 1800, "Scotland": 1800, "Romania": 1800,
        "Iran": 1800, "Australia": 1800, "Russia": 1800,
        "Chile": 1800, "Peru": 1800,
        # Tier C (emerging / mid-level)
        "Ecuador": 1700, "Paraguay": 1700, "Venezuela": 1700,
        "Costa Rica": 1700, "Panama": 1700, "Canada": 1700,
        "Ghana": 1700, "Cameroon": 1700, "Ivory Coast": 1700,
        "South Africa": 1700, "Mali": 1700,
        "Saudi Arabia": 1700, "Iraq": 1700, "Qatar": 1700,
        "Uzbekistan": 1700, "Algeria": 1750,
        "Tunisia": 1700,
    }

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.match_counts: dict[str, int] = {}
        self.history: dict[str, list] = {}

    def _get_rating(self, team: str) -> float:
        """Get rating with lazy init (pickle-friendly)."""
        if team not in self.ratings:
            self.ratings[team] = self._get_initial(team)
        return self.ratings[team]

    def _get_initial(self, team: str) -> float:
        """Get tiered initial Elo if available, else default 1500."""
        return self.TIER_ELO.get(team, self.INITIAL_ELO)

    def _k(self, tournament: str) -> float:
        for key, k in self.TOURNAMENT_WEIGHTS.items():
            if key.lower() in tournament.lower():
                return k
        return self.DEFAULT_K

    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home: str, away: str, gh: int, ga: int,
               tournament: str, neutral: bool):
        ra = self._get_rating(home)
        rb = self._get_rating(away)
 # Home advantage: +100 Elo unless neutral venue
        if not neutral:
            ra += 100
        ea = self.expected(ra, rb)

        if gh > ga:
            sa = 1.0
        elif gh == ga:
            sa = 0.5
        else:
            sa = 0.0

        # Goal difference multiplier (margin of victory)
        gd = abs(gh - ga)
        gd_mult = math.log(max(gd, 1) + 1) + (1 if gd > 1 else 0)

        k = self._k(tournament) * gd_mult
        delta = k * (sa - ea)

        self.ratings[home] += delta
        self.ratings[away] -= delta
        self.match_counts[home] = self.match_counts.get(home, 0) + 1
        self.match_counts[away] = self.match_counts.get(away, 0) + 1
        self.history.setdefault(home, []).append(self.ratings[home])
        self.history.setdefault(away, []).append(self.ratings[away])

    def fit(self, df: pd.DataFrame) -> "EloRatingSystem":
        df_sorted = df.sort_values("date")
        for _, row in df_sorted.iterrows():
            self.update(
                row["home_team"], row["away_team"],
                int(row["home_score"]), int(row["away_score"]),
                str(row.get("tournament", "Friendly")),
                bool(row.get("neutral", False)),
            )
        # Apply Bayesian shrinkage: pull low-match-count teams toward mean
        self._apply_shrinkage()
        return self

    def _apply_shrinkage(self, shrinkage_strength: int = 30):
        """
        Shrink ratings of teams with few matches toward MEAN_ELO.
        A team with 0 matches gets pulled 100% to mean.
        A team with 30+ matches gets negligible pull.
        """
        for team in list(self.ratings.keys()):
            n = self.match_counts.get(team, 0)
            if n < shrinkage_strength:
                weight = 1.0 - (n / shrinkage_strength)
                self.ratings[team] += (self.MEAN_ELO - self.ratings[team]) * weight

    def top_n(self, n: int = 20) -> pd.DataFrame:
        return (
            pd.DataFrame(self.ratings.items(), columns=["team", "elo"])
            .sort_values("elo", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ──────────────────────────────────────────────
# STEP 3 – FEATURE ENGINEERING
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame, elo: EloRatingSystem) -> pd.DataFrame:
    """
    Build a feature matrix from historical match data.
    Each row = one match, with features computed at the time of the match.
    """
    print("⚙️   Engineering features …")

    # Rebuild Elo game-by-game to get snapshot ratings at match time
    # Use tiered initial Elo from the pre-fitted system
    def _snap_init(): return 1500  # fallback if elo not yet fitted for a team
    ratings_snap: dict[str, float] = defaultdict(lambda: _snap_init())
    # Copy over the elo system's initial tiered ratings
    for team in set(df['home_team']).union(set(df['away_team'])):
        ratings_snap[team] = elo._get_initial(team)
    history = []

    for _, row in df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        gh, ga = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", "Friendly"))

        elo_h = ratings_snap[home]
        elo_a = ratings_snap[away]
        elo_diff = elo_h - elo_a + (0 if neutral else 100)

        # Outcome label: 0=away win, 1=draw, 2=home win
        if gh > ga:
            outcome = 2
        elif gh == ga:
            outcome = 1
        else:
            outcome = 0

        history.append({
            "date":           row["date"],
            "home_team":      home,
            "away_team":      away,
            "home_score":     gh,
            "away_score":     ga,
            "elo_home":       elo_h,
            "elo_away":       elo_a,
            "elo_diff":       elo_diff,
            "neutral":        int(neutral),
            "is_wc":          int("world cup" in tournament.lower()),
            "outcome":        outcome,
        })

        # Update snapshot ratings
        elo_obj = EloRatingSystem()
        elo_obj.ratings = dict(ratings_snap)
        elo_obj.update(home, away, gh, ga, tournament, neutral)
        ratings_snap[home] = elo_obj.ratings[home]
        ratings_snap[away] = elo_obj.ratings[away]

    feat_df = pd.DataFrame(history)

    # Rolling form: avg goals scored/conceded in last 5 matches (per team)
    feat_df = _add_rolling_form(feat_df)

    # Head-to-head record (last 10 meetings)
    feat_df = _add_h2h(feat_df)

    print(f"    Feature matrix: {feat_df.shape}")
    return feat_df


def _add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling avg goals scored and conceded for each team."""
    scored_h, conceded_h = {}, {}
    scored_a, conceded_a = {}, {}

    rows = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        gh, ga = row["home_score"], row["away_score"]

        h_scored   = np.mean(scored_h.get(h, [1.5])[-window:])
        h_conceded = np.mean(conceded_h.get(h, [1.2])[-window:])
        a_scored   = np.mean(scored_a.get(a, [1.5])[-window:])
        a_conceded = np.mean(conceded_a.get(a, [1.2])[-window:])

        row = row.copy()
        row["home_form_scored"]   = h_scored
        row["home_form_conceded"] = h_conceded
        row["away_form_scored"]   = a_scored
        row["away_form_conceded"] = a_conceded
        row["form_diff"]          = (h_scored - h_conceded) - (a_scored - a_conceded)
        rows.append(row)

        scored_h.setdefault(h, []).append(gh)
        conceded_h.setdefault(h, []).append(ga)
        scored_a.setdefault(a, []).append(ga)
        conceded_a.setdefault(a, []).append(gh)

    return pd.DataFrame(rows)


def _add_h2h(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add head-to-head win rate for home team vs away team."""
    h2h: dict[tuple, list] = defaultdict(list)
    h2h_rates = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        key  = tuple(sorted([h, a]))
        past = h2h[key][-window:]

        if past:
            # From home team perspective
            h2h_rate = sum(1 for r in past if r == h) / len(past)
        else:
            h2h_rate = 0.5  # no history → neutral

        h2h_rates.append(h2h_rate)

        winner = h if row["home_score"] > row["away_score"] else \
                 (a if row["away_score"] > row["home_score"] else "draw")
        h2h[key].append(winner)

    df = df.copy()
    df["h2h_home_winrate"] = h2h_rates
    return df


# ──────────────────────────────────────────────
# STEP 4 – MODEL TRAINING
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff",
    "elo_home",
    "elo_away",
    "neutral",
    "is_wc",
    "home_form_scored",
    "home_form_conceded",
    "away_form_scored",
    "away_form_conceded",
    "form_diff",
    "h2h_home_winrate",
]


def train_model(feat_df: pd.DataFrame) -> xgb.XGBClassifier:
    print("🤖  Training XGBoost classifier …")

    X = feat_df[FEATURE_COLS].fillna(0)
    y = feat_df["outcome"]  # 0=away, 1=draw, 2=home

    # ── Class imbalance fix ──
    # Compute inverse-frequency weights so draws get equal importance
    class_counts = np.bincount(y)
    n_classes = len(class_counts)
    n_total = len(y)
    # Balanced weight per class: weight inversely proportional to frequency
    class_weights = {c: n_total / (n_classes * class_counts[c]) for c in range(n_classes)}
    sample_weight = np.array([class_weights[yi] for yi in y])

    print(f"    Class distribution: 0(Away)={class_counts[0]}, 1(Draw)={class_counts[1]}, 2(Home)={class_counts[2]}")
    print(f"    Class weights:      0={class_weights[0]:.2f}, 1(Draw)={class_weights[1]:.2f}, 2={class_weights[2]:.2f}")
    print(f"    Weight ratio draw/away: {class_weights[1]/class_weights[0]:.1f}x")

    model = xgb.XGBClassifier(
        n_estimators=600,           # increased for better convergence with weights
        max_depth=5,                # bumped from 4 to capture draw nuance
        learning_rate=0.03,         # lower LR for more epochs with weighted data
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,         # prevent overfitting on weighted draws
        reg_lambda=2.0,             # L2 regularisation for stability
        reg_alpha=0.5,              # L1 to prune noisy features
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation (without sample_weight for CV, as skl CV doesn't use it)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use a separate fit for CV scoring without weights to get comparable accuracy
    cv_model = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=2.0, reg_alpha=0.5,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
    print(f"    CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train final model with sample weights
    model.fit(X, y, sample_weight=sample_weight)
    y_pred = model.predict(X)
    print("\n    Training set report (weighted training):")
    print(classification_report(y, y_pred,
                                target_names=["Away win", "Draw", "Home win"],
                                zero_division=0))
    
    # Show predicted class distribution
    pred_counts = np.bincount(y_pred, minlength=3)
    print(f"    Predicted distribution: Away={pred_counts[0]}, Draw={pred_counts[1]}, Home={pred_counts[2]}")
    
    return model


# ──────────────────────────────────────────────
# STEP 5 – PREDICTION ENGINE
# ──────────────────────────────────────────────

def predict_match(
    home: str,
    away: str,
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    neutral: bool = True,
    is_wc: bool = True,
) -> dict:
    """
    Predict probabilities and expected goals for a single match.
    Returns a dict with full prediction details.
    """

    def team_form(team: str, window: int = 5) -> tuple[float, float]:
        team_matches = feat_df[
            (feat_df["home_team"] == team) | (feat_df["away_team"] == team)
        ].tail(window * 2)
        scored, conceded = [], []
        for _, r in team_matches.iterrows():
            if r["home_team"] == team:
                scored.append(r["home_score"]); conceded.append(r["away_score"])
            else:
                scored.append(r["away_score"]); conceded.append(r["home_score"])
        return (
            np.mean(scored[-window:])   if scored   else 1.5,
            np.mean(conceded[-window:]) if conceded else 1.2,
        )

    def h2h_rate(home: str, away: str, window: int = 10) -> float:
        mask = (
            ((feat_df["home_team"] == home) & (feat_df["away_team"] == away)) |
            ((feat_df["home_team"] == away) & (feat_df["away_team"] == home))
        )
        past = feat_df[mask].tail(window)
        if past.empty:
            return 0.5
        wins = sum(
            1 for _, r in past.iterrows()
            if (r["home_team"] == home and r["home_score"] > r["away_score"]) or
               (r["away_team"] == home and r["away_score"] > r["home_score"])
        )
        return wins / len(past)

    elo_h = elo.ratings.get(home, EloRatingSystem.INITIAL_ELO)
    elo_a = elo.ratings.get(away, EloRatingSystem.INITIAL_ELO)
    elo_diff = elo_h - elo_a + (0 if neutral else 100)

    h_sc, h_cc = team_form(home)
    a_sc, a_cc = team_form(away)

    X = pd.DataFrame([{
        "elo_diff":             elo_diff,
        "elo_home":             elo_h,
        "elo_away":             elo_a,
        "neutral":              int(neutral),
        "is_wc":                int(is_wc),
        "home_form_scored":     h_sc,
        "home_form_conceded":   h_cc,
        "away_form_scored":     a_sc,
        "away_form_conceded":   a_cc,
        "form_diff":            (h_sc - h_cc) - (a_sc - a_cc),
        "h2h_home_winrate":     h2h_rate(home, away),
    }])

    probs = model.predict_proba(X)[0]  # [away, draw, home]

    # Expected goals via Dixon-Coles-style attack/defence balance
    # Using form-based lambda with Elo adjustment
    elo_factor = 10 ** (elo_diff / 800)
    xg_home = max(0.3, h_sc * elo_factor * 0.7 + (1 - a_cc / 2) * 0.3)
    xg_away = max(0.3, a_sc / elo_factor * 0.7 + (1 - h_cc / 2) * 0.3)

    # Cap xG to prevent extreme inflation from Poisson lambda
    XG_CAP = 3.5
    xg_home = min(xg_home, XG_CAP)
    xg_away = min(xg_away, XG_CAP)

    # v4.9 — Poisson + Dixon-Coles derived market from xG
    try:
        dc_probs = match_probs_from_xg(xg_home, xg_away)
        # Override 1X2 with DC-adjusted if confidence is high (close xG)
        if abs(xg_home - xg_away) < 1.0:
            dc_adj = True
        else:
            dc_adj = False
    except Exception:
        dc_probs = {}
        dc_adj = False

    return {
        "home":      home,
        "away":      away,
        "elo_home":  round(elo_h, 0),
        "elo_away":  round(elo_a, 0),
        "p_home":    round(float(probs[2]), 3),
        "p_draw":    round(float(probs[1]), 3),
        "p_away":    round(float(probs[0]), 3),
        "xg_home":   round(float(xg_home), 2),
        "xg_away":   round(float(xg_away), 2),
        "favorite":  home if probs[2] > probs[0] else away,
        # v4.9: Dixon-Coles derived market probs
        "dc":        dc_probs,
        "dc_adjusted": dc_adj,
    }


# ──────────────────────────────────────────────
# UTILITY – FETCH MARKET ODDS
# ──────────────────────────────────────────────


import urllib.request
import json as _json
import os

_ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

def fetch_wc_odds(api_key: str = "") -> list[dict]:
    if not api_key:
        api_key = _ODDS_API_KEY or os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("⚠️  ODDS_API_KEY not set. Set it in .env or as environment variable.")
        return []
    """
    Fetch all World Cup match odds from The Odds API.
    Returns list of dicts with home_team, away_team, h2h odds.
    """
    url = (
        f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
        f"/?apiKey={api_key}&regions=eu,us&markets=h2h&oddsFormat=decimal"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
    except Exception:
        return []

    results = []
    for match in data:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        outcomes = match.get("bookmakers", [{}])[0].get("markets", [{}])[0].get("outcomes", [])
        # The Odds API returns outcomes as [team1, team2, Draw] in alphabetical order
        # Map by matching outcome name to home/away team
        price_map = {o.get("name", ""): o.get("price", 2.0) for o in outcomes}
        if "Draw" in price_map:
            home_price = price_map.get(home)
            away_price = price_map.get(away)
            draw_price = price_map["Draw"]
            if home_price and away_price:
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "h2h_home": float(home_price),
                    "h2h_draw": float(draw_price),
                    "h2h_away": float(away_price),
                })
    return results


def match_odds(home: str, away: str, odds_list: list[dict]) -> dict | None:
    """Look up H2H odds for a specific match from the odds list."""
    for o in odds_list:
        if o["home_team"].lower() == home.lower() and o["away_team"].lower() == away.lower():
            return o
        # Some APIs use different team name casing
        if o["home_team"].lower() == away.lower() and o["away_team"].lower() == home.lower():
            return {
                "h2h_home": o["h2h_away"],
                "h2h_draw": o["h2h_draw"],
                "h2h_away": o["h2h_home"],
            }
    return None


# ──────────────────────────────────────────────
# STEP 5 – V5.0 QUALITATIVE LAYER
# ──────────────────────────────────────────────
# v5.0 additions: injury_factor, knockout_defense_adjuster,
# brand_premium_corrector, altitude_factor, battle_intent
# ──────────────────────────────────────────────

# Known brand-inflation coefficients (market & model overrate these)
# Based on post-hoc analysis of WC26 group+knockout betting patterns
BRAND_PREMIUM = {
    # team: correction_factor (>1 = devalue, <1 = upvalue)
    "Brazil":     0.82,   # -18% brand inflation  ← BIGGEST offender
    "Argentina":  0.88,   # -12% (defending champion halo)
    "Germany":    0.85,   # -15% (history premium, current roster weaker)
    "England":    0.90,   # -10% (valuation vs performance gap)
    "France":     0.95,   # -5%  (legitimately strong, minimal inflation)
    "Spain":      0.93,   # -7%
    "Portugal":   0.92,   # -8%
    "Netherlands":0.92,   # -8%
}

# Known defensive stalwarts in WC26 (key for knockout defense adjuster)
# Teams whose primary identity is defensive solidity/bus-parking
DEFENSIVE_TEAMS = {
    "Cape Verde":        0.72,  # xG multiplier: only concede 72% of opponent xG
    "Paraguay":          0.75,  # conceded 1 goal in 4 matches vs TOP opponents
    "Morocco":           0.78,  # 2022 semi-finalist defensive DNA
    "Egypt":             0.80,  # 541 bus parkers
    "Saudi Arabia":      0.82,
    "Iran":              0.78,
    "Canada":            0.85,
    "Croatia":           0.88,
    "Uruguay":           0.88,
}

# Known attacking powerhouses — their xG overperforms in open games
ATTACKING_TEAMS = {
    "France":    1.15,  # overperforms xG by 15%
    "England":   1.10,
    "Argentina": 1.08,
    "Spain":     1.10,
}

# Stadium altitudes (metres above sea level) for altitude adjuster
STADIUM_ALTITUDE = {
    # Mexico City is the famous high-altitude venue
    "Mexico City":     2240,
    "Guadalajara":     1566,
    "Monterrey":       540,
    "Puebla":          2175,
    # Default: most US/Canada venues are near sea level
    "DEFAULT":         10,
}


def _team_altitude(team: str) -> int:
    """Return estimated stadium altitude for a team's 'home' venue."""
    ALT_MAP = {
        "Mexico": 2240,  # Mexico City Azteca
        "United States": 10,
        "Canada": 10,
    }
    return ALT_MAP.get(team, STADIUM_ALTITUDE["DEFAULT"])


def injury_factor(
    home_injuries: list[str] | None = None,
    away_injuries: list[str] | None = None,
) -> dict:
    """
    v5.0: Quantify injury impact on team strength.

    Returns dict with:
        elo_adjustment: dict of {team_name: elo_delta} to apply
        xg_multiplier:  dict of {team_name: xg_scale} (0.0-1.0)

    Injury severity tiers:
        Tier 1 (Captain / Ballon d'Or candidate):  -80 Elo, 0.75x xG
        Tier 2 (Star player / key position):       -50 Elo, 0.85x xG
        Tier 3 (Rotation / squad depth):           -20 Elo, 0.93x xG
    """
    # Pre-defined injury tiers for WC26 known absentees
    TIER1 = {"Messi", "Mbappé", "Haaland", "Kane", "Neymar", "Vinícius Júnior",
             "De Bruyne", "Salah", "Lewandowski", "Bellingham"}
    TIER2 = {"Raphinha", "Paquetá", "Rodri", "Alisson", "Courtois", "Casemiro",
             "Foden", "Musiala", "Gavi", "Pedri", "Ødegaard", "Davies",
             "Saliba", "Rice", "Valverde", "Araújo", "De Arrascaeta",
             "Grimaldo", "Wirtz", "Yamal","Saka", "Palmer", "Rice"}

    result = {"elo_adjust": {}, "xg_scale": {}}

    def _rate(name: str) -> tuple:
        name_clean = name.strip()
        if name_clean in TIER1:
            return (-80, 0.75)
        elif name_clean in TIER2:
            return (-50, 0.85)
        else:
            return (-20, 0.93)

    for team, injuries in [("home", home_injuries or []), ("away", away_injuries or [])]:
        total_elo = 0
        total_xg = 1.0
        for inj in injuries:
            elo_d, xg_s = _rate(inj)
            total_elo += elo_d
            total_xg *= xg_s
        result["elo_adjust"][team] = total_elo
        result["xg_scale"][team] = total_xg

    return result


def knockout_defense_adjuster(
    opponent: str,
    is_knockout: bool,
    team_is_deep_attacking: bool = False,
) -> float:
    """
    v5.0: Adjust xG conversion rate based on opponent's defensive profile.

    In knockouts, defensive teams compress space, reducing per-shot xG conversion.
    Returns a multiplier to apply to the ATTACKING team's xG.
    """
    if not is_knockout:
        return 1.0  # group stage — no adjustment

    # If the attacking team is itself a powerhouse, less impact
    base_mul = 0.88 if team_is_deep_attacking else 0.92

    # Look up opponent's defensive coefficient
    opp_d = DEFENSIVE_TEAMS.get(opponent, 1.0)

    # Blend: if opponent is defensive (0.72), apply stronger reduction
    if opp_d < 0.85:
        # Real defensive team: big xG reduction
        return min(0.95, opp_d + 0.12)
    elif opp_d < 0.95:
        # Moderately defensive
        return min(0.98, opp_d + 0.10)
    else:
        # Open playing team: no adjustment
        return 1.0


def brand_premium_corrector(
    home: str,
    away: str,
    market_probs: np.ndarray,
) -> np.ndarray:
    """
    v5.0: Correct market odds for brand inflation.

    Some teams (Brazil, Argentina, Germany) have inflated market prices
    due to historical brand value, not current form. This function
    deflates their probability and redistributes to opponent/draw.
    """
    h_correct = BRAND_PREMIUM.get(home, 1.0)
    a_correct = BRAND_PREMIUM.get(away, 1.0)

    # Only adjust if brand inflation detected (correction < 1.0)
    if h_correct >= 1.0 and a_correct >= 1.0:
        return market_probs

    corrected = market_probs.copy()

    if h_correct < 1.0:
        # Home brand inflated: reduce home prob, redistribute
        inflation = (1.0 - h_correct) * 0.7  # apply 70% of correction
        home_reduction = corrected[0] * inflation
        corrected[0] -= home_reduction
        # Redistribute 60% to away, 40% to draw
        corrected[1] += home_reduction * 0.40
        corrected[2] += home_reduction * 0.60

    if a_correct < 1.0:
        inflation_a = (1.0 - a_correct) * 0.7
        away_reduction = corrected[2] * inflation_a
        corrected[2] -= away_reduction
        corrected[0] += away_reduction * 0.40
        corrected[1] += away_reduction * 0.60

    # Renormalise
    corrected = corrected / corrected.sum()
    return corrected


def altitude_factor(
    home_team: str,
    away_team: str,
    neutral: bool,
) -> dict:
    """
    v5.0: Adjust for altitude effects.

    High altitude (>1500m) reduces away team's stamina and ball physics.
    Returns Elo adjustment for away team.
    """
    if neutral:
        return {"elo_adjust": 0}  # neutral venue, no altitude effect

    alt_h = _team_altitude(home_team)
    alt_a = _team_altitude(away_team)

    alt_diff = alt_h - alt_a

    if alt_diff > 1500:
        return {"elo_adjust": -80}
    elif alt_diff > 800:
        return {"elo_adjust": -50}
    elif alt_diff > 300:
        return {"elo_adjust": -25}
    else:
        return {"elo_adjust": 0}


def battle_intent_factor(
    is_knockout: bool,
    home_already_qualified: bool = False,
    away_already_qualified: bool = False,
    home_must_win: bool = False,
    away_must_win: bool = False,
) -> dict:
    """
    v5.0: Quantify team motivation/battle intent.

    Knockout: all teams at 100% intent → no adjustment.
    Group stage: teams already qualified may rest players.
    """
    if is_knockout:
        # Everyone fights at 100%
        return {"home_elo_bonus": 0, "away_elo_bonus": 0}

    h_bonus = 0
    a_bonus = 0

    if home_already_qualified:
        h_bonus = -60  # resting key players
    elif home_must_win:
        h_bonus = 25   # do-or-die motivation

    if away_already_qualified:
        a_bonus = -60
    elif away_must_win:
        a_bonus = 25

    return {"home_elo_bonus": h_bonus, "away_elo_bonus": a_bonus}


# ──────────────────────────────────────────────
# STEP 5a – ENSEMBLE PREDICTOR
# ──────────────────────────────────────────────


def predict_ensemble(
    home: str,
    away: str,
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    market_odds: dict | None = None,
    neutral: bool = True,
    is_wc: bool = True,
    is_knockout: bool = False,
    # ── v5.0 qualitative parameters ──
    home_injuries: list[str] | None = None,
    away_injuries: list[str] | None = None,
    home_qualified: bool = False,
    away_qualified: bool = False,
    home_must_win: bool = False,
    away_must_win: bool = False,
    opponent_is_defensive: bool | None = None,
    disable_brand_correction: bool = False,
    disable_altitude: bool = True,  # default True since most WC26 venues are near sea level
) -> dict:
    """
    V5.0 Ensemble predictor with full qualitative layer.

    V5.0 additions (compared to v4.9):
      1. INJURY FACTOR — elo/xg adjustment based on missing key players
      2. BRAND PREMIUM CORRECTION — de-bias Brazil/ARG/GER brand inflation
      3. ALTITUDE ADJUSTMENT — high-altitude away team penalty
      4. DEFENSE ADJUSTER — knockout xG reduction vs bus-parking teams
      5. BATTLE INTENT — group-stage motivation/resting adjustment

    KO Mode (is_knockout=True):
      - Market weight: 65%% — market prices in ALL info
      - Elo weight:    20%% — elo_v3 is now live-updated
      - ML weight:     15%%

    market_odds format:
        {"h2h_home": 1.50, "h2h_draw": 4.00, "h2h_away": 6.00}
    (decimal odds). Pass None to fall back to XGBoost-only.
    """
    # ════════════════════════════════════════════════
    # V5.0: INJURY ADJUSTMENT — adjust Elo ratings
    # ════════════════════════════════════════════════
    injury = injury_factor(home_injuries, away_injuries)
    elo_h_raw = elo._get_rating(home)
    elo_a_raw = elo._get_rating(away)

    # Apply injury elo adjustment (clamped to prevent extreme swings)
    elo_h = max(1400, elo_h_raw + injury["elo_adjust"].get("home", 0))
    elo_a = max(1400, elo_a_raw + injury["elo_adjust"].get("away", 0))

    # ════════════════════════════════════════════════
    # V5.0: ALTITUDE ADJUSTMENT
    # ════════════════════════════════════════════════
    if not disable_altitude and not neutral:
        alt_adj = altitude_factor(home, away, neutral)
        elo_a += alt_adj.get("elo_adjust", 0)

    # ════════════════════════════════════════════════
    # V5.0: BATTLE INTENT — group-stage motivation
    # ════════════════════════════════════════════════
    intent = battle_intent_factor(
        is_knockout,
        home_already_qualified=home_qualified,
        away_already_qualified=away_qualified,
        home_must_win=home_must_win,
        away_must_win=away_must_win,
    )
    elo_h += intent.get("home_elo_bonus", 0)
    elo_a += intent.get("away_elo_bonus", 0)

    # 1. XGBoost ML prediction
    ml = predict_match(home, away, elo, model, feat_df, neutral, is_wc)
    ml_probs = np.array([ml["p_home"], ml["p_draw"], ml["p_away"]])

    # V5.0: Apply injury xG scale to ML xG
    ml["xg_home"] = ml["xg_home"] * injury["xg_scale"].get("home", 1.0)
    ml["xg_away"] = ml["xg_away"] * injury["xg_scale"].get("away", 1.0)

    # 2. Elo-based prediction — using adjusted Elo
    if not neutral:
        p_home_elo = elo.expected(elo_h + 100, elo_a)
    else:
        p_home_elo = elo.expected(elo_h, elo_a)
    elo_diff = abs(elo_h - elo_a)
    p_draw_elo = max(0.12, min(0.35, 0.35 - elo_diff / 4000))
    p_draw_elo = min(p_draw_elo, min(p_home_elo, 1 - p_home_elo) * 0.8)
    p_away_elo = 1 - p_home_elo - p_draw_elo
    elo_probs = np.array([p_home_elo, p_draw_elo, p_away_elo])

    # 3. Market odds (remove vigorish)
    if market_odds and all(k in market_odds for k in ("h2h_home", "h2h_draw", "h2h_away")):
        raw = np.array([
            1.0 / market_odds["h2h_home"],
            1.0 / market_odds["h2h_draw"],
            1.0 / market_odds["h2h_away"],
        ])
        market_probs = raw / raw.sum()

        # ════════════════════════════════════════════════
        # V5.0: BRAND PREMIUM CORRECTION
        # ════════════════════════════════════════════════
        if not disable_brand_correction:
            corrected = brand_premium_corrector(home, away, market_probs)
            # Only apply if non-trivial change
            if np.max(np.abs(corrected - market_probs)) > 0.01:
                market_probs = corrected
    else:
        market_probs = ml_probs  # fallback

    # 4. Weighted combination
    if is_knockout and market_odds:
        W_ML, W_ELO, W_MARKET = 0.15, 0.20, 0.65
    else:
        W_ML, W_ELO, W_MARKET = 0.20, 0.30, 0.50

    final = W_ML * ml_probs + W_ELO * elo_probs + W_MARKET * market_probs
    final = final / final.sum()

    # ════════════════════════════════════════════════
    # V5.0: KNOCKOUT DEFENSE ADJUSTER — adjust xG for bus-parking teams
    # ════════════════════════════════════════════════
    xg_h = ml["xg_home"]
    xg_a = ml["xg_away"]
    xg_h_adjusted = xg_h
    xg_a_adjusted = xg_a

    if is_knockout:
        opponent_for_home = opponent_is_defensive if opponent_is_defensive is not None else \
            (away in DEFENSIVE_TEAMS or away in brand_premium_corrector.__globals__.get('DEFENSIVE_TEAMS', {}))
        opponent_for_away = opponent_is_defensive if opponent_is_defensive is not None else \
            (home in DEFENSIVE_TEAMS)

        # Determine if attacking teams are deep attacking
        home_is_attacking = home in ATTACKING_TEAMS
        away_is_attacking = away in ATTACKING_TEAMS

        # Apply defense adjuster to opponent xG
        if away in DEFENSIVE_TEAMS or opponent_is_defensive:
            adj = knockout_defense_adjuster(away, True, home_is_attacking)
            xg_h_adjusted *= adj
        if home in DEFENSIVE_TEAMS or opponent_is_defensive:
            adj = knockout_defense_adjuster(home, True, away_is_attacking)
            xg_a_adjusted *= adj

    # 5. Draw calibration
    if is_knockout:
        if not market_odds:
            if final[1] < 0.25:
                bump = min(0.04, 0.25 - final[1])
                final[1] += bump
                total_nondraw = final[0] + final[2]
                if total_nondraw > 0:
                    final[0] -= bump * (final[0] / total_nondraw)
                    final[2] -= bump * (final[2] / total_nondraw)
    else:
        if not market_odds and final[1] > 0.22:
            penalty = min(0.05, final[1] - 0.15)
            final[1] -= penalty
            total_nondraw = final[0] + final[2]
            if total_nondraw > 0:
                final[0] += penalty * (final[0] / total_nondraw)
                final[2] += penalty * (final[2] / total_nondraw)

    mode_tag = "knockout" if is_knockout else "group"
    dc = ml.get("dc", {})

    # Re-run DC probs if xG was adjusted
    adjusted_xg_used = abs(xg_h_adjusted - xg_h) > 0.01 or abs(xg_a_adjusted - xg_a) > 0.01

    v5_fields = {
        "v5_injury_adjust": injury,
        "v5_elo_adjusted_orig": {"home": round(elo_h_raw, 1), "away": round(elo_a_raw, 1)},
        "v5_intent": intent,
        "v5_xg_adjusted": {
            "home": round(xg_h_adjusted, 2),
            "away": round(xg_a_adjusted, 2),
        } if adjusted_xg_used else None,
        "v5_brand_corrected": True if (market_odds and not disable_brand_correction and
            (BRAND_PREMIUM.get(home, 1.0) < 1.0 or BRAND_PREMIUM.get(away, 1.0) < 1.0)) else False,
    }

    return {
        "home":                home,
        "away":                away,
        "ensemble_home":       round(float(final[0]), 3),
        "ensemble_draw":       round(float(final[1]), 3),
        "ensemble_away":       round(float(final[2]), 3),
        "ml_home":             ml["p_home"],
        "ml_draw":             ml["p_draw"],
        "ml_away":             ml["p_away"],
        "elo_home":            round(float(p_home_elo), 3),
        "elo_draw":            round(float(p_draw_elo), 3),
        "elo_away":            round(float(p_away_elo), 3),
        "market_home":         round(float(market_probs[0]), 3) if market_odds else None,
        "market_draw":         round(float(market_probs[1]), 3) if market_odds else None,
        "market_away":         round(float(market_probs[2]), 3) if market_odds else None,
        "xg_home":             round(xg_h_adjusted if adjusted_xg_used else xg_h, 2),
        "xg_away":             round(xg_a_adjusted if adjusted_xg_used else xg_a, 2),
        "elo_home_rating":     round(elo_h, 1),
        "elo_away_rating":     round(elo_a, 1),
        "weights":             {"ml": W_ML, "elo": W_ELO, "market": W_MARKET},
        "mode":                mode_tag,
        "over_15":             dc.get("over_15"),
        "over_25":             dc.get("over_25"),
        "over_35":             dc.get("over_35"),
        "btts_yes":            dc.get("btts_yes"),
        "btts_no":             dc.get("btts_no"),
        "ah_home_05":          dc.get("ah_home_05"),
        "ah_home_025":         dc.get("ah_home_025"),
        "ah_home_075":         dc.get("ah_home_075"),
        "ah_away_05":          dc.get("ah_away_05"),
        "ah_away_025":         dc.get("ah_away_025"),
        "ah_away_075":         dc.get("ah_away_075"),
        "correct_score":       dc.get("correct_score"),
        "dc_probs":            dc.get("home_win"),
        # v5.0 debug fields
        "v5":                  v5_fields,
    }


# ══════════════════════════════════════════════════════════════════
# 竞彩推荐层 — China Sports Lottery Recommendation Layer
# v1.0  Added 2026-06-26 — converts ensemble prediction into
#         中国体彩竞彩 format recommendations
# ════════════════════════════════════════════════════════════════

def predict_jingcai(
    match_no: str,
    home: str,
    away: str,
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    sp_odds: dict | None = None,
    rangqiu_odds: dict | None = None,
    neutral: bool = True,
    is_wc: bool = True,
    is_knockout: bool = False,
    note: str = "",
    re_du_mismatch: float | None = None,
    # ── v5.0 parameters ──
    home_injuries: list[str] | None = None,
    away_injuries: list[str] | None = None,
    home_qualified: bool = False,
    away_qualified: bool = False,
    home_must_win: bool = False,
    away_must_win: bool = False,
    opponent_is_defensive: bool | None = None,
    disable_brand_correction: bool = False,
    disable_altitude: bool = True,
) -> dict:
    """
    中国体彩竞彩格式预测推荐 (v5.0)
    推荐以 竞彩SP 为准，模型概率作为辅助参考。

    Parameters:
        match_no: 体彩场次编号 (e.g. "061")
        home: 主队名
        away: 客队名
        sp_odds: {"home": 主胜SP, "draw": 和局SP, "away": 客胜SP}
        rangqiu_odds: {"win": 让胜SP, "draw": 让平SP, "lose": 让负SP}
        re_du_mismatch: 热度偏差（%），正=大眾高估主隊，負=大眾低估主隊
        note: 额外情报备注
        home_injuries: 主隊傷缺球員名單
        away_injuries: 客隊傷缺球員名單
        home_qualified: 主隊已鎖定出線
        away_qualified: 客隊已鎖定出線
        home_must_win: 主隊必須贏波
        away_must_win: 客隊必須贏波
        opponent_is_defensive: 對手是否死守型（覆蓋自動判斷）
        disable_brand_correction: 禁用品牌溢價修正
        disable_altitude: 禁用海拔高度修正

    """
    # 获取 Ensemble 概率 (v5.0 with full qualitative layer)
    market_for_ensemble = None
    if sp_odds:
        market_for_ensemble = {
            "h2h_home": sp_odds["home"],
            "h2h_draw": sp_odds["draw"],
            "h2h_away": sp_odds["away"],
        }
    ens = predict_ensemble(
        home, away, elo, model, feat_df,
        market_for_ensemble, neutral, is_wc,
        is_knockout=is_knockout,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
        home_qualified=home_qualified,
        away_qualified=away_qualified,
        home_must_win=home_must_win,
        away_must_win=away_must_win,
        opponent_is_defensive=opponent_is_defensive,
        disable_brand_correction=disable_brand_correction,
        disable_altitude=disable_altitude,
    )

    probs = [ens["ensemble_home"], ens["ensemble_draw"], ens["ensemble_away"]]
    labels_spf = ["胜", "平", "负"]
    max_idx = probs.index(max(probs))

    # 竞彩反向概率（去水）
    if sp_odds:
        raw = np.array([1.0 / sp_odds["home"], 1.0 / sp_odds["draw"], 1.0 / sp_odds["away"]])
        market_probs = raw / raw.sum()
        sp_max_idx = int(np.argmax(market_probs))
    else:
        market_probs = np.array(probs)
        sp_max_idx = max_idx

    # 方向：以竞彩SP市场概率为准
    # 但如果冷热数据强烈指示大众高估某方向（|偏差|≥10），反着走
    spf_direction = labels_spf[sp_max_idx]
    
    if re_du_mismatch is not None and abs(re_du_mismatch) >= 10:
        # 大众高估主队（re_du_mismatch>0）→ 不推主胜
        if re_du_mismatch > 0 and sp_max_idx == 0:
            # 反着推：如果市场第二高是和局，推和局；否则推客胜
            sorted_idx = np.argsort(market_probs)[::-1]
            alt_idx = sorted_idx[1] if market_probs[sorted_idx[1]] > 0.25 else sorted_idx[2]
            spf_direction = labels_spf[alt_idx]
        # 大众低估主队（re_du_mismatch<0）→ 推主胜
        elif re_du_mismatch < 0 and sp_max_idx != 0:
            spf_direction = "胜"
        sp_max_idx = labels_spf.index(spf_direction)

    # SP key mapping: Chinese -> English
    _spf_to_key = {"胜": "home", "平": "draw", "负": "away"}
    sp_key = _spf_to_key.get(spf_direction, "home")
    sp_val = sp_odds.get(sp_key, "—") if sp_odds else "—"

    # 信心级别
    model_agree = abs(probs[sp_max_idx] - market_probs[sp_max_idx]) < 0.12
    if sp_odds and market_probs[sp_max_idx] >= 0.55 and model_agree:
        confidence = "🟢高"
    elif sp_odds and market_probs[sp_max_idx] >= 0.40:
        confidence = "🟡中"
    elif sp_odds and abs(re_du_mismatch or 0) >= 15:
        confidence = "🔥冷"
    else:
        confidence = "🟡中"

    # 让球推荐（按竞彩让球盘推荐）
    rangqiu_pick = None
    if rangqiu_odds:
        rq_probs = np.array([1.0 / v for v in rangqiu_odds.values() if v > 0])
        rq_labels = list(rangqiu_odds.keys())
        if len(rq_probs) == 3:
            rq_probs = rq_probs / rq_probs.sum()
            rq_best_idx = int(np.argmax(rq_probs))
            label_map = ["让胜", "让平", "让负"]
            rq_pick_label = label_map[rq_best_idx]
            rq_pick_odds = rq_labels[rq_best_idx]  # key name ("win"/"draw"/"lose")
            rangqiu_pick = (rq_pick_label, rangqiu_odds[rq_pick_odds])
        else:
            # fallback: lowest odds = most likely
            best_rq = min(rangqiu_odds, key=lambda k: rangqiu_odds[k])
            label_map_rq = {"win": "让胜", "draw": "让平", "lose": "让负"}
            rangqiu_pick = (label_map_rq.get(best_rq, "让平"), rangqiu_odds[best_rq])
    
    # 比分推荐（基于xG，Poisson）
    xg_h, xg_a = ens["xg_home"], ens["xg_away"]
    likely_scores = []
    for s_h in range(0, 5):
        for s_a in range(0, 5):
            like = (xg_h ** s_h * math.exp(-xg_h) / math.factorial(s_h)) * \
                   (xg_a ** s_a * math.exp(-xg_a) / math.factorial(s_a))
            if like > 0.05 and s_h + s_a <= 8:
                likely_scores.append((f"{s_h}:{s_a}", like))
    likely_scores.sort(key=lambda x: -x[1])
    score_pick = [s[0] for s in likely_scores[:2]] if likely_scores else []

    # 热度标记
    re_du_tag = ""
    if re_du_mismatch is not None:
        if re_du_mismatch > 10:
            re_du_tag = f"⚠️大眾高估主隊{re_du_mismatch:+.0f}%"
        elif re_du_mismatch < -10:
            re_du_tag = f"⛽大眾低估主隊{re_du_mismatch:+.0f}%"

    entry = {
        "match_no": match_no,
        "home": home,
        "away": away,
        "sp_odds": sp_odds or {},
        "rangqiu_odds": rangqiu_odds or {},
        "ensemble": {
            "h": round(probs[0], 3),
            "d": round(probs[1], 3),
            "a": round(probs[2], 3),
        },
        "xg": {"home": round(xg_h, 2), "away": round(xg_a, 2)},
        "re_du_mismatch": re_du_mismatch,
        "recommendation": {
            "spf": spf_direction,
            "spf_odds": sp_val,
            "rangqiu": f"{rangqiu_pick[0]}@{rangqiu_pick[1]}" if rangqiu_pick else "—",
            "score": score_pick,
            "confidence": confidence,
            "re_du": re_du_tag,
        },
        "note": note,
    }
    return entry


def format_jingcai_picks(picks: list[dict]) -> str:
    """Format 竞彩 prediction entries into clean lines."""
    lines = []
    for p in picks:
        r = p["recommendation"]
        mn = p["match_no"]
        home, away = p["home"], p["away"]
        score_str = " | 比分 " + "/".join(r["score"][:2]) if r["score"] else ""
        re_du_str = f" {r['re_du']}" if r.get("re_du") else ""
        rangqiu_str = f" | {r['rangqiu']}" if r['rangqiu'] != '—' else ""

        lines.append(f"{mn} {home} vs {away}")
        lines.append(f"推薦：{r['spf']}@{r['spf_odds']}{rangqiu_str} | 信心{r['confidence']}{re_du_str}{score_str}")
        if p.get('note'):
            lines.append(f"  {p['note']}")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# STEP 6 – VISUALISATION
# ──────────────────────────────────────────────

COLORS = {
    "home":    "#2563EB",   # blue
    "draw":    "#64748B",   # gray
    "away":    "#DC2626",   # red
    "bg":      "#0F172A",   # dark bg (YouTube thumbnail style)
    "text":    "#F8FAFC",
    "accent":  "#F59E0B",   # amber
}


def plot_match_prediction(pred: dict, save_path: Path | None = None):
    """
    Generate a YouTube-thumbnail-ready match prediction card.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor=COLORS["bg"])
    fig.suptitle(
        f"AI MATCH PREDICTION  |  FIFA World Cup 2026",
        color=COLORS["accent"], fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Left: probability bar chart ──────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(COLORS["bg"])
    labels = [pred["home"], "Draw", pred["away"]]
    probs  = [pred["p_home"], pred["p_draw"], pred["p_away"]]
    colors = [COLORS["home"], COLORS["draw"], COLORS["away"]]
    bars   = ax1.barh(labels, [p * 100 for p in probs], color=colors,
                      height=0.5, edgecolor="none")

    for bar, p in zip(bars, probs):
        ax1.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%", va="center", color=COLORS["text"], fontsize=14,
            fontweight="bold",
        )

    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Win probability (%)", color=COLORS["text"], fontsize=11)
    ax1.tick_params(colors=COLORS["text"], labelsize=12)
    ax1.spines[:].set_visible(False)
    ax1.set_title("Win probabilities", color=COLORS["text"],
                  fontsize=12, pad=10)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.xaxis.set_tick_params(color=COLORS["text"])

    # ── Right: expected goals gauge ───────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # xG circles
    for x, xg, label, color in [
        (0.25, pred["xg_home"], pred["home"], COLORS["home"]),
        (0.75, pred["xg_away"], pred["away"], COLORS["away"]),
    ]:
        circle = plt.Circle((x, 0.55), 0.18, color=color, alpha=0.25)
        ax2.add_patch(circle)
        circle_border = plt.Circle((x, 0.55), 0.18,
                                   color=color, fill=False, lw=2)
        ax2.add_patch(circle_border)
        ax2.text(x, 0.55, f"{xg:.2f}", ha="center", va="center",
                 color=COLORS["text"], fontsize=22, fontweight="bold")
        ax2.text(x, 0.30, label, ha="center", va="center",
                 color=COLORS["text"], fontsize=11, fontweight="bold")

    ax2.text(0.5, 0.92, "Expected Goals (xG)", ha="center",
             color=COLORS["text"], fontsize=12)
    ax2.text(0.5, 0.55, "vs", ha="center", va="center",
             color=COLORS["accent"], fontsize=16, fontweight="bold")

    # Elo ratings
    ax2.text(0.25, 0.12, f"Elo: {pred['elo_home']:.0f}",
             ha="center", color=COLORS["home"], fontsize=10)
    ax2.text(0.75, 0.12, f"Elo: {pred['elo_away']:.0f}",
             ha="center", color=COLORS["away"], fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_elo_rankings(elo: EloRatingSystem, save_path: Path | None = None,
                      top_n: int = 20):
    """Bar chart of top-N team Elo ratings — great for YouTube intros."""
    top = elo.top_n(top_n)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bar_colors = [
        COLORS["accent"] if i < 3 else COLORS["home"]
        for i in range(len(top))
    ]
    bars = ax.barh(top["team"][::-1], top["elo"][::-1],
                   color=bar_colors[::-1], height=0.65, edgecolor="none")

    for bar, elo_val in zip(bars, top["elo"][::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{elo_val:.0f}", va="center", color=COLORS["text"],
                fontsize=9)

    ax.set_xlabel("Elo Rating", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.spines[:].set_visible(False)
    ax.set_title("AI Power Rankings — FIFA World Cup 2026",
                 color=COLORS["accent"], fontsize=14, fontweight="bold", pad=12)

    gold  = mpatches.Patch(color=COLORS["accent"], label="Top 3")
    other = mpatches.Patch(color=COLORS["home"],   label="Top 20")
    ax.legend(handles=[gold, other], facecolor=COLORS["bg"],
              labelcolor=COLORS["text"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Rankings chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# STEP 7 – TOURNAMENT SIMULATOR
# ──────────────────────────────────────────────

def simulate_tournament(
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of the full tournament.
    Returns championship probability for every team.
    """
    print(f"\n🏆  Running {n_sims:,} tournament simulations …")

    # Simplified: use teams from WC2026_MATCHES
    all_teams = sorted(set(
        t for m in WC2026_MATCHES for t in [m[0], m[1]]
    ))

    wins = defaultdict(int)

    for _ in range(n_sims):
        # Sample winner of each group match and advance top 2 per group
        # (simplified bracket: randomly pick 8 QF matchups from group winners)
        remaining = list(all_teams)

        # Knock out progressively until 1 team remains
        while len(remaining) > 1:
            next_round = []
            rng = np.random.default_rng()
            rng.shuffle(remaining)
            for i in range(0, len(remaining) - 1, 2):
                h, a = remaining[i], remaining[i + 1]
                pred = predict_match(h, a, elo, model, feat_df,
                                     neutral=True, is_wc=True)
                roll = rng.random()
                if roll < pred["p_home"]:
                    next_round.append(h)
                elif roll < pred["p_home"] + pred["p_draw"]:
                    # Penalty shootout: 50/50
                    next_round.append(h if rng.random() > 0.5 else a)
                else:
                    next_round.append(a)
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])  # bye
            remaining = next_round

        wins[remaining[0]] += 1

    results = pd.DataFrame([
        {"team": t, "championship_prob": round(wins[t] / n_sims * 100, 2)}
        for t in sorted(all_teams, key=lambda x: -wins[x])
    ])
    return results


def plot_championship_probs(sim_results: pd.DataFrame,
                            save_path: Path | None = None):
    top = sim_results.head(12)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bar_colors = [
        COLORS["accent"] if i == 0 else
        ("#94A3B8" if i < 3 else COLORS["home"])
        for i in range(len(top))
    ]
    ax.bar(top["team"], top["championship_prob"],
           color=bar_colors, edgecolor="none")

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(i, row["championship_prob"] + 0.2,
                f"{row['championship_prob']:.1f}%",
                ha="center", color=COLORS["text"], fontsize=9,
                fontweight="bold")

    ax.set_ylabel("Championship probability (%)", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], axis="both")
    ax.spines[:].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    ax.set_title("Who Will Win World Cup 2026? — AI Simulation",
                 color=COLORS["accent"], fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Simulation chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def print_prediction(pred: dict):
    bar = "█"
    total = 40
    h_bar = bar * round(pred["p_home"] * total)
    d_bar = bar * round(pred["p_draw"] * total)
    a_bar = bar * round(pred["p_away"] * total)

    print(f"\n  ⚽  {pred['home']:>16}  vs  {pred['away']:<16}")
    print(f"  {'─'*52}")
    print(f"  {'Home win':<12} {h_bar:<40} {pred['p_home']*100:5.1f}%  (Elo {pred['elo_home']:.0f})")
    print(f"  {'Draw':<12} {d_bar:<40} {pred['p_draw']*100:5.1f}%")
    print(f"  {'Away win':<12} {a_bar:<40} {pred['p_away']*100:5.1f}%  (Elo {pred['elo_away']:.0f})")
    print(f"  {'─'*52}")
    print(f"  xG: {pred['home']} {pred['xg_home']:.2f} – {pred['xg_away']:.2f} {pred['away']}")
    print(f"  🏅  Favorite: {pred['favorite']}")


def main():
    parser = argparse.ArgumentParser(description="FIFA 2026 Match Predictor")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a specific match")
    parser.add_argument("--simulate", action="store_true",
                        help="Run full tournament simulation")
    parser.add_argument("--rankings", action="store_true",
                        help="Show Elo power rankings")
    args = parser.parse_args()

    # ── Pipeline ──────────────────────────────
    df      = load_results()
    elo     = EloRatingSystem().fit(df)
    feat_df = build_features(df, elo)
    model   = train_model(feat_df)

    # ── Rankings ──────────────────────────────
    print("\n📊  Current AI Power Rankings (Elo):")
    print(elo.top_n(15).to_string(index=False))
    plot_elo_rankings(elo, save_path=VIS_DIR / "elo_rankings.png")

    # ── Match predictions ─────────────────────
    if args.match:
        home, away = args.match
        pred = predict_match(home, away, elo, model, feat_df)
        print_prediction(pred)
        plot_match_prediction(pred,
            save_path=VIS_DIR / f"pred_{home.lower()}_{away.lower()}.png")
    else:
        # Demo: predict a selection of high-interest WC26 group matches
        demo_matches = [
            ("France",    "Argentina"),
            ("Brazil",    "Germany"),
            ("England",   "Spain"),
            ("USA",       "Canada"),
        ]
        print("\n🔮  Pre-tournament predictions for key group-stage clashes:\n")
        for home, away in demo_matches:
            pred = predict_match(home, away, elo, model, feat_df)
            print_prediction(pred)
            plot_match_prediction(
                pred,
                save_path=VIS_DIR / f"pred_{home.lower()}_{away.lower()}.png",
            )

    # ── Tournament simulation ──────────────────
    if args.simulate:
        sim = simulate_tournament(elo, model, feat_df, n_sims=10_000)
        print("\n🏆  Championship probabilities:")
        print(sim.head(12).to_string(index=False))
        plot_championship_probs(sim,
            save_path=VIS_DIR / "championship_probs.png")


if __name__ == "__main__":
    main()
