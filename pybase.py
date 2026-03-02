"""
Minnesota Twins Rebuild Analysis
=================================
Position-aware Statcast + FanGraphs analytics for rebuild evaluation.

Pitcher roles (SP/RP/CL) and batter positions (SS/1B/OF etc.) each get
their own baseline population, stat weights, and Twins roster averages
so grades reflect what actually matters at each position.

Dependencies: pybaseball, pandas, numpy, matplotlib, scipy
Install:      pip install pybaseball pandas numpy matplotlib scipy
"""

import pybaseball as pb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
pb.cache.enable()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR     = "twins_rebuild_output"
ANALYSIS_YEARS = ["2022", "2023", "2024"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (Full Name, Position, Age, Contract, Years of Control)
#
# Pitcher roles:  "SP" = starter  |  "RP" = reliever  |  "CL" = closer
# Batter slots:   "C" "1B" "2B" "3B" "SS" "LF" "CF" "RF" "DH" "OF"
#
# Position drives: FanGraphs baseline population, stat weights,
#                  Twins positional roster average for delta comparison.

PLAYERS = [
    # ── Pitchers ──────────────────────────────────────────────
    ("Framber Valdez",   "SP",  30, "$16.5m",  2),
    ("Jalen Beeks",      "RP",  30, "$3.2m",   2),
    ("Jose Leclerc",     "CL",  30, "$5.5m",   1),
    ("Taj Bradley",      "SP",  24, "PRE-ARB", 4),
    ("Connelly Early",   "SP",  25, "PRE-ARB", 4),
    # ── Batters ───────────────────────────────────────────────
    ("Carlos Correa",    "SS",  30, "$35.1m",  4),
    ("Yandy Diaz",       "1B",  33, "$9m",     2),
    ("Andrew McCutchen", "DH",  37, "FA",      0),
    ("Mike Tauchman",    "OF",  33, "$1.5m",   1),
]

# ── Position groupings ────────────────────────────────────────
PITCHER_ROLES   = {"SP", "RP", "CL"}
PREMIUM_DEF_POS = {"C", "SS", "2B", "CF"}
CORNER_POS      = {"1B", "3B", "LF", "RF", "DH", "OF"}

def is_pitcher(pos: str) -> bool:
    return pos in PITCHER_ROLES

REBUILD_ARCHETYPES = {
    "Cornerstone":   {"label": "Cornerstone"},
    "Bridge":        {"label": "Bridge"},
    "Trade Chip":    {"label": "Trade Chip"},
    "DFA Candidate": {"label": "DFA Candidate"},
}

# ─────────────────────────────────────────────────────────────
# ID LOADING
# ─────────────────────────────────────────────────────────────

_CSV_IDS: dict = {}
_CSV_PATH = "player_ids.csv"
if os.path.exists(_CSV_PATH):
    try:
        _id_df   = pd.read_csv(_CSV_PATH)
        _CSV_IDS = dict(zip(_id_df["Player"], _id_df["MLBAM_ID"].astype(int)))
        print(f"  Loaded {len(_CSV_IDS)} MLBAM IDs from {_CSV_PATH}")
    except Exception as e:
        print(f"  Could not load {_CSV_PATH}: {e}")


def get_player_id(name: str):
    # Try exact match first
    if name in _CSV_IDS:
        return _CSV_IDS[name]
    # Try accent-stripped variants (e.g. "Yandy Diaz" matches "Yandy Díaz")
    for csv_name, mlbam_id in _CSV_IDS.items():
        if _strip_accents(csv_name) == _strip_accents(name):
            return mlbam_id
    # Live lookup fallback
    parts = name.split()
    last, first = parts[-1], parts[0]
    try:
        result = pb.playerid_lookup(last, first)
        if not result.empty:
            mlbam_id = int(result.iloc[0]["key_mlbam"])
            print(f"  Live lookup: {name} -> {mlbam_id}")
            return mlbam_id
    except Exception as e:
        print(f"  ID lookup failed for {name}: {e}")
    return None


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()

# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_statcast(player_id: int, year: str, pitcher: bool) -> pd.DataFrame:
    start, end = f"{year}-03-01", f"{year}-11-01"
    try:
        df = pb.statcast_pitcher(start, end, player_id) if pitcher \
             else pb.statcast_batter(start, end, player_id)
        df["season"] = int(year)
        return df
    except Exception as e:
        print(f"    Statcast fetch failed ({year}): {e}")
        return pd.DataFrame()


def fetch_all_seasons(player_id: int, pitcher: bool) -> pd.DataFrame:
    frames = []
    for yr in ANALYSIS_YEARS:
        print(f"    Pulling {yr}...", end=" ")
        df = fetch_statcast(player_id, yr, pitcher)
        print(f"{len(df)} rows")
        frames.append(df)
    valid = [f for f in frames if not f.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# BATTER ANALYTICS
# ─────────────────────────────────────────────────────────────

def analyze_batter(data: pd.DataFrame) -> dict:
    if data.empty:
        return {}
    r  = {}
    ev = data["launch_speed"].dropna()
    la = data["launch_angle"].dropna()

    r["Avg EV"]       = round(ev.mean(), 1)       if len(ev) else None
    r["Max EV"]       = round(ev.max(),  1)        if len(ev) else None
    r["Hard Hit %"]   = round((ev >= 95).mean() * 100, 1) if len(ev) else None
    r["Barrel %"]     = _barrel_pct(data)
    r["xwOBA"]        = _col_mean(data, "estimated_woba_using_speedangle", 3)
    r["xBA"]          = _col_mean(data, "estimated_ba_using_speedangle", 3)
    r["xSLG"]         = _col_mean(data, "estimated_slg_using_speedangle", 3)
    r["Sweet Spot %"] = round(((la >= 8) & (la <= 32)).mean() * 100, 1) if len(la) else None

    swings = data[data["description"].str.contains(
        "swing|hit_into|foul|in_play", case=False, na=False)]
    swstr  = data[data["description"].str.contains(
        "swinging_strike", case=False, na=False)]

    r["K %"]       = _event_pct(data, "strikeout")
    r["BB %"]      = _event_pct(data, "walk")
    r["Whiff %"]   = round(len(swstr) / max(len(swings), 1) * 100, 1)
    r["Chase %"]   = _chase_pct(data)
    r["Contact %"] = round((1 - len(swstr) / max(len(swings), 1)) * 100, 1)

    bip = data[data["type"] == "X"]
    if not bip.empty and "bb_type" in bip.columns:
        vc = bip["bb_type"].value_counts(normalize=True) * 100
        r["GB %"]   = round(vc.get("ground_ball", 0), 1)
        r["LD %"]   = round(vc.get("line_drive",  0), 1)
        r["FB %"]   = round(vc.get("fly_ball",    0), 1)
        r["IFFB %"] = round(vc.get("popup",       0), 1)
        r["Pull %"] = _spray_pct(bip, "pull")
        r["Oppo %"] = _spray_pct(bip, "opposite")

    if "season" in data.columns:
        yr_ev = data.groupby("season")["launch_speed"].mean().dropna()
        r["EV Trend (y/y)"] = round(yr_ev.iloc[-1] - yr_ev.iloc[0], 1) \
                               if len(yr_ev) >= 2 else None

    r["PA"] = len(data)
    return r

# ─────────────────────────────────────────────────────────────
# PITCHER ANALYTICS
# ─────────────────────────────────────────────────────────────

def analyze_pitcher(data: pd.DataFrame) -> dict:
    if data.empty:
        return {}
    r = {}
    r["Avg Velo"]           = _col_mean(data, "release_speed", 1)
    r["Max Velo"]           = round(data["release_speed"].dropna().max(), 1) \
                              if "release_speed" in data.columns else None
    r["Avg Spin (FB)"]      = _pitch_stat(data, ["FF","SI","FC"], "release_spin_rate", 0)
    r["Avg Extension"]      = _col_mean(data, "release_extension", 2)

    swstr  = data[data["description"].str.contains("swinging_strike", case=False, na=False)]
    swings = data[data["description"].str.contains(
        "swing|hit_into|foul|in_play", case=False, na=False)]
    called = data[data["description"] == "called_strike"]

    r["Whiff %"]            = round(len(swstr) / max(len(swings), 1) * 100, 1)
    r["CSW %"]              = round((len(swstr)+len(called)) / max(len(data), 1) * 100, 1)
    r["K %"]                = _event_pct(data, "strikeout")
    r["BB %"]               = _event_pct(data, "walk")
    r["xwOBA Against"]      = _col_mean(data, "estimated_woba_using_speedangle", 3)
    r["Avg EV Against"]     = _col_mean(data, "launch_speed", 1)
    r["Hard Hit % Against"] = round((data["launch_speed"].dropna() >= 95).mean() * 100, 1) \
                              if "launch_speed" in data.columns else None
    r["Barrel % Against"]   = _barrel_pct(data)

    if "pitch_type" in data.columns:
        pt = data["pitch_type"].value_counts(normalize=True) * 100
        r["Arsenal"] = {p: round(v, 1) for p, v in pt.items() if v >= 2.0}

        whiff_d, velo_d = {}, {}
        for p in r["Arsenal"]:
            pd_ = data[data["pitch_type"] == p]
            sw  = pd_[pd_["description"].str.contains(
                "swing|hit_into|foul|in_play", case=False, na=False)]
            ss  = pd_[pd_["description"].str.contains(
                "swinging_strike", case=False, na=False)]
            if len(sw) > 10:
                whiff_d[p] = round(len(ss)/len(sw)*100, 1)
            v = pd_["release_speed"].dropna()
            if len(v) > 10:
                velo_d[p] = round(v.mean(), 1)
        r["Pitch Whiff %"] = whiff_d
        r["Pitch Velo"]    = velo_d

    if "season" in data.columns:
        yr_v = data.groupby("season")["release_speed"].mean().dropna()
        r["Velo Trend (y/y)"] = round(yr_v.iloc[-1] - yr_v.iloc[0], 1) \
                                 if len(yr_v) >= 2 else None

    r["Pitches"] = len(data)
    return r

# ── Helpers ───────────────────────────────────────────────────

def _barrel_pct(data):
    if "barrel" not in data.columns:
        return None
    bip = data[data["type"] == "X"]
    return round(bip["barrel"].fillna(0).mean() * 100, 1) if not bip.empty else None

def _col_mean(data, col, dec=2):
    if col not in data.columns:
        return None
    v = data[col].dropna()
    return round(v.mean(), dec) if len(v) else None

def _event_pct(data, event):
    if "events" not in data.columns:
        return None
    pas = data.dropna(subset=["events"])
    return round((pas["events"] == event).sum() / max(len(pas), 1) * 100, 1)

def _chase_pct(data):
    oz = data[data["zone"].isin([11,12,13,14]) | (data["zone"] > 9)]
    sw = oz[oz["description"].str.contains(
        "swing|hit_into|foul|in_play", case=False, na=False)]
    return round(len(sw)/max(len(oz),1)*100, 1) if not oz.empty else None

def _spray_pct(bip, direction):
    if "hit_location" not in bip.columns:
        return None
    locs = [3,4,9] if direction == "pull" else [5,6,7]
    return round(bip["hit_location"].isin(locs).mean()*100, 1)

def _pitch_stat(data, pitch_types, col, dec):
    sub = data[data["pitch_type"].isin(pitch_types)] \
          if "pitch_type" in data.columns else data
    return _col_mean(sub, col, dec)

def _ordinal(n: float) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', etc."""
    n = int(round(n))
    if 11 <= (n % 100) <= 13:
        sfx = "th"
    else:
        sfx = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{sfx}"

# ─────────────────────────────────────────────────────────────
# POSITION-AWARE BASELINES
# ─────────────────────────────────────────────────────────────

_BASELINE_CACHE: dict = {}

# FanGraphs stores ALL percentage stats as decimals (0.22 = 22%)
# We divide our 0-100 values before comparison
FG_DECIMAL_STATS = {"K %", "BB %", "CSW %", "Barrel %", "Barrel % Against"}

# Stats where LOWER is BETTER — flip percentile rank
PITCHER_INVERT = {"xwOBA Against", "Barrel % Against", "BB %"}
BATTER_INVERT  = {"K %"}

# Stats where positive (player − avg) delta means WORSE — flip sign
PITCHER_DELTA_INVERT = {"xwOBA Against", "Barrel % Against", "BB %"}
BATTER_DELTA_INVERT  = {"K %"}

PITCHER_STAT_MAP = {
    "xwOBA":   "xwOBA Against",
    "Barrel%": "Barrel % Against",
    "K%":      "K %",
    "BB%":     "BB %",
    "CSW%":    "CSW %",
}
BATTER_STAT_MAP = {
    "xwOBA":   "xwOBA",
    "Barrel%": "Barrel %",
    "BB%":     "BB %",
    "K%":      "K %",
}
FG_COL_ALIASES = {
    "xwOBA":   ["xwOBA","xwoba"],
    "Barrel%": ["Barrel%","Barrel%1","Barrel"],
    "BB%":     ["BB%"],
    "K%":      ["K%"],
    "CSW%":    ["CSW%"],
}

# ── Position-specific stat weights ───────────────────────────
# Each position emphasises what actually matters there.
WEIGHTS = {
    # Pitchers — command weight increases for SP; K% weight increases for CL
    "SP": {"xwOBA Against":0.35, "K %":0.25, "BB %":0.20, "CSW %":0.20},
    "RP": {"xwOBA Against":0.30, "K %":0.30, "BB %":0.15, "CSW %":0.25},
    "CL": {"xwOBA Against":0.25, "K %":0.40, "BB %":0.10, "CSW %":0.25},
    # Batters — power emphasis increases toward corner positions
    "SS": {"xwOBA":0.40, "Barrel %":0.15, "BB %":0.25, "K %":0.20},
    "2B": {"xwOBA":0.40, "Barrel %":0.15, "BB %":0.25, "K %":0.20},
    "C":  {"xwOBA":0.45, "Barrel %":0.20, "BB %":0.20, "K %":0.15},
    "CF": {"xwOBA":0.40, "Barrel %":0.20, "BB %":0.25, "K %":0.15},
    "3B": {"xwOBA":0.40, "Barrel %":0.25, "BB %":0.20, "K %":0.15},
    "1B": {"xwOBA":0.35, "Barrel %":0.35, "BB %":0.15, "K %":0.15},
    "DH": {"xwOBA":0.35, "Barrel %":0.35, "BB %":0.15, "K %":0.15},
    "OF": {"xwOBA":0.40, "Barrel %":0.25, "BB %":0.20, "K %":0.15},
    "LF": {"xwOBA":0.40, "Barrel %":0.25, "BB %":0.20, "K %":0.15},
    "RF": {"xwOBA":0.40, "Barrel %":0.25, "BB %":0.20, "K %":0.15},
}

# ── Twins positional roster averages (2024) ──────────────────
# Update from FanGraphs Twins splits page as needed.
# Pitching split into SP/RP/CL averages.
# Batting split by position (league avg used where Twins had no incumbent).
TWINS_AVGS = {
    "SP":  {"xwOBA Against":.335, "Barrel % Against":8.5,
            "K %":21.0, "BB %":8.5,  "CSW %":27.0},
    "RP":  {"xwOBA Against":.325, "Barrel % Against":7.8,
            "K %":23.5, "BB %":9.5,  "CSW %":27.5},
    "CL":  {"xwOBA Against":.320, "Barrel % Against":7.5,
            "K %":25.0, "BB %":9.0,  "CSW %":28.0},
    "SS":  {"xwOBA":.310, "Barrel %":6.5,  "BB %":8.0, "K %":21.0},
    "1B":  {"xwOBA":.320, "Barrel %":9.0,  "BB %":8.0, "K %":22.0},
    "DH":  {"xwOBA":.315, "Barrel %":8.5,  "BB %":8.5, "K %":22.5},
    "OF":  {"xwOBA":.315, "Barrel %":7.5,  "BB %":8.5, "K %":22.0},
    "CF":  {"xwOBA":.305, "Barrel %":6.5,  "BB %":8.5, "K %":22.0},
    "2B":  {"xwOBA":.305, "Barrel %":6.5,  "BB %":8.5, "K %":21.0},
    "3B":  {"xwOBA":.320, "Barrel %":9.0,  "BB %":8.5, "K %":21.5},
    "C":   {"xwOBA":.300, "Barrel %":6.0,  "BB %":7.5, "K %":22.5},
    "LF":  {"xwOBA":.315, "Barrel %":8.0,  "BB %":8.5, "K %":22.5},
    "RF":  {"xwOBA":.320, "Barrel %":9.0,  "BB %":8.5, "K %":22.0},
}


def _load_baseline(pos: str) -> pd.DataFrame:
    """
    Load (and cache) FanGraphs 2024 baseline for a position group.
    SP  -> all pitchers with GS >= 10
    RP/CL -> all pitchers with GS <= 2
    Batters -> full batting leaderboard (qual=100)
    """
    pitcher   = is_pitcher(pos)
    cache_key = "pitcher" if pitcher else "batter"

    if cache_key not in _BASELINE_CACHE:
        label = "pitcher" if pitcher else "batter"
        print(f"\n  Loading MLB {label} baseline (FanGraphs 2024)...")
        try:
            df = pb.pitching_stats(2024, qual=30) if pitcher \
                 else pb.batting_stats(2024, qual=100)
            df.columns = [c.replace(" ", "") for c in df.columns]
            _BASELINE_CACHE[cache_key] = df
            print(f"    {len(df)} players loaded.")
        except Exception as e:
            print(f"  Baseline load failed: {e}")
            _BASELINE_CACHE[cache_key] = pd.DataFrame()

    base = _BASELINE_CACHE.get(cache_key, pd.DataFrame())

    # Filter pitcher baseline to role-appropriate population
    if pitcher and not base.empty:
        gs_col = next((c for c in base.columns if c.lower() == "gs"), None)
        if gs_col:
            if pos == "SP":
                base = base[base[gs_col] >= 10].copy()
            else:
                base = base[base[gs_col] <= 2].copy()

    return base


def _percentile_rank(value: float, population: pd.Series,
                     invert: bool = False) -> float:
    clean = population.dropna()
    if clean.empty or value is None:
        return 50.0
    pct = float((clean < value).sum() / len(clean) * 100)
    return round(100 - pct if invert else pct, 1)


def _get_percentiles(stats: dict, pos: str) -> dict:
    baseline   = _load_baseline(pos)
    pitcher    = is_pitcher(pos)
    stat_map   = PITCHER_STAT_MAP if pitcher else BATTER_STAT_MAP
    invert_set = PITCHER_INVERT   if pitcher else BATTER_INVERT

    if baseline is None or baseline.empty:
        return {}

    percentiles = {}
    for fg_key, our_key in stat_map.items():
        player_val = stats.get(our_key)
        if player_val is None:
            continue
        col = next((a for a in FG_COL_ALIASES.get(fg_key, [fg_key])
                    if a in baseline.columns), None)
        if col is None:
            continue
        pop         = pd.to_numeric(baseline[col], errors="coerce")
        compare_val = player_val / 100.0 if our_key in FG_DECIMAL_STATS \
                      else player_val
        percentiles[our_key] = _percentile_rank(
            compare_val, pop, invert=(our_key in invert_set))
    return percentiles


def _get_deltas(stats: dict, pos: str) -> dict:
    """
    Compare player stat vs Twins positional average.
    Returns delta where positive always means the player is BETTER.
    """
    avgs      = TWINS_AVGS.get(pos, {})
    pitcher   = is_pitcher(pos)
    delta_inv = PITCHER_DELTA_INVERT if pitcher else BATTER_DELTA_INVERT
    deltas    = {}
    for stat_key, twins_avg in avgs.items():
        player_val = stats.get(stat_key)
        if player_val is None:
            continue
        raw = player_val - twins_avg
        if stat_key in delta_inv:
            raw = -raw
        deltas[stat_key] = round(raw, 3)
    return deltas

# ─────────────────────────────────────────────────────────────
# REBUILD SCORING
# ─────────────────────────────────────────────────────────────

GRADE_BANDS = [
    (90, "A+", "Elite"),
    (80, "A",  "Above-Average"),
    (70, "B+", "Solid Regular"),
    (60, "B",  "Average"),
    (50, "C+", "Below-Average"),
    (40, "C",  "Fringe"),
    (0,  "D",  "Replacement Level"),
]

def _letter_grade(score: float):
    for threshold, letter, label in GRADE_BANDS:
        if score >= threshold:
            return letter, label
    return "D", "Replacement Level"


def rebuild_score(stats: dict, pos: str, age: int, control: int) -> dict:
    """
    0-100 composite rebuild score.
      Pillar 1 — Performance  (60%): position-weighted percentile vs MLB peers
      Pillar 2 — Age/Control  (25%): peaks at 25 for pitchers / 27 for hitters
      Pillar 3 — Twins Impact (15%): improvement vs Twins positional average

    Hard Hit % is EXCLUDED from percentile scoring (Statcast all-events
    vs FanGraphs BIP-only populations are not comparable).
    It remains in the stat profile for descriptive context only.
    """
    pitcher = is_pitcher(pos)
    flags   = []

    # ── Pillar 1: Position-adjusted performance ───────────────
    percentiles = _get_percentiles(stats, pos)
    weights     = WEIGHTS.get(pos, WEIGHTS["OF"])

    w_sum, w_tot = 0.0, 0.0
    for stat_key, w in weights.items():
        pct = percentiles.get(stat_key)
        if pct is not None:
            w_sum += pct * w
            w_tot += w
    perf_score = w_sum / w_tot if w_tot > 0 else 50.0

    for stat, pct in percentiles.items():
        short = stat.replace(" Against","").replace(" %","%")
        if   pct >= 85: flags.append(f"Elite {short} ({_ordinal(pct)} pct)")
        elif pct >= 70: flags.append(f"Above-Avg {short} ({_ordinal(pct)} pct)")
        elif pct <= 20: flags.append(f"Poor {short} ({_ordinal(pct)} pct)")
        elif pct <= 35: flags.append(f"Below-Avg {short} ({_ordinal(pct)} pct)")

    # ── Pillar 2: Age / Control ───────────────────────────────
    # Pitchers age faster offensively; decay starts at 25 vs 27 for hitters
    peak_age  = 25 if pitcher else 27
    age_comp  = max(0.0, 100 - max(0, age - peak_age) * 7.0)
    ctrl_comp = min(control * 20.0, 100.0)
    age_ctrl  = age_comp * 0.6 + ctrl_comp * 0.4

    # ── Pillar 3: Twins Impact ────────────────────────────────
    deltas = _get_deltas(stats, pos)
    if deltas:
        # Sensitivity: how large a delta is "meaningful" at each stat
        SENS = {
            "xwOBA":              0.020,
            "xwOBA Against":      0.020,
            "Barrel %":           2.5,
            "Barrel % Against":   2.5,
            "K %":                3.0,
            "BB %":               2.5,
            "CSW %":              2.5,
        }
        norm_scores = []
        for stat_key, delta in deltas.items():
            sens   = SENS.get(stat_key, 3.0)
            normed = min(max((delta / sens) * 0.5 + 0.5, 0), 1) * 100
            norm_scores.append(normed)
        impact_score = float(np.mean(norm_scores))

        ups   = sum(1 for v in deltas.values() if v > 0)
        downs = sum(1 for v in deltas.values() if v < 0)
        total = ups + downs
        if total > 0:
            if ups >= total * 0.75:
                flags.append(f"Upgrades Twins {pos} in {ups}/{total} metrics")
            elif downs >= total * 0.75:
                flags.append(f"Downgrades Twins {pos} ({downs}/{total} worse)")
            else:
                flags.append(f"Mixed {pos} impact (+{ups}/-{downs})")
    else:
        impact_score = 50.0

    # ── Composite ────────────────────────────────────────────
    composite = round(
        min(max(perf_score*0.60 + age_ctrl*0.25 + impact_score*0.15, 0), 100),
    1)
    letter, label = _letter_grade(composite)

    # ── Archetype ─────────────────────────────────────────────
    if   age <= 26 and control >= 3: arch = "Cornerstone"
    elif age <= 33 and control >= 1: arch = "Bridge"
    elif control >= 1:               arch = "Trade Chip"
    else:                            arch = "DFA Candidate"

    return {
        "Rebuild Score":        composite,
        "Grade":                f"{letter} - {label}",
        "Archetype":            arch,
        "Position":             pos,
        "Flags":                ", ".join(flags) if flags else "-",
        "Perf Score (MLB %)":   round(perf_score, 1),
        "Age/Control Score":    round(age_ctrl, 1),
        "Twins Impact Score":   round(impact_score, 1),
        "MLB Percentiles":      percentiles,
        "Twins Deltas":         deltas,
    }

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

BG, CARD = "#1a1a2e", "#16213e"


def plot_arsenal(pitcher_name: str, arsenal: dict,
                 pitch_whiff: dict, save_dir: str):
    if not arsenal:
        return
    pitches = list(arsenal.keys())
    usage   = [arsenal[p]            for p in pitches]
    whiff   = [pitch_whiff.get(p, 0) for p in pitches]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, len(pitches))))
    fig.suptitle(f"{pitcher_name} - Arsenal (2022-2024)",
                 fontweight="bold", fontsize=13, color="white")
    fig.patch.set_facecolor(BG)
    colors = ["#e94560","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]

    for ax in (ax1, ax2):
        ax.set_facecolor(CARD)
        ax.tick_params(colors="white")
        ax.spines[["top","right","bottom","left"]].set_visible(False)
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    b1 = ax1.barh(pitches, usage, color=colors[:len(pitches)])
    ax1.set_xlabel("Usage %", color="white")
    ax1.set_title("Pitch Mix")
    for bar, val in zip(b1, usage):
        ax1.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%", va="center", color="white", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="white")

    b2 = ax2.barh(pitches, whiff, color=colors[:len(pitches)])
    ax2.set_xlabel("Whiff %", color="white")
    ax2.set_title("Swinging Strike %")
    ax2.axvline(25, color="#e94560", ls="--", lw=1, label="Elite (25%)")
    ax2.legend(facecolor=BG, labelcolor="white", fontsize=8)
    for bar, val in zip(b2, whiff):
        ax2.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%", va="center", color="white", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="white")

    plt.tight_layout()
    fname = f"{save_dir}/{pitcher_name.replace(' ','_')}_arsenal.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {fname}")


def plot_dashboard(summary_df: pd.DataFrame, all_rows: list, save_dir: str):
    """Two-panel: stacked sub-score bars + position-adjusted percentile heatmap."""
    n = len(summary_df)
    fig = plt.figure(figsize=(22, max(7, n * 1.6)))
    fig.patch.set_facecolor(BG)
    gs_layout = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.5], wspace=0.05)
    ax_bar  = fig.add_subplot(gs_layout[0])
    ax_heat = fig.add_subplot(gs_layout[1])

    names      = list(summary_df["Player"])
    scores     = summary_df["Rebuild Score"].fillna(0).tolist()
    archetypes = list(summary_df["Archetype"])
    positions  = list(summary_df["Position"])
    perf_s     = summary_df["Perf Score (MLB %)"].fillna(0).tolist()
    age_s      = summary_df["Age/Control Score"].fillna(0).tolist()
    impact_s   = summary_df["Twins Impact Score"].fillna(0).tolist()

    # ── Left: Stacked score bars ──────────────────────────────
    ax_bar.set_facecolor(CARD)
    y = np.arange(n)
    w_p, w_a, w_i = 0.60, 0.25, 0.15
    bp = [p * w_p for p in perf_s]
    ba = [a * w_a for a in age_s]
    bi = [i * w_i for i in impact_s]

    ax_bar.barh(y, bp, color="#3498db", height=0.65,
                label=f"MLB Perf ({int(w_p*100)}%)")
    ax_bar.barh(y, ba, color="#2ecc71", height=0.65, left=bp,
                label=f"Age/Control ({int(w_a*100)}%)")
    ax_bar.barh(y, bi, color="#f39c12", height=0.65,
                left=[p+a for p,a in zip(bp,ba)],
                label=f"Twins Impact ({int(w_i*100)}%)")

    for i, (score, arch, pos) in enumerate(zip(scores, archetypes, positions)):
        ax_bar.text(score+0.8, i,
                    f"{score:.0f}  [{pos}]  {arch}",
                    va="center", color="white", fontsize=8.5)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, color="white", fontsize=10)
    ax_bar.set_xlim(0, 122)
    ax_bar.set_xlabel("Rebuild Fit Score (0-100)", color="white")
    ax_bar.set_title("Rebuild Score by Component", color="white",
                     fontweight="bold", pad=10)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[["top","right","bottom","left"]].set_visible(False)
    ax_bar.axvline(60, color="white", ls="--", lw=0.8, alpha=0.3)
    ax_bar.legend(facecolor=BG, labelcolor="white", fontsize=8,
                  loc="lower right")

    # ── Right: Percentile heatmap ─────────────────────────────
    ax_heat.set_facecolor(CARD)
    all_keys = sorted({k for row in all_rows
                       for k in row["rebuild"].get("MLB Percentiles",{})})

    if all_keys:
        matrix = np.array([
            [row["rebuild"].get("MLB Percentiles",{}).get(k, np.nan)
             for k in all_keys]
            for row in all_rows
        ], dtype=float)

        im = ax_heat.imshow(matrix, aspect="auto", cmap="RdYlGn",
                            vmin=0, vmax=100, interpolation="nearest")
        for i in range(n):
            for j in range(len(all_keys)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax_heat.text(
                        j, i, _ordinal(val),
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold",
                        color="black" if 25 < val < 75 else "white")

        ax_heat.set_xticks(range(len(all_keys)))
        ax_heat.set_xticklabels(
            [k.replace(" Against","").replace(" %","%") for k in all_keys],
            rotation=35, ha="right", color="white", fontsize=8)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(names, color="white", fontsize=10)
        ax_heat.set_title("Position-Adjusted Percentile Ranks",
                          color="white", fontweight="bold", pad=10)
        ax_heat.tick_params(left=False, bottom=False)
        for sp in ax_heat.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="white")
        cb.ax.set_ylabel("Percentile", color="white", fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    else:
        ax_heat.text(0.5, 0.5, "No percentile data available",
                     ha="center", va="center", color="white",
                     fontsize=12, transform=ax_heat.transAxes)

    fig.suptitle("Minnesota Twins - Position-Aware Rebuild Dashboard",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = f"{save_dir}/twins_rebuild_dashboard.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────
# REPORT PRINTING
# ─────────────────────────────────────────────────────────────

# These stats are shown raw only — excluded from percentile comparison
# because Statcast all-events population != FanGraphs BIP-only population
DESCRIPTIVE_ONLY = {
    "Avg EV", "Max EV", "Avg EV Against",
    "Hard Hit %", "Hard Hit % Against"
}


def print_report(name, pos, age, contract, control, stats, rebuild):
    sep  = "-" * 62
    role = "Pitcher" if is_pitcher(pos) else "Batter"
    print(f"\n{'='*62}")
    print(f"  {name}  |  {pos} ({role})  |  Age {age}  "
          f"|  {contract}  |  {control} yr control")
    print(f"{'='*62}")

    if is_pitcher(pos):
        ordered = ["Avg Velo","Max Velo","Velo Trend (y/y)",
                   "Avg Spin (FB)","Whiff %","CSW %","K %","BB %",
                   "xwOBA Against","Avg EV Against",
                   "Hard Hit % Against","Barrel % Against","Pitches"]
    else:
        ordered = ["xwOBA","xBA","xSLG","Barrel %","Hard Hit %",
                   "Avg EV","Max EV","EV Trend (y/y)","Sweet Spot %",
                   "K %","BB %","Whiff %","Chase %",
                   "GB %","LD %","FB %","Pull %","Oppo %","PA"]

    print(f"\n  STATCAST PROFILE (2022-2024 Combined):")
    print(f"  {sep}")
    for k in ordered:
        v = stats.get(k)
        if v is None:
            continue
        # Format value — xwOBA always 3dp, trend floats 1dp, integers as int
        if k == "xwOBA Against" or k == "xwOBA":
            v_str = f"{v:.3f}"
        elif isinstance(v, float) and v == int(v) and "%" not in k:
            v_str = str(int(v))
        else:
            v_str = str(v)

        unit  = (" mph" if ("Velo" in k or k in
                             ("Avg EV","Max EV","Avg EV Against")) else
                 " rpm" if "Spin" in k else
                 "%" if "%" in k else "")
        trend = (" ^" if "Trend" in k and isinstance(v,float) and v>0 else
                 " v" if "Trend" in k and isinstance(v,float) and v<0 else "")
        note  = "  (descriptive only)" if k in DESCRIPTIVE_ONLY else ""
        print(f"  {k:<30} {v_str}{unit}{trend}{note}")

    if is_pitcher(pos) and "Arsenal" in stats:
        print(f"\n  ARSENAL BREAKDOWN:")
        print(f"  {sep}")
        for pitch, pct in stats["Arsenal"].items():
            wh = stats.get("Pitch Whiff %",{}).get(pitch,"--")
            ve = stats.get("Pitch Velo",{}).get(pitch,"--")
            ws = f"{wh}% whiff" if isinstance(wh,float) else "--"
            vs = f"{ve} mph"    if isinstance(ve,float) else "--"
            print(f"  {pitch:<6}  {pct:>5.1f}% usage   {vs:<12}  {ws}")

    pcts = rebuild.get("MLB Percentiles",{})
    if pcts:
        print(f"\n  PERCENTILE RANKINGS vs {pos} peers (2024 FanGraphs):")
        print(f"  {sep}")
        for stat, pct in pcts.items():
            bar  = "#" * int(pct/5) + "." * (20 - int(pct/5))
            tier = "[ELITE]"   if pct >= 85 else \
                   "[GOOD]"    if pct >= 70 else \
                   "[AVG]"     if pct >= 40 else \
                   "[BELOW]"   if pct >= 20 else "[POOR]"
            print(f"  {stat:<30} [{bar}]  {_ordinal(pct):>6}  {tier}")

    deltas = rebuild.get("Twins Deltas",{})
    if deltas:
        print(f"\n  TWINS {pos} ROSTER IMPACT (vs positional average):")
        print(f"  {sep}")
        for stat, delta in deltas.items():
            arrow = "^^" if delta > 0 else ("vv" if delta < 0 else "--")
            tag   = "UPGRADE" if delta > 0 else "DOWNGRADE"
            print(f"  {stat:<30} {delta:>+.3f}  {arrow} {tag}")

    print(f"\n  REBUILD EVALUATION:")
    print(f"  {sep}")
    print(f"  Composite Score:              {rebuild['Rebuild Score']}/100"
          f"  ({rebuild['Grade']})")
    print(f"  +-- Perf Score (pos-adj %):   {rebuild['Perf Score (MLB %)']}/100")
    print(f"  +-- Age/Control Score:        {rebuild['Age/Control Score']}/100")
    print(f"  +-- Twins Impact Score:       {rebuild['Twins Impact Score']}/100")
    print(f"  Archetype:  {rebuild['Archetype']}")
    print(f"  Flags:      {rebuild['Flags']}")

# ─────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────

def generate_csv(rows: list, save_dir: str) -> pd.DataFrame:
    flat_rows = []
    for r in rows:
        flat = {
            "Player":               r["name"],
            "Position":             r["pos"],
            "Age":                  r["age"],
            "Contract":             r["contract"],
            "Control (yrs)":        r["control"],
            "Rebuild Score":        r["rebuild"]["Rebuild Score"],
            "Grade":                r["rebuild"]["Grade"],
            "Archetype":            r["rebuild"]["Archetype"],
            "Perf Score (MLB %)":   r["rebuild"]["Perf Score (MLB %)"],
            "Age/Control Score":    r["rebuild"]["Age/Control Score"],
            "Twins Impact Score":   r["rebuild"]["Twins Impact Score"],
            "Flags":                r["rebuild"]["Flags"],
        }
        flat.update({k: v for k, v in r["stats"].items()
                     if not isinstance(v, dict)})
        flat_rows.append(flat)
    df = pd.DataFrame(flat_rows)
    fname = f"{save_dir}/twins_rebuild_summary.csv"
    df.to_csv(fname, index=False)
    print(f"\n  Summary saved: {fname}")
    return df

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  MINNESOTA TWINS - REBUILD ANALYSIS ENGINE")
    print(f"  Seasons: {', '.join(ANALYSIS_YEARS)}  |  "
          f"Run: {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 62)

    all_rows = []

    for name, pos, age, contract, control in PLAYERS:
        print(f"\n>> Processing: {name}  [{pos}]")
        player_id = get_player_id(name)
        if player_id is None:
            print(f"  No MLBAM ID -- skipping.")
            continue

        print(f"  MLBAM ID: {player_id}")
        pitcher = is_pitcher(pos)
        data    = fetch_all_seasons(player_id, pitcher)

        if data.empty:
            print(f"  No Statcast data -- skipping.")
            continue

        stats = analyze_pitcher(data) if pitcher else analyze_batter(data)

        if pitcher:
            plot_arsenal(name, stats.get("Arsenal",{}),
                         stats.get("Pitch Whiff %",{}), OUTPUT_DIR)

        rebuild = rebuild_score(stats, pos, age, control)
        print_report(name, pos, age, contract, control, stats, rebuild)

        all_rows.append({
            "name":name, "pos":pos, "age":age,
            "contract":contract, "control":control,
            "stats":stats, "rebuild":rebuild,
        })

    print(f"\n\n{'='*62}")
    print("  REBUILD SUMMARY")
    print(f"{'='*62}")

    if all_rows:
        df = generate_csv(all_rows, OUTPUT_DIR)
        print("\n" + df[["Player","Position","Age","Contract",
                          "Control (yrs)","Rebuild Score",
                          "Archetype","Flags"]].to_string(index=False))
        plot_dashboard(df, all_rows, OUTPUT_DIR)

    print(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")
    print("  Arsenal PNG per pitcher")
    print("  twins_rebuild_dashboard.png")
    print("  twins_rebuild_summary.csv")


if __name__ == "__main__":
    main()