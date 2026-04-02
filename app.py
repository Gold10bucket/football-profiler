"""
Football Player Profiler
Step 1 → Upload CSVs  |  Step 2 → Pick position  |  Step 3 → Browse results
Deploy: streamlit run app.py  OR  push to GitHub → share.streamlit.io
"""

import io
import re
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Football Profiler",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Constants ────────────────────────────────────────────────────────────────
WY_WEIGHT = 0.70
SI_WEIGHT = 0.30
FINISHING = {"Buts hors penaltys / 90", "xG / 90", "Tirs / 90", "% conversion buts"}

# Metrics where lower raw value = better (score gets flipped after percentile rank)
INVERTED = {
    "Fautes / 90 (inversé)",
    "Buts concédés / 90 (inversé)",
}

# ─── Wyscout column map ───────────────────────────────────────────────────────
# None = no mapping available in this dataset for this metric
WY_MAP = {
    # Finishing
    "Buts hors penaltys / 90":            "Non-penalty goals per 90",
    "xG / 90":                            "npxG per 90",
    "Tirs / 90":                          "Shots per 90",
    "% conversion buts":                  "Goal conversion, %",
    # Offensive
    "xA / 90":                            "xA per 90",
    "Passes clés / 90":                   "Shot assists per 90",
    "Passes décisives sur tir / 90":      "Shot assists per 90",
    "Courses progressives / 90":          "Progressive runs per 90",
    "Passes progressives / 90":           "Progressive passes per 90",
    "Touches dans la surface / 90":       "Touches in box per 90",
    "Passes reçues / 90":                 "Received_Passes",
    "% dribbles réussis":                 "Successful dribbles, %",
    "Dribbles / 90":                      "Accelerations per 90",    # proxy: acceleration ≈ dribble attempt
    "Centres / 90":                       "Crosses per 90",
    "% centres précis":                   "Accurate crosses, %",
    "Centres profonds / 90":              None,                       # no deep-cross col in Wyscout
    "Passes vers le dernier tiers / 90":  "Progressive passes per 90",# proxy: progressive pass enters final 3rd
    "Passes vers l'avant / 90":           "Passes per 90",            # proxy: pass volume (different from prog.)
    "Passes / 90":                        "Passes per 90",
    "% passes précises":                  "Accurate passes, %",
    "% passes longues précises":          "Accurate long passes, %",
    "% passes courtes/moyennes précises": "Accurate short / medium passes, %",
    "Passes intelligentes / 90":          "Smart passes per 90",
    "% passes progressives précises":     "Accurate long passes, %",  # proxy: long-pass accuracy ≈ prog-pass accuracy
    "Passes longues reçues / 90":         "Received_Passes",          # proxy: total received passes
    # Aerial
    "Duels aériens / 90":                 "Aerial duels won per 90",
    "% duels aériens gagnés":             "Aerial duels won, %",
    # Defensive
    "Duels défensifs / 90":               None,                       # no raw def-duel count in Wyscout
    "% duels défensifs gagnés":           "Defensive duels won, %",
    "Interceptions / 90":                 "PAdj Interceptions",
    "Actions défensives réussies / 90":   "Successful defensive actions per 90",
    "Actions offensives réussies / 90":   None,
    "Tirs contrés / 90":                  "Shots blocked per 90",
    "Fautes subies / 90":                 "Fouls suffered per 90",
    "Fautes / 90 (inversé)":              "Fouls per 90",
    # Goalkeeper
    "% arrêts (Save rate)":               "Save rate, %",
    "Buts prévenus / 90":                 "Prevented goals per 90",
    "Tirs encaissés / 90":                "Shots against per 90",
    "Buts concédés / 90 (inversé)":       "Conceded goals per 90",
    "Sorties / 90":                       "Exits per 90",
}

# ─── Position mapping ─────────────────────────────────────────────────────────
# Wyscout: primary position code → position group
_WY_POS = {
    "CF": "Forwards", "RWF": "Forwards", "LWF": "Forwards",
    "AMF": "Forwards",                                          # attacking mid = forward context
    "RW": "Wide Players", "LW": "Wide Players",
    "RWB": "Wide Players", "LWB": "Wide Players",
    "RB": "Wide Players", "LB": "Wide Players",
    "CMF": "Midfielders", "RCMF": "Midfielders", "LCMF": "Midfielders",
    "RCMF3": "Midfielders", "LCMF3": "Midfielders",
    "DMF": "Midfielders", "RDMF": "Midfielders", "LDMF": "Midfielders",
    "CB": "Centre Backs", "RCB": "Centre Backs", "LCB": "Centre Backs",
    "RCB3": "Centre Backs", "LCB3": "Centre Backs",
    "GK": "Goalkeepers",
}

# SICS Ruolo dettagliato → position group (takes priority over Ruolo)
_SI_DETAILED = {
    "Quinto destro": "Wide Players", "Quinto sinistro": "Wide Players",
    "Terzino destro": "Wide Players", "Terzino sinistro": "Wide Players",
    "Esterno alto destro": "Wide Players", "Esterno alto sinistro": "Wide Players",
    "Ala destra": "Wide Players", "Ala sinistra": "Wide Players",
    "Difensore centrale": "Centre Backs",
    "Libero": "Centre Backs",
    "Mediano": "Midfielders", "Mezzala": "Midfielders",
    "Regista": "Midfielders", "Trequartista": "Midfielders",
    "Centravanti": "Forwards", "Seconda punta": "Forwards",
    "Portiere": "Goalkeepers",
}
_SI_RUOLO = {
    "Attaccante": "Forwards",
    "Centrocampista": "Midfielders",
    "Difensore": "Centre Backs",
    "Portiere": "Goalkeepers",
}

POSITION_GROUPS = ["Forwards", "Wide Players", "Midfielders", "Goalkeepers", "Centre Backs"]


def wy_position_group(pos_str: str) -> str:
    """Parse Wyscout Position string (may be comma-separated) → position group."""
    if not isinstance(pos_str, str):
        return "Unknown"
    primary = pos_str.split(",")[0].strip()
    return _WY_POS.get(primary, "Unknown")


def si_position_group(ruolo: str, ruolo_det: str) -> str:
    """Map SICS Ruolo + Ruolo dettagliato → position group."""
    if isinstance(ruolo_det, str) and ruolo_det in _SI_DETAILED:
        return _SI_DETAILED[ruolo_det]
    if isinstance(ruolo, str) and ruolo in _SI_RUOLO:
        return _SI_RUOLO[ruolo]
    return "Unknown"


# ─── Name / age helpers ───────────────────────────────────────────────────────
def _norm(name: str) -> str:
    """Uppercase ASCII, strip accents and punctuation."""
    nfkd = unicodedata.normalize("NFKD", str(name).upper())
    return re.sub(r"[^A-Z0-9 ]", "", nfkd.encode("ASCII", "ignore").decode()).strip()


def _birth_year_from_nazionalita(s) -> int | None:
    """Extract birth year from SICS fields like 'Italia (\'04)' or '2004' or '04'."""
    if pd.isna(s):
        return None
    s = str(s)
    # 4-digit year anywhere in string
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if m:
        return int(m.group(1))
    # 2-digit year in parentheses: ('04)
    m = re.search(r"\('?(\d{2})'?\)", s)
    if m:
        y = int(m.group(1))
        return 2000 + y if y <= 30 else 1900 + y
    return None


def _wy_birth_year(age) -> int | None:
    """Estimate birth year from Wyscout Age (current season 2025-26)."""
    try:
        return 2025 - int(age)
    except (TypeError, ValueError):
        return None


def _wy_last_name(player: str) -> str:
    """'A. Guerrisi' → 'GUERRISI'"""
    parts = _norm(player).split()
    return parts[-1] if parts else _norm(player)


def _si_last_name(player: str) -> str:
    """'GUERRISI ANDREA' → 'GUERRISI' (SICS: last name first)"""
    parts = _norm(player).split()
    return parts[0] if parts else _norm(player)


def _si_first_initial(player: str) -> str:
    """'GUERRISI ANDREA' → 'A'"""
    parts = _norm(player).split()
    return parts[1][0] if len(parts) > 1 else ""


def _wy_first_initial(player: str) -> str:
    """'A. Guerrisi' → 'A'"""
    m = re.match(r"([A-Z])", _norm(player))
    return m.group(1) if m else ""


# ─── SICS deduplication (same player, different clubs) ───────────────────────
def _dedup_sics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows for the same player appearing at multiple clubs.
    Identifies duplicates by: normalised name + birth year.
    Counting stats are summed; categorical fields keep first occurrence.
    """
    df = df.copy()
    df["_by"] = df["Nazionalità"].apply(_birth_year_from_nazionalita).fillna(0).astype(int)
    df["_nn"] = df["Giocatori"].apply(_norm)
    df["_key"] = df["_nn"] + "_" + df["_by"].astype(str)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = [c for c in df.columns if c not in num_cols and not c.startswith("_")]

    agg: dict = {c: "sum" for c in num_cols}
    for c in str_cols:
        agg[c] = "first"
    # For team: show both if different
    agg["Squadra"] = lambda x: " / ".join(sorted(set(x.dropna().astype(str))))

    deduped = df.groupby("_key", sort=False).agg(agg).reset_index(drop=True)
    # Restore clean player name (first occurrence)
    name_map = df.groupby("_key")["Giocatori"].first()
    deduped["Giocatori"] = deduped.apply(
        lambda r: name_map.get(r.name, r["Giocatori"]), axis=1
    )
    deduped["_by"] = df.groupby("_key")["_by"].first().values
    return deduped


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_wyscout(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.replace("\n", "_").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Defensive column lookup helpers
    def _col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    pos_col  = _col(["Position", "Pos", "position"])
    age_col  = _col(["Age", "age", "Età"])
    name_col = _col(["Player", "player", "Name"])
    team_col = _col(["Team", "team", "Club", "Squadra"])

    # Drop all non-player rows (sub-headers, title rows, totals, etc.)
    # A real player row always has a numeric Age value.
    if age_col:
        df = df[pd.to_numeric(df[age_col], errors="coerce").notna()].reset_index(drop=True)

    df["_position_group"] = df[pos_col].apply(wy_position_group) if pos_col else "Unknown"
    df["_birth_year"]     = df[age_col].apply(_wy_birth_year)    if age_col  else None
    df["_last_name"]      = df[name_col].apply(_wy_last_name)    if name_col else ""
    df["_first_initial"]  = df[name_col].apply(_wy_first_initial) if name_col else ""
    df["_team_norm"]      = df[team_col].fillna("").apply(lambda x: _norm(str(x))) if team_col else ""
    league_col = _col(["League", "league", "Competition", "Lega"])
    df["_league"] = df[league_col].fillna("") if league_col else ""

    # Detect minutes column (Wyscout exports vary; \n→_ normalisation above)
    _min_candidates = ["Minutes played", "Minutes_played", "Minutes", "Mins played",
                       "Mins_played", "Mins", "Min", "Min."]
    _min_col = next((c for c in _min_candidates if c in df.columns), None)
    if _min_col is None:
        # Fallback: first column whose normalised name contains "minut" or "min"
        _min_col = next((c for c in df.columns
                         if re.search(r"\bmin", c, re.IGNORECASE)), None)
    if _min_col:
        df["_minutes"] = pd.to_numeric(df[_min_col], errors="coerce").fillna(0)
    else:
        df["_minutes"] = np.nan  # not available — won't be filtered out

    # ── Step 2a: percentile-rank all WY_MAP stat columns (0–1) ─────────────────
    wy_target_cols = {v for v in WY_MAP.values() if v is not None}
    for col in wy_target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").rank(pct=True, na_option="bottom")

    return df


@st.cache_data(show_spinner=False)
def process_sics(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))

    # 1 — Dedup players across clubs
    raw = _dedup_sics(raw)

    minutes = raw["Minuti Giocati"].fillna(0)
    p90 = minutes.replace(0, np.nan) / 90

    def rat(num, den):
        return (num.fillna(0) / den.fillna(0).replace(0, np.nan)).fillna(0)

    # 2 — Compute per-90 raw values
    computed = {
        "Buts hors penaltys / 90":            (raw["Gol"].fillna(0) - raw["Gol su Rigore"].fillna(0)) / p90,
        "xG / 90":                             raw["xG"].fillna(0) / p90,
        "Tirs / 90":                           raw["Tiri"].fillna(0) / p90,
        "% conversion buts":                   rat(raw["Gol"].fillna(0), raw["Tiri"].fillna(0)),
        "xA / 90":                             raw["xA"].fillna(0) / p90,
        "Passes clés / 90":                    raw["Passaggi Chiave"].fillna(0) / p90,
        "Passes décisives sur tir / 90":       raw["Assist"].fillna(0) / p90,
        "Courses progressives / 90":           raw["xT da conduzioni"].fillna(0) / p90,
        "Passes progressives / 90":            raw["xT da passaggi"].fillna(0) / p90,
        "Passes vers le dernier tiers / 90":   raw["Passaggi Chiave"].fillna(0) / p90,  # proxy: key passes reach final 3rd
        "Passes vers l'avant / 90":            raw["Passaggi in Area"].fillna(0) / p90,
        "Touches dans la surface / 90":        raw["Tocchi in area avversaria"].fillna(0) / p90,
        "Passes reçues / 90":                  raw["Passaggi Ricevuti"].fillna(0) / p90,
        "Passes longues reçues / 90":          raw["Passaggi Chiave Ricevuti"].fillna(0) / p90,  # proxy: key passes received
        "Dribbles / 90":                       raw["Dribbling"].fillna(0) / p90,
        "% dribbles réussis":                  rat(raw["Dribbling Positivo"].fillna(0), raw["Dribbling"].fillna(0)),
        "Centres / 90":                        raw["Palle Laterali"].fillna(0) / p90,
        "% centres précis":                    rat(raw["Cross Riusciti"].fillna(0), raw["Cross"].fillna(0)),
        "Centres profonds / 90":               raw["Early Cross"].fillna(0) / p90,
        "Passes / 90":                         raw["Passaggi"].fillna(0) / p90,
        "% passes précises":                   rat(raw["Passaggi Riusciti"].fillna(0), raw["Passaggi"].fillna(0)),
        "% passes longues précises":           rat(raw["Lanci Positivi"].fillna(0), raw["Lanci"].fillna(0)),
        "% passes courtes/moyennes précises":  rat(raw["Passaggi Riusciti"].fillna(0), raw["Passaggi"].fillna(0)),
        "% passes progressives précises":      rat(raw["xT da passaggi"].fillna(0),   # proxy: xT per pass attempt
                                                   raw["Passaggi"].fillna(0)),
        "Duels aériens / 90":                  raw["Duelli Aerei"].fillna(0) / p90,
        "% duels aériens gagnés":              rat(raw["Duelli Aerei Vinti"].fillna(0), raw["Duelli Aerei"].fillna(0)),
        "Duels défensifs / 90":                raw["Duelli Difensivi"].fillna(0) / p90,
        "% duels défensifs gagnés":            rat(raw["Duelli Difensivi Vinti"].fillna(0), raw["Duelli Difensivi"].fillna(0)),
        "Interceptions / 90":                  raw["Palle Recuperate"].fillna(0) / p90,
        # Improved: ball recoveries in opponent's half (pressing actions) — was Interventi Positivi (sparse)
        "Actions défensives réussies / 90":    raw["Palle Recuperate Meta Campo Avversaria"].fillna(0) / p90,
        "Actions offensives réussies / 90":    raw["Dribbling Positivo"].fillna(0) / p90,
        "Fautes subies / 90":                  raw["Falli Subiti"].fillna(0) / p90,
        "Fautes / 90 (inversé)":               raw["Falli Fatti"].fillna(0) / p90,
        "% arrêts (Save rate)":                rat(raw["Parate"].fillna(0), raw["Tiri in Porta Subiti"].fillna(0)),
        "Buts prévenus / 90":                  raw["GK Goals Prevented"].fillna(0) / p90,
        "Tirs encaissés / 90":                 raw["Tiri in Porta Subiti"].fillna(0) / p90,
        "Buts concédés / 90 (inversé)":        raw["Gol Subiti dal Portiere"].fillna(0) / p90,
        "Sorties / 90":                        raw["Uscite"].fillna(0) / p90,
    }

    # 3 — Percentile rank within this dataset
    ranked = {}
    for metric, series in computed.items():
        pct = series.rank(pct=True, na_option="bottom")
        if metric in INVERTED:
            pct = 1 - pct
        ranked[metric] = pct.values

    result = pd.DataFrame(ranked)
    result["_player"]         = raw["Giocatori"].values
    result["_team"]           = raw["Squadra"].values
    result["_minutes"]        = minutes.values
    result["_position_group"] = raw.apply(
        lambda r: si_position_group(r.get("Ruolo", ""), r.get("Ruolo dettagliato", "")), axis=1
    )
    result["_birth_year"]     = raw["_by"].values if "_by" in raw.columns else None
    result["_last_name"]      = raw["Giocatori"].apply(_si_last_name)
    result["_first_initial"]  = raw["Giocatori"].apply(_si_first_initial)
    result.set_index("_player", inplace=True)
    return result


# ─── Step 1: Match players between Wyscout and SICS ──────────────────────────
def _team_words(t: str) -> set:
    """Normalised set of meaningful words from a team name."""
    stop = {"FC", "AC", "SC", "SSD", "ASD", "SS", "US", "AS", "CF", "1", "2"}
    return {w for w in _norm(str(t)).split() if w not in stop and len(w) > 1}


def match_players(wy_df: pd.DataFrame | None, si_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per unique player:
      wy_player, si_player, match_type, Player (display), Team,
      _position_group, _birth_year, Minutes, _si (alias for si_player)
    match_type: 'exact' | 'year±1' | 'year±2' | 'name' | 'lastname' | 'wy_only' | 'si_only'
    """
    rows: list[dict] = []
    seen_si: set[str] = set()

    # ── Build SICS lookup buckets ──────────────────────────────────────────────
    # bucket: (last, fi, by) → list of si_names
    from collections import defaultdict
    si_bucket: dict[tuple, list[str]] = defaultdict(list)
    si_ln_bucket: dict[str, list[str]] = defaultdict(list)

    if si_df is not None:
        for si_name in si_df.index:
            ln = si_df.loc[si_name, "_last_name"] if "_last_name" in si_df.columns else _si_last_name(si_name)
            fi = si_df.loc[si_name, "_first_initial"] if "_first_initial" in si_df.columns else _si_first_initial(si_name)
            by_raw = si_df.loc[si_name, "_birth_year"] if "_birth_year" in si_df.columns else None
            by = int(by_raw) if by_raw and not pd.isna(by_raw) else 0
            si_bucket[(ln, fi, by)].append(si_name)
            si_ln_bucket[ln].append(si_name)

    def _team_sim(wy_team: str, si_name: str) -> float:
        """Word-overlap score between Wyscout team and SICS team."""
        if si_df is None:
            return 0.0
        si_team = si_df.loc[si_name, "_team"] if "_team" in si_df.columns else ""
        wy_w = _team_words(wy_team)
        si_w = _team_words(str(si_team))
        if not wy_w or not si_w:
            return 0.0
        return len(wy_w & si_w) / max(len(wy_w), len(si_w))

    def _pick(candidates: list[str], wy_team: str) -> str:
        """From a list of candidates, prefer the one with best team similarity."""
        if len(candidates) == 1:
            return candidates[0]
        best = max(candidates, key=lambda c: _team_sim(wy_team, c))
        return best

    def _find_si(last: str, fi: str, by: int | None, wy_team: str) -> tuple[str | None, str]:
        by = int(by) if by and not pd.isna(by) else 0
        # 1. Exact
        if si_bucket.get((last, fi, by)):
            return _pick(si_bucket[(last, fi, by)], wy_team), "exact"
        # 2. ±1 year
        for d in (-1, 1):
            if si_bucket.get((last, fi, by + d)):
                return _pick(si_bucket[(last, fi, by + d)], wy_team), "year±1"
        # 3. ±2 years
        for d in (-2, 2):
            if si_bucket.get((last, fi, by + d)):
                return _pick(si_bucket[(last, fi, by + d)], wy_team), "year±2"
        # 4. Name + initial across all birth years
        cands = [c for key, lst in si_bucket.items() if key[0] == last and key[1] == fi for c in lst]
        if cands:
            return _pick(cands, wy_team), "name"
        # 5. Last name only
        if si_ln_bucket.get(last):
            return _pick(si_ln_bucket[last], wy_team), "lastname"
        return None, "wy_only"

    # ── Step 3a: Process Wyscout players ──────────────────────────────────────
    if wy_df is not None:
        for _, row in wy_df.iterrows():
            last    = row.get("_last_name", _wy_last_name(row["Player"]))
            fi      = row.get("_first_initial", _wy_first_initial(row["Player"]))
            by      = row.get("_birth_year")
            wy_team = str(row.get("Team", ""))
            wy_mins = row.get("_minutes")

            si_name, mtype = _find_si(last, fi, by, wy_team)
            if si_name:
                seen_si.add(si_name)
                si_mins = si_df.loc[si_name, "_minutes"] if si_df is not None else np.nan
                # Use SICS minutes if available (more accurate), else Wyscout
                minutes = si_mins if pd.notna(si_mins) and si_mins > 0 else wy_mins
                team    = si_df.loc[si_name, "_team"] if si_df is not None else wy_team
                pos_grp = row.get("_position_group", "Unknown")
            else:
                si_mins = np.nan
                minutes = wy_mins
                team    = wy_team
                pos_grp = row.get("_position_group", "Unknown")

            rows.append({
                "Player":          row["Player"],
                "Team":            team,
                "League":          row.get("_league", ""),
                "Position":        row.get("Position", ""),
                "_position_group": pos_grp,
                "_birth_year":     by,
                "Minutes":         minutes,
                "_si":             si_name,
                "_match_type":     mtype,
            })

    # ── Step 3b: SICS-only players ────────────────────────────────────────────
    if si_df is not None:
        for si_name in si_df.index:
            if si_name in seen_si:
                continue
            rows.append({
                "Player":          si_name,
                "Team":            si_df.loc[si_name, "_team"],
                "League":          "",
                "Position":        si_df.loc[si_name, "_position_group"],
                "_position_group": si_df.loc[si_name, "_position_group"],
                "_birth_year":     si_df.loc[si_name, "_birth_year"] if "_birth_year" in si_df.columns else None,
                "Minutes":         si_df.loc[si_name, "_minutes"],
                "_si":             si_name,
                "_match_type":     "si_only",
            })

    return pd.DataFrame(rows)


def merge_players(wy_df: pd.DataFrame | None, si_df: pd.DataFrame | None) -> pd.DataFrame:
    """Thin wrapper kept for compatibility — delegates to match_players."""
    return match_players(wy_df, si_df)


# ─── Scoring ──────────────────────────────────────────────────────────────────
def score_one(wy_row, si_row, metrics, no_finishing=False):
    """
    Returns (wy_score, si_score, global_score, coverage_pct).
    Scores are in [0,1]. coverage_pct = % of total weight backed by real data.
    """
    wy_num = wy_den = si_num = si_den = 0.0
    covered_w = total_w = 0.0

    for metric, weight in metrics:
        if no_finishing and metric in FINISHING:
            continue
        total_w += weight

        wy_col = WY_MAP.get(metric)
        got_wy = got_si = False

        if wy_col and wy_row is not None and wy_col in wy_row.index:
            v = pd.to_numeric(wy_row[wy_col], errors="coerce")
            if pd.notna(v):
                v = float(v)
                if metric in INVERTED:
                    v = 1 - v
                wy_num += v * weight
                wy_den += weight
                got_wy = True

        if si_row is not None and metric in si_row.index:
            val = si_row[metric]
            if pd.notna(val):
                si_num += float(val) * weight
                si_den += weight
                got_si = True

        if got_wy or got_si:
            covered_w += weight

    wy_s = wy_num / wy_den if wy_den > 0 else None
    si_s = si_num / si_den if si_den > 0 else None
    coverage = covered_w / total_w if total_w > 0 else 0.0

    if wy_s is None and si_s is None:
        return None, None, None, coverage

    if wy_s is not None and si_s is not None:
        gl = WY_WEIGHT * wy_s + SI_WEIGHT * si_s
    else:
        gl = wy_s if wy_s is not None else si_s

    return wy_s, si_s, gl, coverage


def _metric_vector(wy_row, si_row, metrics):
    """Return a numpy array of combined metric scores (0–1) for a player."""
    vec = []
    for metric, _ in metrics:
        wy_col = WY_MAP.get(metric)
        v_wy = np.nan
        if wy_col and wy_row is not None and wy_col in wy_row.index:
            raw = pd.to_numeric(wy_row[wy_col], errors="coerce")
            if pd.notna(raw):
                v_wy = float(raw)
                if metric in INVERTED:
                    v_wy = 1 - v_wy
        v_si = np.nan
        if si_row is not None and metric in si_row.index:
            raw = pd.to_numeric(si_row[metric], errors="coerce")
            if pd.notna(raw):
                v_si = float(raw)
        # Combined: prefer average of both, fall back to whichever is available
        if not np.isnan(v_wy) and not np.isnan(v_si):
            vec.append(WY_WEIGHT * v_wy + SI_WEIGHT * v_si)
        elif not np.isnan(v_wy):
            vec.append(v_wy)
        elif not np.isnan(v_si):
            vec.append(v_si)
        else:
            vec.append(0.0)
    return np.array(vec, dtype=float)


def find_similar(player_name, profile_name, master, wy_df, si_df, n=15, mode="profile", min_minutes=0):
    """
    Return DataFrame of most similar players using cosine similarity.
    mode='profile' → metrics from the selected profile only
    mode='full'    → all metrics across every profile (full fingerprint)
    """
    if mode == "profile":
        metrics = PROFILES[profile_name]["metrics"]
    else:
        # Deduplicated union of all metrics across all profiles (weight=1 each)
        seen: set = set()
        metrics = []
        for p in PROFILES.values():
            for m, _ in p["metrics"]:
                if m not in seen:
                    metrics.append((m, 1))
                    seen.add(m)

    wy_row = None
    if wy_df is not None:
        mask = wy_df["Player"] == player_name
        if mask.any():
            wy_row = wy_df[mask].iloc[0]
    si_row = _get_si_row(player_name, si_df, master)
    ref_vec = _metric_vector(wy_row, si_row, metrics)
    ref_norm = np.linalg.norm(ref_vec)

    score_metrics = PROFILES[profile_name]["metrics"]
    rows = []
    for _, p in master.iterrows():
        if p["Player"] == player_name:
            continue
        mins = float(p["Minutes"]) if pd.notna(p.get("Minutes")) else 0
        if min_minutes > 0 and mins > 0 and mins < min_minutes:
            continue
        pw_row = None
        if wy_df is not None:
            mask = wy_df["Player"] == p["Player"]
            if mask.any():
                pw_row = wy_df[mask].iloc[0]
        ps_row = _get_si_row(p["Player"], si_df, master)
        vec = _metric_vector(pw_row, ps_row, metrics)
        norm = np.linalg.norm(vec)
        cosine_sim = float(np.dot(ref_vec, vec) / (ref_norm * norm)) if ref_norm > 0 and norm > 0 else 0.0
        by   = p.get("_birth_year")
        _, _, gl, _ = score_one(pw_row, ps_row, score_metrics, no_finishing=False)
        rows.append({
            "Player":     p["Player"],
            "Team":       p["Team"],
            "League":     p.get("League", ""),
            "Age":        (2025 - int(by)) if pd.notna(by) else "—",
            "Mins":       int(mins) if mins else "—",
            "Score":      round(gl * 100, 1) if gl is not None else None,
            "Similarity": round(cosine_sim * 100, 1),
        })

    return (pd.DataFrame(rows)
            .sort_values("Similarity", ascending=False)
            .head(n)
            .reset_index(drop=True))


def _get_si_row(player_name, si_df, master):
    """Resolve SICS row for a player via master list."""
    if si_df is None:
        return None
    if player_name in si_df.index:
        return si_df.loc[player_name]
    if master is not None:
        pm = master[master["Player"] == player_name]
        if not pm.empty:
            si_name = pm.iloc[0].get("_si")
            if si_name and pd.notna(si_name) and si_name in si_df.index:
                return si_df.loc[si_name]
    return None


def build_export(master, wy_df, si_df):
    """
    Build a wide CSV with one row per player containing:
    - identity columns (Player, Team, Position, Birth Year, Minutes)
    - global score for every profile
    - raw percentile score for every metric (WY and SI separately)
    Suitable for feeding an AI to suggest profiles to monitor.
    """
    # Collect all unique metrics across all profiles
    all_metrics = []
    seen = set()
    for p in PROFILES.values():
        for m, _ in p["metrics"]:
            if m not in seen:
                all_metrics.append(m)
                seen.add(m)

    rows = []
    for _, player in master.iterrows():
        wy_row = None
        if wy_df is not None:
            mask = wy_df["Player"] == player["Player"]
            if mask.any():
                wy_row = wy_df[mask].iloc[0]
        si_row = _get_si_row(player["Player"], si_df, master)

        row = {
            "Player":      player["Player"],
            "Team":        player["Team"],
            "Position":    player.get("_position_group", ""),
            "Birth Year":  player.get("_birth_year", ""),
            "Minutes":     (int(player["Minutes"]) if pd.notna(player.get("Minutes")) and player.get("Minutes") not in ("", "—") else ""),
        }

        # Profile scores (with and without finishing)
        for pname, pdata in PROFILES.items():
            _, _, gl, _ = score_one(wy_row, si_row, pdata["metrics"])
            _, _, gl_nf, _ = score_one(wy_row, si_row, pdata["metrics"], no_finishing=True)
            row[f"Score: {pname}"] = round(gl * 100, 1) if gl is not None else ""
            row[f"Score (no finishing): {pname}"] = round(gl_nf * 100, 1) if gl_nf is not None else ""

        # Per-metric percentile scores
        for metric in all_metrics:
            wy_col = WY_MAP.get(metric)
            v_wy = None
            if wy_col and wy_row is not None and wy_col in wy_row.index and pd.notna(wy_row[wy_col]):
                v_wy = float(wy_row[wy_col])
                if metric in INVERTED:
                    v_wy = 1 - v_wy
            row[f"WY pct: {metric}"] = round(v_wy * 100, 1) if v_wy is not None else ""

            if si_df is not None:
                v_si = None
                if si_row is not None and metric in si_row.index and pd.notna(si_row[metric]):
                    v_si = float(si_row[metric])
                row[f"SI pct: {metric}"] = round(v_si * 100, 1) if v_si is not None else ""

        # Raw Wyscout stat columns
        if wy_row is not None:
            for col in wy_row.index:
                if not col.startswith("_"):
                    row[f"WY raw: {col}"] = wy_row[col]

        # Raw SICS computed values (non-percentile)
        if si_df is not None and si_row is not None:
            for col in si_row.index:
                if not col.startswith("_"):
                    row[f"SI raw: {col}"] = round(float(si_row[col]), 4) if pd.notna(si_row[col]) else ""

        rows.append(row)

    return pd.DataFrame(rows)


def score_all_players(master, wy_df, si_df, profile_name, no_finishing=False,
                      min_minutes=0, position_group=None, matched_only=False,
                      age_min=15, age_max=45):
    metrics = PROFILES[profile_name]["metrics"]
    rows = []
    for _, p in master.iterrows():
        # Filters
        if position_group and p.get("_position_group") not in (position_group, "Unknown"):
            continue
        if matched_only and pd.isna(p.get("_si")):
            continue
        by = p.get("_birth_year")
        if pd.notna(by):
            age = 2025 - int(by)
            if not (age_min <= age <= age_max):
                continue
        mins_raw = p.get("Minutes")
        if pd.notna(mins_raw):
            mins = float(mins_raw)
            if mins < min_minutes:
                continue
        else:
            mins = 0  # unknown — include player but show 0

        wy_row = None
        if wy_df is not None:
            mask = wy_df["Player"] == p["Player"]
            if mask.any():
                wy_row = wy_df[mask].iloc[0]

        si_row = _get_si_row(p["Player"], si_df, master)

        wy_s, si_s, gl, cov = score_one(wy_row, si_row, metrics, no_finishing=False)
        _, _, gl_nf, _ = score_one(wy_row, si_row, metrics, no_finishing=True)
        if gl is None:
            continue

        by = p.get("_birth_year")
        age_val = (2025 - int(by)) if pd.notna(by) else "—"
        rows.append({
            "Player":        p["Player"],
            "Team":          p["Team"],
            "League":        p.get("League", ""),
            "Age":           age_val,
            "Pos":           p.get("_position_group", p.get("Position", "")),
            "Mins":          int(mins) if mins else "—",
            "Global":        round(gl * 100, 1),
            "No Finishing":  round(gl_nf * 100, 1) if gl_nf is not None else None,
            "Wyscout":       round(wy_s * 100, 1) if wy_s is not None else None,
            "SICS":          round(si_s * 100, 1) if si_s is not None else None,
            "Data %":        f"{cov*100:.0f}%",
        })

    has_sics = si_df is not None
    if not rows:
        cols = ["Player", "Team", "League", "Age", "Pos", "Mins", "Global", "No Finishing", "Wyscout"]
        if has_sics:
            cols += ["SICS"]
        cols += ["Data %"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("Global", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ─── Radar chart ──────────────────────────────────────────────────────────────
def make_radar(player_name, profile_name, wy_df, si_df, master, no_finishing=False):
    metrics = PROFILES[profile_name]["metrics"]
    if no_finishing:
        metrics = [(m, w) for m, w in metrics if m not in FINISHING]

    wy_row = None
    if wy_df is not None:
        mask = wy_df["Player"] == player_name
        if mask.any():
            wy_row = wy_df[mask].iloc[0]
    si_row = _get_si_row(player_name, si_df, master)

    labels, wy_vals, si_vals = [], [], []
    for metric, _ in metrics:
        short = (metric.replace(" / 90", "/90").replace("% ", "%")
                       .replace("Passes ", "Pass ").replace("Actions ", "Act "))
        labels.append(short)

        wy_col = WY_MAP.get(metric)
        v_wy = 0.0
        if wy_col and wy_row is not None and wy_col in wy_row.index and pd.notna(wy_row[wy_col]):
            v_wy = float(wy_row[wy_col])
            if metric in INVERTED:
                v_wy = 1 - v_wy
        wy_vals.append(round(v_wy * 100, 1))

        v_si = 0.0
        if si_row is not None and metric in si_row.index and pd.notna(si_row[metric]):
            v_si = float(si_row[metric])
        si_vals.append(round(v_si * 100, 1))

    cats = labels + [labels[0]]
    fig = go.Figure()
    if any(v > 0 for v in wy_vals):
        fig.add_trace(go.Scatterpolar(
            r=wy_vals + [wy_vals[0]], theta=cats, fill="toself", name="Wyscout",
            line=dict(color="#1f77b4", width=2), fillcolor="rgba(31,119,180,0.15)",
        ))
    if any(v > 0 for v in si_vals):
        fig.add_trace(go.Scatterpolar(
            r=si_vals + [si_vals[0]], theta=cats, fill="toself", name="SICS",
            line=dict(color="#ff7f0e", width=2), fillcolor="rgba(255,127,14,0.15)",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
        legend=dict(orientation="h", y=-0.15),
        height=460, margin=dict(l=55, r=55, t=50, b=55),
        title=dict(text=f"<b>{player_name}</b>  ·  {profile_name}", x=0.5),
    )
    return fig


# ─── Profiles ─────────────────────────────────────────────────────────────────
PROFILES = {
    "Target Man #9": {
        "group": "Forwards",
        "metrics": [
            ("Duels aériens / 90", 20), ("% duels aériens gagnés", 15),
            ("Passes vers le dernier tiers / 90", 10),
            ("Fautes subies / 90", 10), ("xG / 90", 10), ("Tirs / 90", 5),
            ("Buts hors penaltys / 90", 5), ("Touches dans la surface / 90", 5),
            ("Passes reçues / 90", 20),
        ],
    },
    "Poacher": {
        "group": "Forwards",
        "metrics": [
            ("Buts hors penaltys / 90", 25), ("xG / 90", 20), ("Tirs / 90", 10),
            ("% conversion buts", 10), ("Touches dans la surface / 90", 10),
            ("Passes reçues / 90", 5), ("xA / 90", 5), ("Courses progressives / 90", 5),
            ("Passes clés / 90", 5), ("Actions offensives réussies / 90", 5),
        ],
    },
    "Pressing Forward": {
        "group": "Forwards",
        "metrics": [
            ("Duels défensifs / 90", 15), ("% duels défensifs gagnés", 10),
            ("Interceptions / 90", 15), ("Actions défensives réussies / 90", 10),
            ("Courses progressives / 90", 10), ("Tirs / 90", 10), ("xG / 90", 10),
            ("xA / 90", 5), ("Touches dans la surface / 90", 10),
            ("Actions offensives réussies / 90", 5),
        ],
    },
    "Creative Forward / False 9": {
        "group": "Forwards",
        "metrics": [
            ("Passes progressives / 90", 15), ("% passes progressives précises", 10),
            ("Centres profonds / 90", 10), ("Passes clés / 90", 15), ("xA / 90", 10),
            ("Courses progressives / 90", 10), ("Passes reçues / 90", 5),
            ("Tirs / 90", 10), ("xG / 90", 10), ("Touches dans la surface / 90", 5),
        ],
    },
    "Wide Forward / Inside 9": {
        "group": "Forwards",
        "metrics": [
            ("Courses progressives / 90", 15), ("Dribbles / 90", 10),
            ("% dribbles réussis", 10), ("Tirs / 90", 15), ("xG / 90", 15),
            ("xA / 90", 10), ("Touches dans la surface / 90", 10),
            ("Centres profonds / 90", 5), ("Passes clés / 90", 5),
            ("Actions offensives réussies / 90", 5),
        ],
    },
    "Full-Back": {
        "group": "Wide Players",
        "metrics": [
            ("Duels défensifs / 90", 15), ("% duels défensifs gagnés", 10),
            ("Interceptions / 90", 10), ("Centres / 90", 10), ("% centres précis", 10),
            ("Courses progressives / 90", 15), ("Passes progressives / 90", 10),
            ("Passes vers l'avant / 90", 10), ("Actions défensives réussies / 90", 5),
            ("Centres profonds / 90", 5),
        ],
    },
    "Wing-Back": {
        "group": "Wide Players",
        "metrics": [
            ("Courses progressives / 90", 20), ("Centres / 90", 15),
            ("% centres précis", 10), ("Passes décisives sur tir / 90", 15),
            ("Passes progressives / 90", 10), ("Interceptions / 90", 5),
            ("Duels défensifs / 90", 10), ("% duels défensifs gagnés", 5),
            ("Touches dans la surface / 90", 5), ("Actions offensives réussies / 90", 5),
        ],
    },
    "Inverted Full-Back": {
        "group": "Wide Players",
        "metrics": [
            ("Passes progressives / 90", 15), ("Courses progressives / 90", 10),
            ("Passes vers l'avant / 90", 10), ("% passes précises", 10),
            ("% passes courtes/moyennes précises", 5), ("Passes intelligentes / 90", 5),
            ("% duels défensifs gagnés", 10), ("Interceptions / 90", 10),
            ("Actions défensives réussies / 90", 10), ("% duels aériens gagnés", 5),
        ],
    },
    "Classic Winger": {
        "group": "Wide Players",
        "metrics": [
            ("Dribbles / 90", 20), ("% dribbles réussis", 15),
            ("Courses progressives / 90", 15), ("Centres / 90", 10),
            ("% centres précis", 10), ("Passes décisives sur tir / 90", 10),
            ("Touches dans la surface / 90", 5), ("Tirs / 90", 5),
            ("xA / 90", 5), ("Actions offensives réussies / 90", 5),
        ],
    },
    "Inverted Winger": {
        "group": "Wide Players",
        "metrics": [
            ("Tirs / 90", 20), ("xG / 90", 15), ("xA / 90", 10),
            ("Courses progressives / 90", 15), ("Passes décisives sur tir / 90", 10),
            ("Touches dans la surface / 90", 5), ("Dribbles / 90", 10),
            ("% dribbles réussis", 5), ("Centres profonds / 90", 5),
            ("Passes clés / 90", 5),
        ],
    },
    "Defensive Midfielder #6": {
        "group": "Midfielders",
        "metrics": [
            ("Interceptions / 90", 20), ("Duels défensifs / 90", 15),
            ("% duels défensifs gagnés", 10), ("Actions défensives réussies / 90", 10),
            ("% passes précises", 15), ("Passes vers l'avant / 90", 10),
            ("Passes vers le dernier tiers / 90", 10), ("Passes progressives / 90", 10),
        ],
    },
    "Deep-Lying Playmaker": {
        "group": "Midfielders",
        "metrics": [
            ("Passes progressives / 90", 25),
            ("Passes reçues / 90", 10), ("% passes longues précises", 15),
            ("Passes vers l'avant / 90", 10), ("Passes / 90", 10),
            ("Passes vers le dernier tiers / 90", 15), ("Interceptions / 90", 5),
            ("Duels défensifs / 90", 5), ("% duels aériens gagnés", 5),
        ],
    },
    "Attacking Midfielder #8": {
        "group": "Midfielders",
        "metrics": [
            ("Passes progressives / 90", 15), ("% passes progressives précises", 10),
            ("Courses progressives / 90", 15), ("xA / 90", 15), ("Tirs / 90", 15),
            ("Touches dans la surface / 90", 10), ("Interceptions / 90", 5),
            ("Passes clés / 90", 5), ("Centres profonds / 90", 5),
            ("Actions offensives réussies / 90", 5),
        ],
    },
    "Box-to-Box Midfielder": {
        "group": "Midfielders",
        "metrics": [
            ("Courses progressives / 90", 20), ("xG / 90", 15), ("Tirs / 90", 15),
            ("Interceptions / 90", 10), ("Duels défensifs / 90", 10),
            ("% duels défensifs gagnés", 5), ("Touches dans la surface / 90", 10),
            ("Actions offensives réussies / 90", 10), ("Passes vers l'avant / 90", 5),
        ],
    },
    "Classic Goalkeeper": {
        "group": "Goalkeepers",
        "metrics": [
            ("% arrêts (Save rate)", 25), ("Buts prévenus / 90", 20),
            ("Tirs encaissés / 90", 10), ("% duels aériens gagnés", 10),
            ("% passes longues précises", 10), ("Buts concédés / 90 (inversé)", 5),
            ("Sorties / 90", 5), ("Duels aériens / 90", 5),
            ("% passes précises", 10),
        ],
    },
    "Sweeper Keeper": {
        "group": "Goalkeepers",
        "metrics": [
            ("Sorties / 90", 20), ("Duels aériens / 90", 10),
            ("% duels aériens gagnés", 10), ("Buts prévenus / 90", 10),
            ("% arrêts (Save rate)", 10), ("Passes progressives / 90", 10),
            ("Passes vers l'avant / 90", 10), ("% passes longues précises", 10),
            ("Tirs encaissés / 90", 5), ("Passes vers le dernier tiers / 90", 5),
        ],
    },
    "Build-Up Keeper": {
        "group": "Goalkeepers",
        "metrics": [
            ("% passes précises", 20), ("% passes longues précises", 20),
            ("Passes progressives / 90", 20), ("Passes vers l'avant / 90", 10),
            ("Passes vers le dernier tiers / 90", 10),
            ("Passes / 90", 5), ("% arrêts (Save rate)", 5),
            ("Buts prévenus / 90", 5), ("Sorties / 90", 5),
        ],
    },
    "Ball-Playing CB": {
        "group": "Centre Backs",
        "metrics": [
            ("Passes progressives / 90", 25),
            ("Passes vers l'avant / 90", 20), ("% passes précises", 10),
            ("% passes longues précises", 10), ("Passes / 90", 10),
            ("Interceptions / 90", 10),
            ("% duels défensifs gagnés", 10), ("% duels aériens gagnés", 5),
        ],
    },
    "Combative CB / Stopper": {
        "group": "Centre Backs",
        "metrics": [
            ("Duels défensifs / 90", 20), ("% duels défensifs gagnés", 20),
            ("Duels aériens / 90", 15), ("% duels aériens gagnés", 15),
            ("Tirs contrés / 90", 10), ("Interceptions / 90", 5),
            ("Fautes / 90 (inversé)", 5), ("Actions défensives réussies / 90", 5),
        ],
    },
    "Libero / Middle Pin CB": {
        "group": "Centre Backs",
        "metrics": [
            ("Passes progressives / 90", 20), ("% passes longues précises", 15),
            ("Passes vers le dernier tiers / 90", 10), ("% passes précises", 10),
            ("Centres profonds / 90", 10), ("Passes intelligentes / 90", 10),
            ("Interceptions / 90", 10), ("xA / 90", 5),
            ("% duels aériens gagnés", 5), ("% duels défensifs gagnés", 5),
        ],
    },
    "Wide CB (in 3)": {
        "group": "Centre Backs",
        "metrics": [
            ("Duels défensifs / 90", 15), ("Courses progressives / 90", 15),
            ("% duels défensifs gagnés", 10), ("Interceptions / 90", 10),
            ("Actions défensives réussies / 90", 10), ("Centres / 90", 10),
            ("% centres précis", 10), ("Passes progressives / 90", 10),
            ("Touches dans la surface / 90", 5), ("% duels aériens gagnés", 5),
        ],
    },
}

# ─── Session state init ───────────────────────────────────────────────────────
for key, default in [("wy_bytes", None), ("si_bytes", None), ("group", None), ("min_mins", 500), ("upload_done", False), ("age_min", 15), ("age_max", 40)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Derived state ────────────────────────────────────────────────────────────
wy_df  = process_wyscout(st.session_state.wy_bytes) if st.session_state.wy_bytes else None
si_df  = process_sics(st.session_state.si_bytes)    if st.session_state.si_bytes else None
master = merge_players(wy_df, si_df) if (wy_df is not None or si_df is not None) else None

# ─── STEP 1 — Upload ──────────────────────────────────────────────────────────
if not st.session_state.get("upload_done"):
    st.title("⚽ Football Profiler")
    st.markdown("Upload your data files to get started.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1 · Wyscout CSV")
        st.caption("Export from Wyscout — filter position & cohort before exporting.")
        wy_file = st.file_uploader("", type="csv", key="wy_upload", label_visibility="collapsed")
        if wy_file:
            st.session_state.wy_bytes = wy_file.read()
        if st.session_state.wy_bytes:
            st.success("✓ Wyscout loaded")
    with col2:
        st.subheader("2 · SICS CSV  *(optional)*")
        st.caption("SICS.tv export. Adds 30% weight to the global score.")
        si_file = st.file_uploader("", type="csv", key="si_upload", label_visibility="collapsed")
        if si_file:
            st.session_state.si_bytes = si_file.read()
        if st.session_state.si_bytes:
            st.success("✓ SICS loaded")
    st.markdown("---")
    if not st.session_state.wy_bytes:
        st.info("Upload the Wyscout CSV to continue.")
    else:
        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.upload_done = True
            st.rerun()
    st.stop()

# ─── STEP 2 — Pick position group ─────────────────────────────────────────────
if st.session_state.group is None:
    st.title("⚽ Football Profiler")
    with st.sidebar:
        if st.button("← Change data"):
            st.session_state.wy_bytes = None
            st.session_state.si_bytes = None
            st.session_state.upload_done = False
            st.rerun()
        st.session_state.min_mins = st.slider("Min minutes", 100, 2000, 500, 50)

    n_wy = len(wy_df) if wy_df is not None else 0
    n_si = len(si_df) if si_df is not None else 0
    n_matched = len(master[master["_si"].notna()]) if master is not None else 0
    n_wy_only = len(master[master["_match_type"] == "wy_only"]) if master is not None and "_match_type" in master.columns else 0
    n_si_only = len(master[master["_match_type"] == "si_only"]) if master is not None and "_match_type" in master.columns else 0
    st.success(f"Data loaded — {n_wy} Wyscout · {n_si} SICS · **{n_matched} matched** · {n_wy_only} WY-only · {n_si_only} SI-only")
    with st.expander("🔍 Merge diagnostics", expanded=False):
        if master is not None and "_match_type" in master.columns:
            type_counts = master["_match_type"].value_counts()
            st.dataframe(type_counts.rename("count").reset_index().rename(columns={"index": "match_type"}),
                         hide_index=True, use_container_width=True)
            matched = master[master["_si"].notna()][["Player", "Team", "_si", "_birth_year", "_match_type"]].head(30)
            st.markdown(f"**Matched players — first 30:**")
            st.dataframe(matched, hide_index=True, use_container_width=True)
            if wy_df is not None:
                wy_cols_missing = [c for c in WY_MAP.values() if c and c not in wy_df.columns]
                if wy_cols_missing:
                    st.warning(f"WY_MAP columns missing from CSV: {wy_cols_missing}")
    st.markdown("### Select a position group")
    st.markdown(" ")
    icons = {"Forwards": "⚡", "Wide Players": "↔️", "Midfielders": "🔄",
             "Goalkeepers": "🧤", "Centre Backs": "🛡️"}
    cols = st.columns(len(POSITION_GROUPS))
    for col, grp in zip(cols, POSITION_GROUPS):
        n_profiles = sum(1 for p in PROFILES.values() if p["group"] == grp)
        with col:
            if st.button(f"{icons[grp]}\n**{grp}**\n{n_profiles} roles",
                         use_container_width=True):
                st.session_state.group = grp
                st.rerun()
    st.stop()

# ─── STEP 3 — Results ─────────────────────────────────────────────────────────
group = st.session_state.group
profiles_in_group = [n for n, p in PROFILES.items() if p["group"] == group]

with st.sidebar:
    st.markdown("### Settings")
    if st.button("← Back to positions"):
        st.session_state.group = None
        st.rerun()
    if st.button("Change data"):
        st.session_state.wy_bytes = None
        st.session_state.si_bytes = None
        st.session_state.group = None
        st.session_state.upload_done = False
        st.rerun()
    st.markdown("---")
    st.session_state.min_mins = st.slider("Min minutes", 0, 2000,
                                           st.session_state.min_mins, 50)
    age_range = st.slider("Age range", 15, 45,
                          (st.session_state.age_min, st.session_state.age_max))
    st.session_state.age_min, st.session_state.age_max = age_range
    filter_pos    = st.toggle("Filter by position group", True,
                               help="Only show players mapped to this position group")
    matched_only  = (st.toggle("Matched players only", False,
                               help="Only show players found in both Wyscout and SICS")
                     if si_df is not None else False)
    no_finishing  = st.toggle("Without finishing", False,
                               help="Excludes xG, shots, goals, conversion rate")
    st.markdown("---")
    st.markdown("**Export**")
    if st.button("Build full export CSV", use_container_width=True):
        export_df = build_export(master, wy_df, si_df)
        st.session_state["export_csv"] = export_df.to_csv(index=False).encode("utf-8")
    if "export_csv" in st.session_state:
        st.download_button(
            "⬇ Download CSV",
            data=st.session_state["export_csv"],
            file_name="profiler_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.title(f"⚽ {group}")

pos_filter = group if filter_pos else None
search = st.text_input("🔍 Search a player (leave empty to browse all roles)",
                       placeholder="Type a player name…")

# ── Player detail view ──────────────────────────────────────────────────────
if search.strip():
    sl = search.strip().lower()
    matched = [p for p in master["Player"].tolist() if sl in p.lower()] \
              if master is not None else []
    if not matched:
        st.warning("No player found.")
    else:
        player = st.selectbox("Select player", matched)
        wy_row = None
        if wy_df is not None:
            mask = wy_df["Player"] == player
            if mask.any():
                wy_row = wy_df[mask].iloc[0]
        si_row = _get_si_row(player, si_df, master)

        st.markdown(f"### {player}")
        st.markdown(f"**Scores across all {group} profiles**")

        score_cols = st.columns(len(profiles_in_group))
        for col, pname in zip(score_cols, profiles_in_group):
            _, _, gl, cov = score_one(wy_row, si_row, PROFILES[pname]["metrics"], no_finishing)
            with col:
                st.metric(pname,
                          f"{gl*100:.0f}" if gl is not None else "—",
                          delta=f"{cov*100:.0f}% data" if gl is not None else None)

        st.markdown("---")
        active_profile = st.selectbox("Radar profile", profiles_in_group)
        wy_s, si_s, gl, cov = score_one(wy_row, si_row,
                                          PROFILES[active_profile]["metrics"], no_finishing)

        if si_df is not None:
            c1, c2, c3, c4 = st.columns(4)
        else:
            c1, c3, c4 = st.columns(3)
            c2 = None
        with c1: st.metric("Wyscout (70%)" if si_df is not None else "Score", f"{wy_s*100:.1f}" if wy_s is not None else "—")
        if c2 is not None:
            with c2: st.metric("SICS (30%)", f"{si_s*100:.1f}" if si_s is not None else "—")
        with c3: st.metric("Global",        f"{gl*100:.1f}"   if gl   is not None else "—")
        with c4: st.metric("Data coverage", f"{cov*100:.0f}%")

        if not no_finishing:
            _, _, gl_nf, _ = score_one(wy_row, si_row,
                                        PROFILES[active_profile]["metrics"], no_finishing=True)
            if gl is not None and gl_nf is not None:
                delta = (gl_nf - gl) * 100
                st.caption(f"Without finishing → **{gl_nf*100:.1f}**  (delta {delta:+.1f})")

        col_radar, col_table = st.columns([3, 2])
        with col_radar:
            st.plotly_chart(
                make_radar(player, active_profile, wy_df, si_df, master, no_finishing),
                use_container_width=True,
            )
        with col_table:
            st.markdown("**Metric breakdown**")
            metrics = PROFILES[active_profile]["metrics"]
            if no_finishing:
                metrics = [(m, w) for m, w in metrics if m not in FINISHING]
            breakdown = []
            for m, w in metrics:
                wy_col = WY_MAP.get(m)
                v_wy = None
                if wy_col and wy_row is not None and wy_col in wy_row.index and pd.notna(wy_row[wy_col]):
                    v_wy = float(wy_row[wy_col])
                    if m in INVERTED:
                        v_wy = 1 - v_wy
                v_si = None
                if si_row is not None and m in si_row.index and pd.notna(si_row[m]):
                    v_si = float(si_row[m])
                proxy_note = ""
                if wy_col and wy_col not in (m, ""):
                    proxy_note = "~"
                breakdown.append({
                    "Metric":  m,
                    "W%":      w,
                    "WY":      f"{v_wy*100:.0f}{proxy_note}" if v_wy is not None else "—",
                    "SI":      f"{v_si*100:.0f}" if v_si is not None else "—",
                    "★":       "★" if m in FINISHING else "",
                })
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True,
                         column_config={
                             "W%": st.column_config.NumberColumn(width="small"),
                             "★":  st.column_config.TextColumn(width="small"),
                         })

        st.markdown("---")
        st.markdown("### 🔁 Similar players")
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            n_sim = st.slider("Number of similar players", 5, 50, 15, key="sim_slider")
        with sc2:
            sim_mode = st.radio(
                "Similarity based on",
                ["Profile metrics", "Full dataset"],
                help="Profile metrics: cosine similarity on the selected profile's metrics only.\nFull dataset: all metrics across every profile — broader stylistic fingerprint.",
                key="sim_mode",
            )
        sim_df = find_similar(player, active_profile, master, wy_df, si_df, n=n_sim,
                              mode="profile" if sim_mode == "Profile metrics" else "full",
                              min_minutes=st.session_state.min_mins)
        if sim_df.empty:
            st.info("No similar players found.")
        else:
            def sim_colour(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 90: return "background-color:#d4edda;color:#155724"
                if val >= 75: return "background-color:#fff3cd;color:#856404"
                return ""
            def score_colour(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 70: return "background-color:#d4edda;color:#155724"
                if val >= 50: return "background-color:#fff3cd;color:#856404"
                return "background-color:#f8d7da;color:#721c24"
            st.dataframe(
                sim_df.style
                    .map(sim_colour, subset=["Similarity"])
                    .map(score_colour, subset=["Score"]),
                hide_index=True, use_container_width=True,
            )

# ── Profile tabs view ────────────────────────────────────────────────────────
else:
    tabs = st.tabs([f"**{n}**" for n in profiles_in_group])
    for tab, pname in zip(tabs, profiles_in_group):
        with tab:
            st.caption(
                f"Wyscout 70% + SICS 30%"
                + (" · no finishing" if no_finishing else "")
                + f" · min {st.session_state.min_mins} min"
                + (" · position filtered" if filter_pos else "")
            )
            scored = score_all_players(
                master, wy_df, si_df, pname,
                no_finishing=no_finishing,
                min_minutes=st.session_state.min_mins,
                position_group=pos_filter,
                matched_only=matched_only,
                age_min=st.session_state.age_min,
                age_max=st.session_state.age_max,
            )
            if scored.empty:
                st.info("No players found. Try lowering the minutes threshold or disabling position filter.")
                continue

            def colour(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 70: return "background-color:#d4edda;color:#155724"
                if val >= 50: return "background-color:#fff3cd;color:#856404"
                return "background-color:#f8d7da;color:#721c24"

            _cols = ["Player", "Team", "League", "Age", "Pos", "Mins", "Global", "No Finishing", "Wyscout"]
            if si_df is not None:
                _cols += ["SICS"]
            _cols += ["Data %"]
            display = scored[[c for c in _cols if c in scored.columns]]
            st.dataframe(
                display.style.map(colour, subset=["Global", "No Finishing"]),
                use_container_width=True,
            )
