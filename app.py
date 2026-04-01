"""
Football Player Profiler
Step 1 → Upload CSVs  |  Step 2 → Pick position  |  Step 3 → Browse results
Deploy: streamlit run app.py  OR  push to GitHub → share.streamlit.io
"""

import io
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
INVERTED  = {"Fautes / 90 (inversé)", "Buts concédés / 90 (inversé)"}

# ─── Wyscout column map ───────────────────────────────────────────────────────
WY_MAP = {
    "Buts hors penaltys / 90":            "Non-penalty goals per 90",
    "xG / 90":                            "npxG per 90",
    "Tirs / 90":                          "Shots per 90",
    "% conversion buts":                  "Goal conversion, %",
    "xA / 90":                            "xA per 90",
    "Passes clés / 90":                   "Shot assists per 90",
    "Passes décisives sur tir / 90":      "Shot assists per 90",
    "Courses progressives / 90":          "Progressive runs per 90",
    "Passes progressives / 90":           "Progressive passes per 90",
    "Touches dans la surface / 90":       "Touches in box per 90",
    "Passes reçues / 90":                 "Received_Passes",
    "% dribbles réussis":                 "Successful dribbles, %",
    "Dribbles / 90":                      None,
    "Centres / 90":                       "Crosses per 90",
    "% centres précis":                   "Accurate crosses, %",
    "Centres profonds / 90":              None,
    "Passes vers le dernier tiers / 90":  None,
    "Passes vers l'avant / 90":           None,
    "Passes / 90":                        "Passes per 90",
    "% passes précises":                  "Accurate passes, %",
    "% passes longues précises":          "Accurate long passes, %",
    "% passes courtes/moyennes précises": "Accurate short / medium passes, %",
    "Passes intelligentes / 90":          "Smart passes per 90",
    "Longueur moyenne de passe (m)":      None,
    "% passes progressives précises":     None,
    "Duels aériens / 90":                 "Aerial duels won per 90",
    "% duels aériens gagnés":             "Aerial duels won, %",
    "Duels défensifs / 90":               None,
    "% duels défensifs gagnés":           "Defensive duels won, %",
    "Interceptions / 90":                 "PAdj Interceptions",
    "Actions défensives réussies / 90":   "Successful defensive actions per 90",
    "Actions offensives réussies / 90":   None,
    "Tirs contrés / 90":                  "Shots blocked per 90",
    "Fautes subies / 90":                 "Fouls suffered per 90",
    "Fautes / 90 (inversé)":              "Fouls per 90",
    "Passes longues reçues / 90":         None,
    "% arrêts (Save rate)":               "Save rate, %",
    "Buts prévenus / 90":                 "Prevented goals per 90",
    "Tirs encaissés / 90":                "Shots against per 90",
    "Buts concédés / 90 (inversé)":       "Conceded goals per 90",
    "Sorties / 90":                       "Exits per 90",
}

# ─── Profiles ─────────────────────────────────────────────────────────────────
PROFILES = {
    "Target Man #9": {
        "group": "Forwards",
        "metrics": [
            ("Duels aériens / 90", 20), ("% duels aériens gagnés", 15),
            ("Passes longues reçues / 90", 10), ("Passes vers le dernier tiers / 90", 10),
            ("Fautes subies / 90", 10), ("xG / 90", 10), ("Tirs / 90", 5),
            ("Buts hors penaltys / 90", 5), ("Touches dans la surface / 90", 5),
            ("Passes reçues / 90", 10),
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
            ("% passes précises", 10), ("Passes vers l'avant / 90", 10),
            ("Passes vers le dernier tiers / 90", 10), ("Passes progressives / 90", 10),
            ("Longueur moyenne de passe (m)", 5),
        ],
    },
    "Deep-Lying Playmaker": {
        "group": "Midfielders",
        "metrics": [
            ("Passes progressives / 90", 20), ("% passes progressives précises", 15),
            ("Passes reçues / 90", 10), ("% passes longues précises", 10),
            ("Passes vers l'avant / 90", 10), ("Passes / 90", 10),
            ("Passes vers le dernier tiers / 90", 10), ("Interceptions / 90", 5),
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
            ("% passes précises", 5), ("Longueur moyenne de passe (m)", 5),
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
            ("% passes précises", 20), ("% passes longues précises", 15),
            ("Passes progressives / 90", 15), ("Passes vers l'avant / 90", 10),
            ("Passes vers le dernier tiers / 90", 10), ("Longueur moyenne de passe (m)", 10),
            ("Passes / 90", 5), ("% arrêts (Save rate)", 5),
            ("Buts prévenus / 90", 5), ("Sorties / 90", 5),
        ],
    },
    "Ball-Playing CB": {
        "group": "Centre Backs",
        "metrics": [
            ("Passes progressives / 90", 20), ("% passes progressives précises", 15),
            ("Passes vers l'avant / 90", 10), ("% passes précises", 10),
            ("% passes longues précises", 10), ("Passes / 90", 10),
            ("Interceptions / 90", 10), ("Longueur moyenne de passe (m)", 5),
            ("% duels défensifs gagnés", 5), ("% duels aériens gagnés", 5),
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

POSITION_GROUPS = ["Forwards", "Wide Players", "Midfielders", "Goalkeepers", "Centre Backs"]

# ─── Data Processing ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_wyscout(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.replace("\n", "_").strip() for c in df.columns]
    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


@st.cache_data(show_spinner=False)
def process_sics(file_bytes: bytes) -> pd.DataFrame:
    """Load SICS, compute per-90, percentile-rank. Returns wide DataFrame keyed by player name."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    minutes = df["Minuti Giocati"].fillna(0)
    p90 = minutes.replace(0, np.nan) / 90

    def r(num, den):
        return (num.fillna(0) / den.fillna(0).replace(0, np.nan)).fillna(0)

    raw = {
        "Buts hors penaltys / 90":            (df["Gol"].fillna(0) - df["Gol su Rigore"].fillna(0)) / p90,
        "xG / 90":                             df["xG"].fillna(0) / p90,
        "Tirs / 90":                           df["Tiri"].fillna(0) / p90,
        "% conversion buts":                   r(df["Gol"].fillna(0), df["Tiri"].fillna(0)),
        "xA / 90":                             df["xA"].fillna(0) / p90,
        "Passes clés / 90":                    df["Passaggi Chiave"].fillna(0) / p90,
        "Passes décisives sur tir / 90":       df["Assist"].fillna(0) / p90,
        "Courses progressives / 90":           df["xT da conduzioni"].fillna(0) / p90,
        "Passes progressives / 90":            df["xT da passaggi"].fillna(0) / p90,
        "Touches dans la surface / 90":        df["Tocchi in area avversaria"].fillna(0) / p90,
        "Passes reçues / 90":                  df["Passaggi Ricevuti"].fillna(0) / p90,
        "Dribbles / 90":                       df["Dribbling"].fillna(0) / p90,
        "% dribbles réussis":                  r(df["Dribbling Positivo"].fillna(0), df["Dribbling"].fillna(0)),
        "Centres / 90":                        df["Palle Laterali"].fillna(0) / p90,
        "% centres précis":                    r(df["Cross Riusciti"].fillna(0), df["Cross"].fillna(0)),
        "Centres profonds / 90":               df["Early Cross"].fillna(0) / p90,
        "Passes vers le dernier tiers / 90":   df["Third Pass"].fillna(0) / p90,
        "Passes vers l'avant / 90":            df["Passaggi in Area"].fillna(0) / p90,
        "Passes / 90":                         df["Passaggi"].fillna(0) / p90,
        "% passes précises":                   r(df["Passaggi Riusciti"].fillna(0), df["Passaggi"].fillna(0)),
        "% passes longues précises":           r(df["Lanci Positivi"].fillna(0), df["Lanci"].fillna(0)),
        "% passes courtes/moyennes précises":  r(df["Passaggi Riusciti"].fillna(0), df["Passaggi"].fillna(0)),
        "Duels aériens / 90":                  df["Duelli Aerei"].fillna(0) / p90,
        "% duels aériens gagnés":              r(df["Duelli Aerei Vinti"].fillna(0), df["Duelli Aerei"].fillna(0)),
        "Duels défensifs / 90":                df["Duelli Difensivi"].fillna(0) / p90,
        "% duels défensifs gagnés":            r(df["Duelli Difensivi Vinti"].fillna(0), df["Duelli Difensivi"].fillna(0)),
        "Interceptions / 90":                  df["Palle Recuperate"].fillna(0) / p90,
        "Actions défensives réussies / 90":    df["Interventi Positivi"].fillna(0) / p90,
        "Actions offensives réussies / 90":    df["Dribbling Positivo"].fillna(0) / p90,
        "Fautes subies / 90":                  df["Falli Subiti"].fillna(0) / p90,
        "Fautes / 90 (inversé)":               df["Falli Fatti"].fillna(0) / p90,
        "% arrêts (Save rate)":                r(df["Parate"].fillna(0), df["Tiri in Porta Subiti"].fillna(0)),
        "Buts prévenus / 90":                  df["GK Goals Prevented"].fillna(0) / p90,
        "Tirs encaissés / 90":                 df["Tiri in Porta Subiti"].fillna(0) / p90,
        "Buts concédés / 90 (inversé)":        df["Gol Subiti dal Portiere"].fillna(0) / p90,
        "Sorties / 90":                        df["Uscite"].fillna(0) / p90,
    }

    ranked = {}
    for metric, series in raw.items():
        pct = series.rank(pct=True, na_option="bottom")
        if metric in INVERTED:
            pct = 1 - pct
        ranked[metric] = pct.values

    result = pd.DataFrame(ranked)
    result["_player"]   = df["Giocatori"].values
    result["_team"]     = df["Squadra"].values
    result["_minutes"]  = minutes.values
    result["_position"] = df["Ruolo"].values
    result.set_index("_player", inplace=True)
    return result


def merge_players(wy_df: pd.DataFrame, si_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build master player list from Wyscout.
    SICS enriches via last-name matching.
    Returns DataFrame with columns: Player, Team, Position, Minutes, _wy_idx, _si_name
    """
    rows = []
    si_index = {}
    if si_df is not None:
        # Build last-name lookup from SICS: last-name → list of SICS player names
        for si_name in si_df.index:
            last = si_name.split()[-1].upper() if si_name.split() else si_name.upper()
            si_index.setdefault(last, []).append(si_name)

    seen_si = set()
    for _, row in wy_df.iterrows():
        name = row["Player"]
        # Try to match Wyscout player to SICS by last name
        parts = name.replace(".", "").split()
        last_wy = parts[-1].upper() if parts else name.upper()
        candidates = si_index.get(last_wy, [])
        si_name = candidates[0] if len(candidates) == 1 else None
        if si_name:
            seen_si.add(si_name)

        rows.append({
            "Player":   name,
            "Team":     row.get("Team", ""),
            "Position": row.get("Position", ""),
            "Minutes":  si_df.loc[si_name, "_minutes"] if si_name and si_df is not None else None,
            "_si":      si_name,
        })

    # Add SICS-only players (not matched to any Wyscout player)
    if si_df is not None:
        for si_name in si_df.index:
            if si_name not in seen_si:
                rows.append({
                    "Player":   si_name,
                    "Team":     si_df.loc[si_name, "_team"],
                    "Position": si_df.loc[si_name, "_position"],
                    "Minutes":  si_df.loc[si_name, "_minutes"],
                    "_si":      si_name,
                })

    return pd.DataFrame(rows)


# ─── Scoring ──────────────────────────────────────────────────────────────────
def get_player_row(player_name: str, wy_df, si_df):
    """Return (wy_row, si_row) for a player. Either can be None."""
    wy_row = None
    si_row = None
    if wy_df is not None and "Player" in wy_df.columns:
        mask = wy_df["Player"] == player_name
        if mask.any():
            wy_row = wy_df[mask].iloc[0]
    if si_df is not None:
        # try direct name, then via master list
        if player_name in si_df.index:
            si_row = si_df.loc[player_name]
    return wy_row, si_row


def score_one(wy_row, si_row, metrics, no_finishing=False):
    """Returns (wy_score, si_score, global_score) each in [0,1] or None."""
    wy_num = wy_den = si_num = si_den = 0.0
    for metric, weight in metrics:
        if no_finishing and metric in FINISHING:
            continue
        wy_col = WY_MAP.get(metric)
        if wy_col and wy_row is not None and wy_col in wy_row.index:
            val = wy_row[wy_col]
            if pd.notna(val):
                if metric in INVERTED:
                    val = 1 - float(val)
                wy_num += float(val) * weight
                wy_den += weight
        if si_row is not None and metric in si_row.index:
            val = si_row[metric]
            if pd.notna(val):
                si_num += float(val) * weight
                si_den += weight

    wy_s = wy_num / wy_den if wy_den > 0 else None
    si_s = si_num / si_den if si_den > 0 else None
    if wy_s is None and si_s is None:
        return None, None, None
    if wy_s is not None and si_s is not None:
        gl = WY_WEIGHT * wy_s + SI_WEIGHT * si_s
    else:
        gl = wy_s if wy_s is not None else si_s
    return wy_s, si_s, gl


def score_all_players(master: pd.DataFrame, wy_df, si_df, profile_name: str,
                      no_finishing=False, min_minutes=0) -> pd.DataFrame:
    metrics = PROFILES[profile_name]["metrics"]
    rows = []
    for _, p in master.iterrows():
        mins = p["Minutes"] if pd.notna(p.get("Minutes", None)) else 0
        if mins < min_minutes:
            continue
        wy_row, si_row = get_player_row(p["Player"], wy_df, si_df)
        # For SICS-only players, look up via _si key
        if si_row is None and si_df is not None and pd.notna(p.get("_si")):
            si_name = p["_si"]
            if si_name in si_df.index:
                si_row = si_df.loc[si_name]
        wy_s, si_s, gl = score_one(wy_row, si_row, metrics, no_finishing)
        if gl is None:
            continue
        rows.append({
            "Player":  p["Player"],
            "Team":    p["Team"],
            "Pos":     p["Position"],
            "Mins":    int(mins) if mins else "—",
            "Global":  round(gl * 100, 1),
            "Wyscout": round(wy_s * 100, 1) if wy_s is not None else None,
            "SICS":    round(si_s * 100, 1) if si_s is not None else None,
        })
    df = pd.DataFrame(rows).sort_values("Global", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ─── Radar Chart ──────────────────────────────────────────────────────────────
def make_radar(player_name, profile_name, wy_df, si_df, master, no_finishing=False):
    metrics = PROFILES[profile_name]["metrics"]
    if no_finishing:
        metrics = [(m, w) for m, w in metrics if m not in FINISHING]

    wy_row, si_row = get_player_row(player_name, wy_df, si_df)
    if si_row is None and si_df is not None:
        p_match = master[master["Player"] == player_name]
        if not p_match.empty and pd.notna(p_match.iloc[0].get("_si")):
            si_name = p_match.iloc[0]["_si"]
            if si_name in si_df.index:
                si_row = si_df.loc[si_name]

    labels, wy_vals, si_vals = [], [], []
    for metric, _ in metrics:
        short = metric.replace(" / 90", "/90").replace("% ", "%").replace("Passes ", "Pass ").replace("Actions ", "Act ")
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

    fig = go.Figure()
    cats = labels + [labels[0]]
    if any(v > 0 for v in wy_vals):
        fig.add_trace(go.Scatterpolar(
            r=wy_vals + [wy_vals[0]], theta=cats, fill="toself", name="Wyscout",
            line=dict(color="#1f77b4", width=2), fillcolor="rgba(31,119,180,0.15)"
        ))
    if any(v > 0 for v in si_vals):
        fig.add_trace(go.Scatterpolar(
            r=si_vals + [si_vals[0]], theta=cats, fill="toself", name="SICS",
            line=dict(color="#ff7f0e", width=2), fillcolor="rgba(255,127,14,0.15)"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
        legend=dict(orientation="h", y=-0.15),
        height=460, margin=dict(l=55, r=55, t=50, b=55),
        title=dict(text=f"<b>{player_name}</b>  ·  {profile_name}", x=0.5),
    )
    return fig


# ─── Session state init ───────────────────────────────────────────────────────
for key, default in [("wy_bytes", None), ("si_bytes", None), ("group", None), ("min_mins", 500)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Derived state ────────────────────────────────────────────────────────────
wy_df  = process_wyscout(st.session_state.wy_bytes) if st.session_state.wy_bytes else None
si_df  = process_sics(st.session_state.si_bytes)    if st.session_state.si_bytes else None
master = merge_players(wy_df, si_df) if wy_df is not None or si_df is not None else None

# ─── STEP 1 — Upload ─────────────────────────────────────────────────────────
if wy_df is None:
    st.title("⚽ Football Profiler")
    st.markdown("Upload your data files to get started.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1 · Wyscout CSV")
        st.caption("Export from Wyscout — all positions, filtered as needed.")
        wy_file = st.file_uploader("", type="csv", key="wy_upload", label_visibility="collapsed")
        if wy_file:
            st.session_state.wy_bytes = wy_file.read()
            st.rerun()
    with col2:
        st.subheader("2 · SICS CSV  *(optional)*")
        st.caption("SICS.tv export. Adds a 30 % weight to the global score.")
        si_file = st.file_uploader("", type="csv", key="si_upload", label_visibility="collapsed")
        if si_file:
            st.session_state.si_bytes = si_file.read()
            st.rerun()
    st.stop()

# ─── STEP 2 — Pick position group ─────────────────────────────────────────────
if st.session_state.group is None:
    st.title("⚽ Football Profiler")
    with st.sidebar:
        if st.button("← Change data"):
            st.session_state.wy_bytes = None
            st.session_state.si_bytes = None
            st.rerun()
        st.session_state.min_mins = st.slider("Min minutes", 100, 2000, 500, 50)

    n_wy = len(wy_df) if wy_df is not None else 0
    n_si = len(si_df) if si_df is not None else 0
    st.success(f"Data loaded — {n_wy} Wyscout players · {n_si} SICS players")
    st.markdown("### Select a position group")
    st.markdown(" ")
    cols = st.columns(len(POSITION_GROUPS))
    icons = {"Forwards": "⚡", "Wide Players": "↔️", "Midfielders": "🔄", "Goalkeepers": "🧤", "Centre Backs": "🛡️"}
    for col, grp in zip(cols, POSITION_GROUPS):
        n_profiles = sum(1 for p in PROFILES.values() if p["group"] == grp)
        with col:
            if st.button(f"{icons[grp]}\n**{grp}**\n{n_profiles} roles", use_container_width=True):
                st.session_state.group = grp
                st.rerun()
    st.stop()

# ─── STEP 3 — Results ─────────────────────────────────────────────────────────
group = st.session_state.group
profiles_in_group = [n for n, p in PROFILES.items() if p["group"] == group]

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    if st.button("← Back to positions"):
        st.session_state.group = None
        st.rerun()
    if st.button("Change data"):
        st.session_state.wy_bytes = None
        st.session_state.si_bytes = None
        st.session_state.group = None
        st.rerun()
    st.markdown("---")
    st.session_state.min_mins = st.slider("Min minutes", 100, 2000, st.session_state.min_mins, 50)
    no_finishing = st.toggle("Without finishing", False,
                             help="Excludes xG, shots, goals, conversion rate from the score")
    n_similar = st.slider("Players to show", 5, 50, 15)

st.title(f"⚽ {group}")
suffix = " *(no finishing)*" if no_finishing else ""

# Player search
search = st.text_input("🔍 Search a player (optional — browse all roles for them)",
                       placeholder="Type a player name…")

# ── Player detail view ──
if search.strip():
    search_lower = search.strip().lower()
    matched = [p for p in master["Player"].tolist() if search_lower in p.lower()] if master is not None else []
    if not matched:
        st.warning("No player found.")
    else:
        player = st.selectbox("Select player", matched)
        wy_row, si_row = get_player_row(player, wy_df, si_df)
        if si_row is None and si_df is not None and master is not None:
            pm = master[master["Player"] == player]
            if not pm.empty and pd.notna(pm.iloc[0].get("_si")):
                si_name = pm.iloc[0]["_si"]
                if si_name in si_df.index:
                    si_row = si_df.loc[si_name]

        st.markdown(f"### {player}")
        st.markdown(f"**Scores across all {group} profiles**{suffix}")

        # Score cards for all profiles
        score_cols = st.columns(len(profiles_in_group))
        for col, pname in zip(score_cols, profiles_in_group):
            wy_s, si_s, gl = score_one(wy_row, si_row, PROFILES[pname]["metrics"], no_finishing)
            with col:
                st.metric(pname, f"{gl*100:.0f}" if gl is not None else "—")

        st.markdown("---")
        # Radar for selected profile
        active_profile = st.selectbox("Radar profile", profiles_in_group)

        wy_s, si_s, gl = score_one(wy_row, si_row, PROFILES[active_profile]["metrics"], no_finishing)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Wyscout (70%)", f"{wy_s*100:.1f}" if wy_s is not None else "—")
        with c2: st.metric("SICS (30%)",    f"{si_s*100:.1f}" if si_s is not None else "—")
        with c3: st.metric("Global",        f"{gl*100:.1f}"   if gl   is not None else "—")

        # With/without finishing comparison (always visible when finishing is ON)
        if not no_finishing:
            wy_nf, si_nf, gl_nf = score_one(wy_row, si_row, PROFILES[active_profile]["metrics"], no_finishing=True)
            gl_str  = f"{gl*100:.1f}"   if gl   is not None else "—"
            gl_nf_str = f"{gl_nf*100:.1f}" if gl_nf is not None else "—"
            delta = f"{(gl_nf - gl)*100:+.1f}" if (gl_nf is not None and gl is not None) else ""
            st.caption(f"Without finishing → Global **{gl_nf_str}** (delta {delta})")

        col_radar, col_table = st.columns([3, 2])
        with col_radar:
            st.plotly_chart(make_radar(player, active_profile, wy_df, si_df, master, no_finishing),
                            use_container_width=True)
        with col_table:
            st.markdown("**Metric breakdown**")
            breakdown = []
            metrics = PROFILES[active_profile]["metrics"]
            if no_finishing:
                metrics = [(m, w) for m, w in metrics if m not in FINISHING]
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
                breakdown.append({
                    "Metric": m,
                    "W%": w,
                    "WY": f"{v_wy*100:.0f}" if v_wy is not None else "—",
                    "SI": f"{v_si*100:.0f}" if v_si is not None else "—",
                    "★": "★" if m in FINISHING else "",
                })
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True,
                         column_config={"W%": st.column_config.NumberColumn(width="small"),
                                        "★": st.column_config.TextColumn(width="small")})

# ── Profile tabs view ──
else:
    tabs = st.tabs([f"**{n}**" for n in profiles_in_group])
    for tab, pname in zip(tabs, profiles_in_group):
        with tab:
            st.caption(f"Ranked by global score (Wyscout 70% + SICS 30%){suffix}")
            scored = score_all_players(
                master, wy_df, si_df, pname,
                no_finishing=no_finishing,
                min_minutes=st.session_state.min_mins,
            )
            if scored.empty:
                st.info("No players with enough minutes for this profile.")
                continue

            # Colour-code Global column
            def colour(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 70: return "background-color:#d4edda; color:#155724"
                if val >= 50: return "background-color:#fff3cd; color:#856404"
                return "background-color:#f8d7da; color:#721c24"

            styled = (
                scored[["Player", "Team", "Pos", "Mins", "Global", "Wyscout", "SICS"]]
                .head(n_similar)
                .style.applymap(colour, subset=["Global"])
            )
            st.dataframe(styled, use_container_width=True)
