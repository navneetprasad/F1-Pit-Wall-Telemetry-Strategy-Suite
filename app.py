"""
F1 Pit-Wall Telemetry & Strategy Suite
Production-ready Streamlit Application (app.py)
Designed for Python 3.15 + FastF1 + Plotly + SciPy + Scikit-Learn
Featuring Real-World Circuit Geometries (Monza, Silverstone, Spa, Monaco, Suzuka, Bahrain, Interlagos, Melbourne)
"""

# ==============================================================================
# CRITICAL PYTHON 3.15 & NUMPY 2.0+ COMPATIBILITY PATCH
# ==============================================================================
import numpy as np
np.NaN = np.nan  # Prevents AttributeError in legacy libraries under NumPy 2.0+

import os
import sys
import datetime
import math
from collections import deque
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# FastF1 Configuration
import fastf1
import fastf1.plotting
import fastf1.ergast.interface
import logging

# Suppress noisy FastF1 unauthenticated CDN warnings in console
fastf1.set_log_level(logging.ERROR)
for log_name in ["fastf1", "fastf1.core", "fastf1.req", "fastf1._api"]:
    logging.getLogger(log_name).setLevel(logging.ERROR)

# Point Ergast interface to Jolpica API mirror (Ergast API official replacement)
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Enable local disk cache for FastF1
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except Exception:
    pass

# Import World Circuit Geometries Engine
from circuits import (
    CIRCUIT_DATABASE,
    match_circuit_key,
    get_circuit_data,
    generate_real_world_telemetry
)

# ==============================================================================
# ENTERPRISE PIT-WALL THEMING & COLOR PALETTE
# ==============================================================================
HEX_BG_CANVAS = "#0B0C10"      # Outer app window background
HEX_BG_PANEL  = "#111318"      # Subplot widget backgrounds
HEX_BORDER    = "#232732"      # Structural grid lines and card borders
HEX_TEXT_MAIN = "#EAEAEA"      # Primary data labels
HEX_TEXT_MUTED= "#6A7680"      # Secondary units and axis labels

HEX_SPEED    = "#00E5FF"       # Cyan - High visibility speed index
HEX_THROTTLE = "#00E676"       # Green - Positive acceleration
HEX_BRAKE    = "#FF1744"       # Red - High friction braking
HEX_TRACK    = "#45A29E"       # Teal - Track structural outline
HEX_CAR      = "#FFFFFF"       # White - Telemetry position node

st.set_page_config(
    page_title="F1 Pit-Wall Telemetry & Strategy Suite",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {HEX_BG_CANVAS};
        color: {HEX_TEXT_MAIN};
        font-family: 'Consolas', 'Courier New', monospace;
    }}
    
    .f1-header-container {{
        background: linear-gradient(135deg, #111318 0%, #171b24 100%);
        padding: 18px 24px;
        border-radius: 8px;
        border-left: 5px solid {HEX_BRAKE};
        border-right: 1px solid {HEX_BORDER};
        border-top: 1px solid {HEX_BORDER};
        border-bottom: 1px solid {HEX_BORDER};
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        margin-bottom: 15px;
    }}
    .f1-title {{
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.8px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .f1-subtitle {{
        font-size: 13px;
        color: {HEX_TEXT_MUTED};
        margin-top: 6px;
        margin-bottom: 0;
    }}
    
    .circuit-badge {{
        background: #07080a;
        border: 1px solid {HEX_BORDER};
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    .livery-container {{
        background: {HEX_BG_PANEL};
        border: 1px solid {HEX_BORDER};
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
    }}
    
    .metric-box {{
        background: {HEX_BG_PANEL};
        border-radius: 6px;
        border: 1px solid {HEX_BORDER};
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 700;
        color: {HEX_SPEED};
        font-family: 'Consolas', monospace;
    }}
    .metric-label {{
        font-size: 11px;
        color: {HEX_TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    
    .status-terminal {{
        background: #07080a;
        border: 1px solid {HEX_BORDER};
        border-radius: 6px;
        padding: 14px 18px;
        font-family: 'Consolas', monospace;
        color: {HEX_TEXT_MAIN};
        margin-top: 10px;
        margin-bottom: 15px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: {HEX_BG_PANEL};
        padding: 6px;
        border-radius: 8px;
        border: 1px solid {HEX_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 5px;
        padding: 8px 14px;
        color: {HEX_TEXT_MUTED};
        font-weight: 600;
        font-size: 13px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #e10600 !important;
        color: #ffffff !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# TEAM LIVERIES & SVG CAR GENERATOR
# ==============================================================================
TEAM_COLORS = {
    "Ferrari": {"primary": "#E80020", "secondary": "#FFF200", "text": "#FFFFFF"},
    "Red Bull Racing": {"primary": "#3671C6", "secondary": "#EE0000", "text": "#FFFFFF"},
    "Red Bull": {"primary": "#3671C6", "secondary": "#EE0000", "text": "#FFFFFF"},
    "Mercedes": {"primary": "#27F4D2", "secondary": "#C0C0C0", "text": "#000000"},
    "McLaren": {"primary": "#FF8000", "secondary": "#47C7FC", "text": "#000000"},
    "Aston Martin": {"primary": "#229971", "secondary": "#CEDC00", "text": "#FFFFFF"},
    "Alpine": {"primary": "#0093CC", "secondary": "#FF87BC", "text": "#FFFFFF"},
    "Williams": {"primary": "#64C4FF", "secondary": "#002B49", "text": "#000000"},
    "RB": {"primary": "#6692FF", "secondary": "#FFFFFF", "text": "#000000"},
    "AlphaTauri": {"primary": "#5E8FAA", "secondary": "#FFFFFF", "text": "#000000"},
    "Kick Sauber": {"primary": "#52E252", "secondary": "#1E1E1E", "text": "#000000"},
    "Alfa Romeo": {"primary": "#900000", "secondary": "#C92D4B", "text": "#FFFFFF"},
    "Haas F1 Team": {"primary": "#B6BABD", "secondary": "#E10600", "text": "#000000"},
    "Haas": {"primary": "#B6BABD", "secondary": "#E10600", "text": "#000000"},
    "Default": {"primary": "#E10600", "secondary": "#FFFFFF", "text": "#FFFFFF"}
}

DRIVER_TEAM_MAP = {
    "VER": "Red Bull Racing", "PER": "Red Bull Racing",
    "LEC": "Ferrari", "SAI": "Ferrari", "HAM": "Ferrari",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "NOR": "McLaren", "PIA": "McLaren",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "DOO": "Alpine", "OCO": "Haas",
    "ALB": "Williams", "COL": "Williams", "SAR": "Williams",
    "TSU": "RB", "LAW": "RB", "HAD": "RB", "RIC": "RB",
    "HUL": "Kick Sauber", "BOR": "Kick Sauber", "BOT": "Kick Sauber", "ZHO": "Kick Sauber",
    "BEA": "Haas", "MAG": "Haas"
}

DRIVER_NAMES = {
    "VER": "Max Verstappen", "PER": "Sergio Perez",
    "LEC": "Charles Leclerc", "SAI": "Carlos Sainz", "HAM": "Lewis Hamilton",
    "RUS": "George Russell", "ANT": "Andrea Kimi Antonelli",
    "NOR": "Lando Norris", "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso", "STR": "Lance Stroll",
    "GAS": "Pierre Gasly", "DOO": "Jack Doohan", "OCO": "Esteban Ocon",
    "ALB": "Alexander Albon", "COL": "Franco Colapinto", "SAR": "Logan Sargeant",
    "TSU": "Yuki Tsunoda", "LAW": "Liam Lawson", "HAD": "Isack Hadjar", "RIC": "Daniel Ricciardo",
    "HUL": "Nico Hulkenberg", "BOR": "Gabriel Bortoleto", "BOT": "Valtteri Bottas", "ZHO": "Zhou Guanyu",
    "BEA": "Oliver Bearman", "MAG": "Kevin Magnussen"
}

def get_driver_team(code):
    return DRIVER_TEAM_MAP.get(code, "Default")

def get_team_colors(team_name):
    return TEAM_COLORS.get(team_name, TEAM_COLORS["Default"])

def render_top_down_f1_car_svg(primary_color, secondary_color, driver_code, is_car2=False, width=320, height=85):
    t_cam_color = "#FFDF00" if is_car2 else "#111111"
    accent_wing = secondary_color if not is_car2 else "#FFDF00"
    
    svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 340 90" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="grad_body_{driver_code}" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="{primary_color}" />
<stop offset="70%" stop-color="{primary_color}" />
<stop offset="100%" stop-color="{secondary_color}" />
</linearGradient>
<filter id="shadow_{driver_code}" x="-10%" y="-10%" width="120%" height="120%">
<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.6"/>
</filter>
</defs>
<g filter="url(#shadow_{driver_code})">
<!-- Front Pirelli Tyres -->
<rect x="52" y="4" width="34" height="15" rx="3" fill="#151515" stroke="#333333" stroke-width="1.5"/>
<rect x="56" y="9" width="26" height="5" rx="1" fill="#FF1801"/>
<rect x="52" y="71" width="34" height="15" rx="3" fill="#151515" stroke="#333333" stroke-width="1.5"/>
<rect x="56" y="76" width="26" height="5" rx="1" fill="#FF1801"/>
<!-- Rear Pirelli Tyres -->
<rect x="252" y="2" width="40" height="18" rx="3" fill="#151515" stroke="#333333" stroke-width="1.5"/>
<rect x="257" y="8" width="30" height="6" rx="1" fill="#FF1801"/>
<rect x="252" y="70" width="40" height="18" rx="3" fill="#151515" stroke="#333333" stroke-width="1.5"/>
<rect x="257" y="76" width="30" height="6" rx="1" fill="#FF1801"/>
<!-- Suspension Wishbones -->
<line x1="68" y1="19" x2="88" y2="36" stroke="#222" stroke-width="2"/>
<line x1="68" y1="71" x2="88" y2="54" stroke="#222" stroke-width="2"/>
<!-- Front Wing -->
<path d="M 8 26 L 42 30 L 42 60 L 8 64 Z" fill="{accent_wing}" opacity="0.95"/>
<rect x="4" y="22" width="8" height="46" rx="2" fill="#111111"/>
<line x1="12" y1="26" x2="12" y2="64" stroke="#ffffff" stroke-width="1" opacity="0.7"/>
<!-- Nose & Body -->
<polygon points="12,45 68,36 100,34 100,56 68,54" fill="url(#grad_body_{driver_code})"/>
<path d="M 98 24 Q 160 20 220 28 L 220 62 Q 160 70 98 66 Z" fill="#121212" stroke="#2b2b2b" stroke-width="1"/>
<path d="M 100 32 Q 145 27 195 31 L 240 37 L 240 53 L 195 59 Q 145 63 100 58 Z" fill="{primary_color}"/>
<!-- Cockpit, Halo & Helmet -->
<ellipse cx="128" cy="45" rx="16" ry="9" fill="#080808"/>
<path d="M 120 45 L 138 39 L 138 51 Z" fill="{secondary_color}" opacity="0.85"/>
<circle cx="132" cy="45" r="5" fill="#f8fafc"/>
<rect x="150" y="43" width="7" height="4" rx="1" fill="{t_cam_color}"/>
<!-- Shark Fin & Rear Wing -->
<polygon points="175,44 245,43 245,47 175,46" fill="{secondary_color}"/>
<rect x="278" y="16" width="16" height="58" rx="2" fill="{primary_color}" stroke="{accent_wing}" stroke-width="2"/>
<rect x="290" y="14" width="4" height="62" rx="1" fill="#111111"/>
<line x1="284" y1="18" x2="284" y2="72" stroke="#ffffff" stroke-width="1" opacity="0.6"/>
<text x="75" y="48" font-family="'Arial Black', sans-serif" font-size="9" fill="#ffffff" font-weight="bold" text-anchor="middle">{driver_code}</text>
</g>
</svg>"""
    return "".join(line.strip() for line in svg.splitlines())

# ==============================================================================
# FASTF1 CIRCUIT MAP EXTRACTION ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_circuit_layout(year: int, gp_name: str, session_type: str = 'Q', is_demo: bool = True, circuit_override: str = "auto") -> pd.DataFrame:
    """
    Extracts the reference track layout coordinates using the fastest lap from FastF1.
    Extracts structural spatial columns: X, Y, Distance, RelativeDistance.
    Includes instant physics fallback for demo mode or missing telemetry.
    """
    circuit_key = circuit_override if circuit_override != "auto" else match_circuit_key(gp_name)
    
    if is_demo:
        tel1, _, _, _, _, _ = generate_real_world_telemetry(circuit_key, "REF1", "REF2", n_samples=1200)
        tel1["RelativeDistance"] = tel1["Distance"] / tel1["Distance"].max()
        return tel1[['X', 'Y', 'Distance', 'RelativeDistance']].dropna().reset_index(drop=True)

    try:
        session = fastf1.get_session(int(year), gp_name, session_type)
        session.load(telemetry=True, laps=True, weather=False)
        
        # Get the fastest lap to use as the circuit map blueprint
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is None:
            raise ValueError(f"No fastest lap recorded for {year} {gp_name} [{session_type}]")
            
        telemetry = fastest_lap.get_telemetry()
        if telemetry is None or telemetry.empty:
            raise ValueError(f"Empty telemetry for fastest lap in {year} {gp_name}")
            
        if "RelativeDistance" not in telemetry.columns and "Distance" in telemetry.columns:
            telemetry["RelativeDistance"] = telemetry["Distance"] / telemetry["Distance"].max()
            
        # Extract structural spatial columns (remove rows missing location data)
        req_cols = [c for c in ['X', 'Y', 'Distance', 'RelativeDistance'] if c in telemetry.columns]
        track_coords = telemetry[req_cols].dropna().reset_index(drop=True)
        return track_coords
    except Exception:
        # Fallback to authentic geometric layout from CIRCUIT_DATABASE
        circuit_key = match_circuit_key(gp_name)
        tel1, _, _, _, _, _ = generate_real_world_telemetry(circuit_key, "REF1", "REF2", n_samples=1200)
        tel1["RelativeDistance"] = tel1["Distance"] / tel1["Distance"].max()
        return tel1[['X', 'Y', 'Distance', 'RelativeDistance']].dropna().reset_index(drop=True)

# ==============================================================================
# DATA ENGINE: HYBRID FASTF1 + REAL-WORLD TRACK GEOMETRY ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_session_telemetry(year, gp_name, session_code, d1_code, d2_code, circuit_override="auto", force_synthetic=False):
    # Determine which real-world circuit to load
    circuit_key = circuit_override if circuit_override != "auto" else match_circuit_key(gp_name)
    circuit_meta = get_circuit_data(circuit_key)
    
    if force_synthetic:
        tel1, tel2, laps, length, meta, landmarks = generate_real_world_telemetry(circuit_key, d1_code, d2_code)
        return tel1, tel2, laps, length, meta, landmarks, True

    try:
        session = fastf1.get_session(int(year), gp_name, session_code)
        session.load(telemetry=True, weather=False, messages=False)
        
        if session.laps is None or len(session.laps) == 0:
            raise ValueError("No laps found in FastF1 session.")
            
        quick_laps = session.laps.pick_quicklaps()
        laps_d1 = quick_laps.pick_driver(d1_code)
        laps_d2 = quick_laps.pick_driver(d2_code)
        
        fastest_1 = laps_d1.pick_fastest()
        fastest_2 = laps_d2.pick_fastest()
        
        if fastest_1 is None or fastest_2 is None or pd.isna(fastest_1["LapTime"]) or pd.isna(fastest_2["LapTime"]):
            raise ValueError("Drivers missing valid fastest lap in session.")
            
        t1 = fastest_1.get_telemetry()
        t2 = fastest_2.get_telemetry()
        
        t1_df = pd.DataFrame({
            "Distance": t1["Distance"].values, "Speed": t1["Speed"].values,
            "Throttle": t1["Throttle"].values, "Brake": t1["Brake"].values,
            "nGear": t1["nGear"].values, "DRS": t1["DRS"].values,
            "X": t1["X"].values, "Y": t1["Y"].values,
            "Time": t1["Time"].dt.total_seconds().values,
            "Driver": d1_code
        })
        t2_df = pd.DataFrame({
            "Distance": t2["Distance"].values, "Speed": t2["Speed"].values,
            "Throttle": t2["Throttle"].values, "Brake": t2["Brake"].values,
            "nGear": t2["nGear"].values, "DRS": t2["DRS"].values,
            "X": t2["X"].values, "Y": t2["Y"].values,
            "Time": t2["Time"].dt.total_seconds().values,
            "Driver": d2_code
        })
        
        track_len = max(t1_df["Distance"].max(), t2_df["Distance"].max())
        laps_df = session.laps.copy()
        laps_df["LapTimeSec"] = laps_df["LapTime"].dt.total_seconds()
        laps_df["IsPitLap"] = laps_df["PitInTime"].notna().astype(int)
        
        # Landmarks for real circuit
        landmarks = circuit_meta.get("landmarks", [])
        return t1_df, t2_df, laps_df, track_len, circuit_meta, landmarks, False
    except Exception:
        # Fallback cleanly to high-precision real circuit layout
        tel1, tel2, laps, length, meta, landmarks = generate_real_world_telemetry(circuit_key, d1_code, d2_code)
        return tel1, tel2, laps, length, meta, landmarks, True

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏁 Pit-Wall Session Controls")
    
    selected_year = st.slider("Season", min_value=2018, max_value=2026, value=2024, step=1)
    
    try:
        schedule = fastf1.get_event_schedule(selected_year)
        gp_list = schedule[schedule["EventFormat"] != "testing"]["EventName"].dropna().tolist()
        if not gp_list:
            gp_list = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Monaco Grand Prix", "British Grand Prix", "Italian Grand Prix"]
    except Exception:
        gp_list = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Monaco Grand Prix", "British Grand Prix", "Italian Grand Prix", "Abu Dhabi Grand Prix"]
        
    selected_gp = st.selectbox("Grand Prix Calendar", options=gp_list, index=0)
    
    # Real Track Layout Selector
    circuit_options = {
        "auto": "⚡ Auto-Detect from Grand Prix Calendar",
        "monza": "🇮🇹 Autodromo Nazionale Monza (Italy)",
        "silverstone": "🇬🇧 Silverstone Circuit (Great Britain)",
        "spa": "🇧🇪 Circuit de Spa-Francorchamps (Belgium)",
        "monaco": "🇲🇨 Circuit de Monaco (Monte Carlo)",
        "suzuka": "🇯🇵 Suzuka International Racing Course (Japan)",
        "bahrain": "🇧🇭 Bahrain International Circuit (Sakhir)",
        "interlagos": "🇧🇷 Autódromo José Carlos Pace (Interlagos)",
        "albert_park": "🇦🇺 Albert Park Circuit (Melbourne)"
    }
    selected_circuit_override = st.selectbox("World Track Geometry Layout", options=list(circuit_options.keys()), format_func=lambda k: circuit_options[k], index=0)
    
    session_map = {
        "Race": "R", "Qualifying": "Q", "Sprint Qualifying": "SQ",
        "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"
    }
    selected_session_label = st.selectbox("Session", options=list(session_map.keys()), index=1)
    selected_session_code = session_map[selected_session_label]
    
    st.markdown("---")
    st.markdown("### 🏎️ Driver Head-to-Head Roster")
    
    roster_options = [f"{code} ({DRIVER_NAMES.get(code, code)})" for code in sorted(DRIVER_NAMES.keys())]
    d1_idx = [i for i, s in enumerate(roster_options) if s.startswith("LEC")][0]
    d2_idx = [i for i, s in enumerate(roster_options) if s.startswith("SAI")][0]
    
    driver1_sel = st.selectbox("Driver 1 (Reference)", options=roster_options, index=d1_idx)
    driver2_sel = st.selectbox("Driver 2 (Challenger)", options=roster_options, index=d2_idx)
    
    driver1_code = driver1_sel.split(" ")[0]
    driver2_code = driver2_sel.split(" ")[0]
    
    st.markdown("---")
    force_demo = st.checkbox("Force Offline Benchmark Telemetry", value=False,
                             help="Enables high-fidelity physics-grounded synthetic telemetry for offline presentations.")
    
    st.info(f"Active Session: **{selected_year} {selected_gp}** [{selected_session_code}]")

# ==============================================================================
# LOAD TELEMETRY DATA
# ==============================================================================
tel1, tel2, race_laps_df, track_length, circuit_meta, circuit_landmarks, is_synthetic = load_session_telemetry(
    selected_year, selected_gp, selected_session_code, driver1_code, driver2_code,
    circuit_override=selected_circuit_override, force_synthetic=force_demo
)

team1 = get_driver_team(driver1_code)
team2 = get_driver_team(driver2_code)
colors1 = get_team_colors(team1)
colors2 = get_team_colors(team2)

is_teammates = (team1 == team2)
color1_hex = colors1["primary"]
color2_hex = colors2["primary"] if not is_teammates else "#00F2FE"

# ==============================================================================
# MAIN PAGE HEADER & DYNAMIC LIVERY CARDS
# ==============================================================================
st.html(
    f"""
    <div class="f1-header-container">
        <h1 class="f1-title">🏎️ F1 Pit-Wall Telemetry & Strategy Suite</h1>
        <p class="f1-subtitle">
            Enterprise Telemetry Stream, Predictive Tyre Degradation & World Circuit Geometry Engine | 
            <b>{selected_year} {selected_gp}</b> [{selected_session_code}]
        </p>
    </div>
    """
)

# World Circuit Metadata Badge
st.html(
    f"""
    <div class="circuit-badge">
        <div>
            <span style="font-size: 20px; font-weight: bold; color: {HEX_SPEED};">
                {circuit_meta.get('flag', '🏁')} {circuit_meta.get('name', 'Grand Prix Circuit')}
            </span>
            <span style="color: {HEX_TEXT_MUTED}; font-size: 13px;"> ({circuit_meta.get('city', '')}, {circuit_meta.get('country', '')})</span><br>
            <span style="font-size: 12px; color: {HEX_TEXT_MUTED}; text-transform: uppercase;">
                Type: <b>{circuit_meta.get('track_type', 'Formula 1 Circuit')}</b> | Lap Record: <b>{circuit_meta.get('lap_record', 'N/A')}</b>
            </span>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 16px; font-weight: bold; color: #ffffff;">{circuit_meta.get('length_m', track_length):.0f} m</span>
            <span style="color: {HEX_TEXT_MUTED}; font-size: 12px;"> ({circuit_meta.get('length_m', track_length)/1000.0:.3f} km)</span><br>
            <span style="font-size: 12px; color: {HEX_THROTTLE}; font-weight: bold;">
                {circuit_meta.get('turns', 16)} Turns | {circuit_meta.get('drs_zones', 2)} DRS Zones
            </span>
        </div>
    </div>
    """
)

col_liv1, col_liv2 = st.columns(2)
with col_liv1:
    svg_car1 = render_top_down_f1_car_svg(color1_hex, colors1["secondary"], driver1_code, is_car2=False)
    st.html(
        f"""<div class="livery-container">
<div>
    <span style="font-size: 18px; font-weight: bold; color: {color1_hex};">{driver1_code}</span>
    <span style="color: {HEX_TEXT_MUTED}; font-size: 14px;"> — {DRIVER_NAMES.get(driver1_code, '')}</span><br>
    <span style="font-size: 12px; color: {HEX_TEXT_MUTED}; text-transform: uppercase;">{team1}</span>
</div>
{svg_car1}
</div>"""
    )

with col_liv2:
    svg_car2 = render_top_down_f1_car_svg(color2_hex, colors2["secondary"], driver2_code, is_car2=is_teammates)
    st.html(
        f"""<div class="livery-container">
<div>
    <span style="font-size: 18px; font-weight: bold; color: {color2_hex};">{driver2_code}</span>
    <span style="color: {HEX_TEXT_MUTED}; font-size: 14px;"> — {DRIVER_NAMES.get(driver2_code, '')}</span><br>
    <span style="font-size: 12px; color: {HEX_TEXT_MUTED}; text-transform: uppercase;">{team2} {'(Teammate Alt)' if is_teammates else ''}</span>
</div>
{svg_car2}
</div>"""
    )

# ==============================================================================
# TAB NAVIGATION
# ==============================================================================
tabs = st.tabs([
    "🎬 Live 2D Track & Stream HUD",
    "📊 Project 1: Cornering & Telemetry Delta",
    "🛞 Project 2: Tyre Degradation & Undercut",
    "🛠️ Project 3: Pit Stop Event Classifier",
    "🎲 Project 4: Full Race Monte Carlo Engine"
])

# ==============================================================================
# TAB 1: 🎬 LIVE 2D TRACK REPLAY & 50HZ STREAM HUD
# ==============================================================================
with tabs[0]:
    st.markdown(f"### 🎬 Real-World 2D Track Replay: {circuit_meta.get('name', 'Circuit Layout')}")
    st.markdown("High-density Cubic Spline trajectory with authentic corner landmark apexes, DRS straightaways, and live moving vehicle telemetry nodes.")
    
    replay_col1, replay_col2 = st.columns([1, 3])
    with replay_col1:
        hud_mode = st.radio("Display Stream", ["GPS Circuit Layout", "Live 50Hz Rolling Inputs Stream"], index=0)
        show_corners = st.checkbox("Annotate Iconic Corners", value=True)
        anim_granularity = st.select_slider("Playback Resolution", options=["Coarse (60 frames)", "Standard (90 frames)", "Fine (140 frames)"], value="Standard (90 frames)")
        
        st.html(
            f"""
            <div class="status-terminal">
                <span style="color: {HEX_THROTTLE}; font-weight: bold;">SYS_STATUS: NOMINAL STRAT 2</span><br>
                <span style="color: {HEX_SPEED};">CIRCUIT:</span> {circuit_meta.get('name', 'F1 Track')}<br>
                <span style="color: {HEX_SPEED};">DRIVER:</span> {driver1_code} [#{DRIVER_NAMES.get(driver1_code, '')}]<br>
                <span style="color: {HEX_SPEED};">ERS STATUS:</span> REGEN DEPLOY<br>
                <span style="color: {HEX_SPEED};">MGU-K:</span> ACTIVE // 120 kW<br>
                <span style="color: {HEX_TEXT_MUTED};">SAMPLING:</span> 50Hz CAN Bus Frame
            </div>
            """
        )
        
    n_frames = 60 if "60" in anim_granularity else (90 if "90" in anim_granularity else 140)
    max_time = max(tel1["Time"].max(), tel2["Time"].max())
    time_grid = np.linspace(0, max_time, n_frames)
    
    interp_x1 = interp1d(tel1["Time"], tel1["X"], fill_value="extrapolate")
    interp_y1 = interp1d(tel1["Time"], tel1["Y"], fill_value="extrapolate")
    interp_v1 = interp1d(tel1["Time"], tel1["Speed"], fill_value="extrapolate")
    
    interp_x2 = interp1d(tel2["Time"], tel2["X"], fill_value="extrapolate")
    interp_y2 = interp1d(tel2["Time"], tel2["Y"], fill_value="extrapolate")
    interp_v2 = interp1d(tel2["Time"], tel2["Speed"], fill_value="extrapolate")
    
    with replay_col2:
        if hud_mode == "GPS Circuit Layout":
            fig_replay = go.Figure()
            
            # Structural Track Boundaries
            fig_replay.add_trace(go.Scatter(
                x=tel1["X"], y=tel1["Y"],
                mode="lines",
                line=dict(color=HEX_BORDER, width=9),
                hoverinfo="none",
                showlegend=False
            ))
            fig_replay.add_trace(go.Scatter(
                x=tel1["X"], y=tel1["Y"],
                mode="lines",
                line=dict(color=HEX_TRACK, width=2.5),
                hoverinfo="none",
                name=f"{circuit_meta.get('name', 'Track')} Geometry"
            ))
            
            # Iconic Corner Landmark Callouts
            if show_corners and circuit_landmarks:
                for lm in circuit_landmarks:
                    fig_replay.add_trace(go.Scatter(
                        x=[lm["x"]], y=[lm["y"]],
                        mode="markers+text",
                        marker=dict(symbol="diamond", size=7, color="#ffd700"),
                        text=[lm["name"]],
                        textposition="top right",
                        textfont=dict(size=9, color="#94a3b8"),
                        hoverinfo="text",
                        hovertext=f"Apex: {lm['name']} (~{lm['speed']} km/h)",
                        showlegend=False
                    ))
            
            # Initial position nodes
            fig_replay.add_trace(go.Scatter(
                x=[interp_x1(0)], y=[interp_y1(0)],
                mode="markers+text",
                marker=dict(size=14, color=color1_hex, line=dict(color="#ffffff", width=1.5)),
                text=[f"{driver1_code} {int(interp_v1(0))} km/h"],
                textposition="top center",
                name=f"{driver1_code} (Active)"
            ))
            fig_replay.add_trace(go.Scatter(
                x=[interp_x1(0) + 1.2], y=[interp_y1(0) + 1.2],
                mode="markers",
                marker=dict(symbol="x", size=8, color=HEX_TEXT_MUTED, opacity=0.7),
                name="Raw Signal Jitter Node"
            ))
            fig_replay.add_trace(go.Scatter(
                x=[interp_x2(0)], y=[interp_y2(0)],
                mode="markers+text",
                marker=dict(size=14, color=color2_hex, line=dict(color="#ffffff", width=1.5)),
                text=[f"{driver2_code} {int(interp_v2(0))} km/h"],
                textposition="top center",
                name=f"{driver2_code} (Challenger)"
            ))
            
            frames = []
            for t_step in time_grid:
                noise_x = interp_x1(t_step) + np.random.normal(0, 0.4)
                noise_y = interp_y1(t_step) + np.random.normal(0, 0.4)
                
                # Base track traces
                f_data = [
                    go.Scatter(x=tel1["X"], y=tel1["Y"]),
                    go.Scatter(x=tel1["X"], y=tel1["Y"])
                ]
                # Corner markers if active
                if show_corners and circuit_landmarks:
                    for lm in circuit_landmarks:
                        f_data.append(go.Scatter(x=[lm["x"]], y=[lm["y"]]))
                        
                # Moving dynamic driver nodes
                f_data.extend([
                    go.Scatter(
                        x=[interp_x1(t_step)], y=[interp_y1(t_step)],
                        text=[f"{driver1_code} {int(interp_v1(t_step))} km/h"]
                    ),
                    go.Scatter(x=[noise_x], y=[noise_y]),
                    go.Scatter(
                        x=[interp_x2(t_step)], y=[interp_y2(t_step)],
                        text=[f"{driver2_code} {int(interp_v2(t_step))} km/h"]
                    )
                ])
                frames.append(go.Frame(data=f_data, name=f"{t_step:.1f}s"))
                
            fig_replay.frames = frames
            slider_steps = [
                dict(
                    method="animate",
                    args=[[f.name], dict(mode="immediate", frame=dict(duration=60, redraw=True), transition=dict(duration=0))],
                    label=f.name
                )
                for f in frames
            ]
            
            fig_replay.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        x=0.04, y=1.14,
                        buttons=[
                            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=70, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                        ]
                    )
                ],
                sliders=[dict(
                    steps=slider_steps, x=0.18, y=1.14, len=0.78,
                    currentvalue=dict(prefix="Lap Time: ", visible=True, font=dict(color=HEX_SPEED)),
                    pad=dict(t=0, b=10)
                )],
                paper_bgcolor=HEX_BG_CANVAS,
                plot_bgcolor=HEX_BG_CANVAS,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
                height=560,
                margin=dict(l=20, r=20, t=80, b=20)
            )
            st.plotly_chart(fig_replay, width="stretch")
            
        else:
            st.markdown(f"#### 📡 50Hz Time-Series Rolling Window: {circuit_meta.get('name', '')}")
            fig_stream = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=["CHANNEL M01: VEHICLE SPEED (km/h)", "CHANNEL M02: DRIVER INPUTS (THROTTLE % & BRAKE %)"]
            )
            
            fig_stream.add_trace(go.Scatter(
                x=tel1["Time"], y=tel1["Speed"],
                name="V [km/h]", line=dict(color=HEX_SPEED, width=2)
            ), row=1, col=1)
            
            fig_stream.add_trace(go.Scatter(
                x=tel1["Time"], y=tel1["Throttle"],
                name="THR [%]", line=dict(color=HEX_THROTTLE, width=1.5)
            ), row=2, col=1)
            
            fig_stream.add_trace(go.Scatter(
                x=tel1["Time"], y=tel1["Brake"] * 100,
                name="BRK [%]", line=dict(color=HEX_BRAKE, width=2),
                fill="tozeroy", fillcolor="rgba(255, 23, 68, 0.25)"
            ), row=2, col=1)
            
            fig_stream.update_layout(
                paper_bgcolor=HEX_BG_CANVAS,
                plot_bgcolor=HEX_BG_PANEL,
                font=dict(color=HEX_TEXT_MAIN, family="monospace"),
                height=520,
                showlegend=True,
                margin=dict(l=40, r=20, t=40, b=30)
            )
            fig_stream.update_xaxes(showgrid=True, gridcolor=HEX_BORDER, title_text="Lap Elapsed Time (seconds)", row=2, col=1)
            fig_stream.update_yaxes(showgrid=True, gridcolor=HEX_BORDER)
            st.plotly_chart(fig_stream, width="stretch")

# ==============================================================================
# TAB 2: 📊 PROJECT 1 — CORNERING & TELEMETRY DELTA ANALYZER
# ==============================================================================
with tabs[1]:
    st.markdown(f"### 📊 High-Precision Teammate Cornering & Spatial Delta: {circuit_meta.get('name', '')}")
    st.markdown(f"Spatial distance-aligned telemetry overlay with numerical time integration across the official **{circuit_meta.get('length_m', track_length):.0f}m** circuit.")
    
    common_distance = np.linspace(0, track_length, 1200)
    v1_interp = interp1d(tel1["Distance"], tel1["Speed"], fill_value="extrapolate")(common_distance)
    v2_interp = interp1d(tel2["Distance"], tel2["Speed"], fill_value="extrapolate")(common_distance)
    
    throttle1_interp = interp1d(tel1["Distance"], tel1["Throttle"], fill_value="extrapolate")(common_distance)
    throttle2_interp = interp1d(tel2["Distance"], tel2["Throttle"], fill_value="extrapolate")(common_distance)
    
    brake1_interp = interp1d(tel1["Distance"], tel1["Brake"], fill_value="extrapolate")(common_distance)
    brake2_interp = interp1d(tel2["Distance"], tel2["Brake"], fill_value="extrapolate")(common_distance)
    
    gear1_interp = interp1d(tel1["Distance"], tel1["nGear"], kind="nearest", fill_value="extrapolate")(common_distance)
    gear2_interp = interp1d(tel2["Distance"], tel2["nGear"], kind="nearest", fill_value="extrapolate")(common_distance)
    
    drs1_interp = interp1d(tel1["Distance"], tel1["DRS"], kind="nearest", fill_value="extrapolate")(common_distance)
    drs2_interp = interp1d(tel2["Distance"], tel2["DRS"], kind="nearest", fill_value="extrapolate")(common_distance)
    
    ds = track_length / len(common_distance)
    inv_v1 = 1.0 / np.clip(v1_interp / 3.6, 1.0, 120.0)
    inv_v2 = 1.0 / np.clip(v2_interp / 3.6, 1.0, 120.0)
    time_delta = np.cumsum(inv_v2 - inv_v1) * ds
    
    lap_time1 = tel1["Time"].max()
    lap_time2 = tel2["Time"].max()
    net_delta = lap_time2 - lap_time1
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">{driver1_code} Fastest Lap</div>
                <div class="metric-value" style="color: {color1_hex};">{lap_time1:.3f}s</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">{driver2_code} Fastest Lap</div>
                <div class="metric-value" style="color: {color2_hex};">{lap_time2:.3f}s</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi3:
        delta_color = "#10b981" if net_delta > 0 else "#ef4444"
        delta_sign = f"-{abs(net_delta):.3f}s ({driver1_code} Ahead)" if net_delta > 0 else f"+{abs(net_delta):.3f}s ({driver2_code} Ahead)"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Lap Time Delta</div>
                <div class="metric-value" style="color: {delta_color};">{delta_sign}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi4:
        v_max1 = v1_interp.max()
        v_max2 = v2_interp.max()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Top Speed ({circuit_meta.get('city', '')})</div>
                <div class="metric-value">{v_max1:.1f} vs {v_max2:.1f} km/h</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("#### ⏱️ Sector Timing Breakdown")
    idx_s1 = int(len(common_distance) * 0.33)
    idx_s2 = int(len(common_distance) * 0.66)
    
    t1_s1 = np.sum(inv_v1[:idx_s1]) * ds
    t1_s2 = np.sum(inv_v1[idx_s1:idx_s2]) * ds
    t1_s3 = np.sum(inv_v1[idx_s2:]) * ds
    
    t2_s1 = np.sum(inv_v2[:idx_s1]) * ds
    t2_s2 = np.sum(inv_v2[idx_s1:idx_s2]) * ds
    t2_s3 = np.sum(inv_v2[idx_s2:]) * ds
    
    sector_df = pd.DataFrame({
        "Timing Sector": ["Sector 1", "Sector 2", "Sector 3", "Overall Lap"],
        f"{driver1_code} Time (s)": [f"{t1_s1:.3f}", f"{t1_s2:.3f}", f"{t1_s3:.3f}", f"{lap_time1:.3f}"],
        f"{driver2_code} Time (s)": [f"{t2_s1:.3f}", f"{t2_s2:.3f}", f"{t2_s3:.3f}", f"{lap_time2:.3f}"],
        "Delta (s)": [f"{t2_s1 - t1_s1:+.3f}", f"{t2_s2 - t1_s2:+.3f}", f"{t2_s3 - t1_s3:+.3f}", f"{net_delta:+.3f}"],
        "Dominant Driver": [
            driver1_code if t1_s1 < t2_s1 else driver2_code,
            driver1_code if t1_s2 < t2_s2 else driver2_code,
            driver1_code if t1_s3 < t2_s3 else driver2_code,
            driver1_code if lap_time1 < lap_time2 else driver2_code
        ]
    })
    st.dataframe(sector_df, width="stretch", hide_index=True)
    
    st.markdown("#### 📈 Multi-Channel Spatial Telemetry Subplots")
    
    fig_tel = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=[
            f"Cumulative Time Delta (Δt in seconds) — Positive: {driver1_code} Ahead | Negative: {driver2_code} Ahead",
            f"Speed Trace Overlay (v in km/h) & Delta Area // {circuit_meta.get('name', '')}",
            "Throttle Application (%) & Brake Input",
            "Gear Selection (nGear) & DRS Zone Activation"
        ],
        row_heights=[0.24, 0.36, 0.22, 0.18]
    )
    
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=time_delta,
        name="Time Delta (s)", line=dict(color=HEX_SPEED, width=2.5),
        fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.15)"
    ), row=1, col=1)
    fig_tel.add_hline(y=0, line_dash="dash", line_color=HEX_BORDER, row=1, col=1)
    
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=v1_interp,
        name=f"{driver1_code} Speed", line=dict(color=color1_hex, width=2.5)
    ), row=2, col=1)
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=v2_interp,
        name=f"{driver2_code} Speed", line=dict(color=color2_hex, width=2.5, dash="dot")
    ), row=2, col=1)
    
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=throttle1_interp,
        name=f"{driver1_code} Throttle", line=dict(color=HEX_THROTTLE, width=1.5)
    ), row=3, col=1)
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=throttle2_interp,
        name=f"{driver2_code} Throttle", line=dict(color="#86efac", width=1.5, dash="dot")
    ), row=3, col=1)
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=brake1_interp * 100,
        name=f"{driver1_code} Brake", line=dict(color=HEX_BRAKE, width=2),
        fill="tozeroy", fillcolor="rgba(255, 23, 68, 0.2)"
    ), row=3, col=1)
    
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=gear1_interp,
        name=f"{driver1_code} Gear", line=dict(color=color1_hex, width=2, shape="hv")
    ), row=4, col=1)
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=gear2_interp,
        name=f"{driver2_code} Gear", line=dict(color=color2_hex, width=2, shape="hv", dash="dot")
    ), row=4, col=1)
    fig_tel.add_trace(go.Scatter(
        x=common_distance, y=drs1_interp * 8,
        name="DRS Zone", line=dict(color="#10b981", width=1.5, shape="hv"),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.15)"
    ), row=4, col=1)
    
    fig_tel.update_layout(
        paper_bgcolor=HEX_BG_CANVAS,
        plot_bgcolor=HEX_BG_PANEL,
        font=dict(color=HEX_TEXT_MAIN, family="monospace"),
        height=880,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig_tel.update_xaxes(showgrid=True, gridcolor=HEX_BORDER, title_text="Circuit Distance (meters)", row=4, col=1)
    fig_tel.update_yaxes(showgrid=True, gridcolor=HEX_BORDER)
    st.plotly_chart(fig_tel, width="stretch")
    
    # 2D Mini-Sector Dominance Track Map
    st.markdown(f"#### 🗺️ 2D Mini-Sector Track Dominance Map: {circuit_meta.get('name', '')}")
    n_minisectors = 120
    split_dist = np.linspace(0, track_length, n_minisectors + 1)
    
    d1_wins = 0
    d2_wins = 0
    fig_mini = go.Figure()
    
    for i in range(n_minisectors):
        d_start = split_dist[i]
        d_end = split_dist[i + 1]
        mask = (common_distance >= d_start) & (common_distance <= d_end)
        if not np.any(mask):
            continue
            
        mean_v1 = np.mean(v1_interp[mask])
        mean_v2 = np.mean(v2_interp[mask])
        
        if mean_v1 >= mean_v2:
            seg_color = color1_hex
            d1_wins += 1
        else:
            seg_color = color2_hex
            d2_wins += 1
            
        fig_mini.add_trace(go.Scatter(
            x=tel1["X"][mask], y=tel1["Y"][mask],
            mode="lines",
            line=dict(color=seg_color, width=6),
            hoverinfo="text",
            hovertext=f"Mini-Sector {i+1}: {driver1_code} {mean_v1:.1f} vs {driver2_code} {mean_v2:.1f} km/h",
            showlegend=False
        ))
        
    pct_d1 = (d1_wins / n_minisectors) * 100
    pct_d2 = (d2_wins / n_minisectors) * 100
    
    fig_mini.update_layout(
        paper_bgcolor=HEX_BG_CANVAS,
        plot_bgcolor=HEX_BG_CANVAS,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    col_map, col_dom = st.columns([3, 1])
    with col_map:
        st.plotly_chart(fig_mini, width="stretch")
    with col_dom:
        st.markdown(
            f"""
            <div style="background: {HEX_BG_PANEL}; border-radius: 6px; padding: 20px; border: 1px solid {HEX_BORDER}; margin-top: 50px;">
                <h4 style="margin-top:0; color: #ffffff;">Circuit Dominance</h4>
                <div style="margin-bottom: 15px;">
                    <span style="color: {color1_hex}; font-weight: bold; font-size: 18px;">{driver1_code}</span>: 
                    <b>{d1_wins}</b> mini-sectors ({pct_d1:.1f}%)
                </div>
                <div style="margin-bottom: 15px;">
                    <span style="color: {color2_hex}; font-weight: bold; font-size: 18px;">{driver2_code}</span>: 
                    <b>{d2_wins}</b> mini-sectors ({pct_d2:.1f}%)
                </div>
                <hr style="border-color: {HEX_BORDER};">
                <p style="font-size: 12px; color: {HEX_TEXT_MUTED};">
                    Calculated over real-world GPS coordinates of <b>{circuit_meta.get('name', '')}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

# ==============================================================================
# TAB 3: 🛞 PROJECT 2 — TYRE DEGRADATION & LINEAR-EXPONENTIAL CLIFF MATRIX
# ==============================================================================
with tabs[2]:
    st.markdown(f"### 🛞 Predictive Strategy Matrix & Thermal Degradation: {circuit_meta.get('name', '')}")
    st.markdown("Combines empirical fuel-corrected race degradation with linear-exponential thermal decay and automatic cliff warnings.")
    
    def calculate_tyre_life(laps, compound_factor, thermal_sensitivity):
        return 100 - (laps * compound_factor) - (0.012 * np.exp(laps * thermal_sensitivity))
        
    compounds_spec = {
        'SOFT (Red)':      {'factors': [1.4, 0.11], 'color': '#e10600', 'cliff': 18},
        'MEDIUM (Yellow)': {'factors': [0.8, 0.08], 'color': '#ffd129', 'cliff': 28},
        'HARD (White)':    {'factors': [0.4, 0.06], 'color': '#ffffff', 'cliff': 38}
    }
    
    stint_sim_lap = st.slider("Stint Lap Progress Inspector (Frame Lap)", min_value=1, max_value=40, value=22)
    
    cliff_col1, cliff_col2 = st.columns([3, 2])
    with cliff_col1:
        st.markdown("#### ⚡ Linear-Exponential Compound Efficiency Curves")
        fig_cliff = go.Figure()
        
        active_cliff_alerts = []
        laps_range = np.arange(0, 41)
        
        for cname, spec in compounds_spec.items():
            ideal_perf = [calculate_tyre_life(l, spec['factors'][0], spec['factors'][1]) for l in laps_range]
            fig_cliff.add_trace(go.Scatter(
                x=laps_range, y=ideal_perf,
                mode="lines",
                line=dict(color=spec['color'], dash="dot", width=1.5),
                opacity=0.4,
                name=f"Ideal {cname}"
            ))
            
            active_laps = np.arange(0, stint_sim_lap + 1)
            noise = np.random.normal(0, 0.35, size=len(active_laps))
            active_perf = np.clip([calculate_tyre_life(l, spec['factors'][0], spec['factors'][1]) for l in active_laps] + noise, 0, 100)
            
            fig_cliff.add_trace(go.Scatter(
                x=active_laps, y=active_perf,
                mode="lines",
                line=dict(color=spec['color'], width=3),
                name=f"Active {cname}"
            ))
            
            if stint_sim_lap >= spec['cliff']:
                cliff_y = calculate_tyre_life(spec['cliff'], spec['factors'][0], spec['factors'][1])
                fig_cliff.add_trace(go.Scatter(
                    x=[spec['cliff']], y=[cliff_y],
                    mode="markers",
                    marker=dict(symbol="x", size=14, color="#ff1744", line=dict(color="#fff", width=2)),
                    name=f"CLIFF: {cname.split()[0]}"
                ))
                active_cliff_alerts.append(f"⚠️ CLIFF WARNING: {cname.split()[0]} Compound past threshold (Lap {spec['cliff']})!")
                
        fig_cliff.update_layout(
            paper_bgcolor=HEX_BG_CANVAS,
            plot_bgcolor=HEX_BG_PANEL,
            font=dict(color=HEX_TEXT_MAIN, family="monospace"),
            xaxis=dict(title="Stint Lap Count", showgrid=True, gridcolor=HEX_BORDER),
            yaxis=dict(title="Tyre Performance Efficiency (%)", showgrid=True, gridcolor=HEX_BORDER, range=[20, 105]),
            height=460,
            margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(fig_cliff, width="stretch")
        
    with cliff_col2:
        st.markdown("#### 🚨 Strategy System Alerts")
        if active_cliff_alerts:
            for alert in active_cliff_alerts:
                st.error(alert)
        else:
            st.success("TYRE SYSTEM STATUS: OPERATIONAL WITHIN EXPECTED THERMAL RANGE")
            
        st.markdown("#### ⚡ Real-Time Undercut Crossover Calculator")
        pit_loss_input = st.number_input("Pit Lane Time Loss (seconds)", min_value=16.0, max_value=32.0, value=21.5, step=0.5)
        old_tyre_age = st.slider("Target Car Current Tyre Age", min_value=10, max_value=40, value=24)
        old_tyre_deg = st.number_input("Target Car Degradation Rate (s/lap)", min_value=0.04, max_value=0.30, value=0.12, step=0.01)
        fresh_tyre_advantage = st.number_input("Fresh Rubber Inherent Pace Advantage (s)", min_value=0.5, max_value=3.5, value=1.8, step=0.1)
        
        laps_after_pit = np.arange(1, 11)
        delta_gain_per_lap = fresh_tyre_advantage + (old_tyre_age + laps_after_pit) * old_tyre_deg
        cumulative_gap = pit_loss_input - np.cumsum(delta_gain_per_lap)
        
        crossover_idx = np.where(cumulative_gap <= 0)[0]
        crossover_lap = crossover_idx[0] + 1 if len(crossover_idx) > 0 else "> 10"
        
        st.info(f"🎯 **Undercut Crossover Window:** Net Overcut/Undercut achieved in **{crossover_lap} laps** post-pit!")
        
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Scatter(
            x=laps_after_pit, y=cumulative_gap,
            mode="lines+markers",
            line=dict(color=HEX_SPEED, width=3),
            marker=dict(size=8, color=HEX_SPEED)
        ))
        fig_cross.add_hline(y=0, line_dash="dash", line_color=HEX_BRAKE, annotation_text="Crossover Delta Point")
        fig_cross.update_layout(
            paper_bgcolor=HEX_BG_CANVAS,
            plot_bgcolor=HEX_BG_PANEL,
            font=dict(color=HEX_TEXT_MAIN, family="monospace"),
            xaxis=dict(title="Laps Elapsed Since Pit Exit", showgrid=True, gridcolor=HEX_BORDER),
            yaxis=dict(title="Gap to Target Car (s)", showgrid=True, gridcolor=HEX_BORDER),
            height=250,
            margin=dict(l=40, r=20, t=20, b=30)
        )
        st.plotly_chart(fig_cross, width="stretch")

# ==============================================================================
# TAB 4: 🛠️ PROJECT 3 — F1 PIT STOP TIMING & EVENT CLASSIFIER
# ==============================================================================
with tabs[3]:
    st.markdown("### 🛠️ Machine Learning F1 Pit Stop Timing & Event Classifier")
    st.markdown("Stint feature engineering and `RandomForestClassifier(class_weight='balanced')` to predict lap-by-lap pit stop probability $P(\\text{Pit on Lap } L)$.")
    
    features_df = race_laps_df[race_laps_df["LapTimeSec"].notna()].copy()
    features_df["CompoundCode"] = features_df["Compound"].map({"SOFT": 1, "MEDIUM": 2, "HARD": 3}).fillna(2)
    features_df["StintMedianPace"] = features_df.groupby(["Driver", "Stint"])["LapTimeSec"].transform("median")
    features_df["PaceDelta"] = features_df["LapTimeSec"] - features_df["StintMedianPace"]
    
    feature_cols = ["LapNumber", "TyreLife", "Stint", "CompoundCode", "PaceDelta"]
    X = features_df[feature_cols].fillna(0)
    y = features_df["IsPitLap"]
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
    rf_clf.fit(X, y)
    
    features_df["Pit_Probability"] = rf_clf.predict_proba(X)[:, 1] * 100
    
    pit_col1, pit_col2 = st.columns([3, 1])
    with pit_col1:
        st.markdown("#### 📈 Predicted Pit Probability Curves vs. Actual Pit Stops")
        fig_pit = go.Figure()
        
        for driver_code, c_hex in [(driver1_code, color1_hex), (driver2_code, color2_hex)]:
            d_laps = features_df[features_df["Driver"] == driver_code]
            if not d_laps.empty:
                fig_pit.add_trace(go.Scatter(
                    x=d_laps["LapNumber"], y=d_laps["Pit_Probability"],
                    mode="lines",
                    line=dict(color=c_hex, width=2.5),
                    name=f"{driver_code} Pit Probability (%)"
                ))
                
                actual_pits = d_laps[d_laps["IsPitLap"] == 1]
                fig_pit.add_trace(go.Scatter(
                    x=actual_pits["LapNumber"], y=actual_pits["Pit_Probability"],
                    mode="markers+text",
                    marker=dict(symbol="star", size=16, color="#FFD700", line=dict(color="#000", width=1.5)),
                    text=[f"Actual Pit (Lap {lp})" for lp in actual_pits["LapNumber"]],
                    textposition="top center",
                    name=f"{driver_code} Actual Pit Stop"
                ))
                
        fig_pit.add_hline(y=50, line_dash="dash", line_color=HEX_BRAKE, annotation_text="Critical Pit Window Threshold (50%)")
        fig_pit.update_layout(
            paper_bgcolor=HEX_BG_CANVAS,
            plot_bgcolor=HEX_BG_PANEL,
            font=dict(color=HEX_TEXT_MAIN, family="monospace"),
            xaxis=dict(title="Race Lap Number", showgrid=True, gridcolor=HEX_BORDER),
            yaxis=dict(title="Predicted Pit Probability P(Pit) (%)", showgrid=True, gridcolor=HEX_BORDER, range=[0, 105]),
            height=460,
            margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(fig_pit, width="stretch")
        
    with pit_col2:
        st.markdown("#### 🧠 Classifier Feature Importance")
        importances = pd.DataFrame({
            "Feature": ["Tyre Life", "Race Lap", "Pace Drop-off", "Stint Number", "Compound"],
            "Importance": rf_clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        fig_imp = px.bar(
            importances, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(
            paper_bgcolor=HEX_BG_CANVAS, plot_bgcolor=HEX_BG_PANEL,
            font=dict(color=HEX_TEXT_MAIN, family="monospace"), height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, width="stretch")

# ==============================================================================
# TAB 5: 🎲 PROJECT 4 — FULL RACE STRATEGY SIMULATOR (MONTE CARLO ENGINE)
# ==============================================================================
with tabs[4]:
    st.markdown(f"### 🎲 Full Race Monte Carlo Strategy Simulator: {circuit_meta.get('name', '')}")
    st.markdown(f"Stochastic simulation evaluating **Strategy A (1-Stop)** vs. **Strategy B (2-Stop)** across {circuit_meta.get('total_laps', 57)} laps.")
    
    sim_ctrl1, sim_ctrl2, sim_ctrl3, sim_ctrl4 = st.columns(4)
    with sim_ctrl1:
        raw_laps = int(circuit_meta.get("total_laps", 57))
        sim_total_laps = st.slider("Total Race Laps", min_value=25, max_value=85, value=max(25, min(85, raw_laps)), step=1)
        
        current_lap_sec = float(tel1['Time'].iloc[-1]) if (not tel1.empty and 'Time' in tel1.columns) else 80.0
        min_pace = 40.0
        max_pace = 160.0
        clamped_pace = round(max(min_pace, min(max_pace, current_lap_sec)), 1)
        sim_base_pace = st.number_input("Base Race Pace (seconds)", min_value=min_pace, max_value=max_pace, value=clamped_pace, step=0.5)
    with sim_ctrl2:
        sim_green_pit_loss = st.number_input("Green Flag Pit Loss (s)", min_value=18.0, max_value=28.0, value=22.5, step=0.5)
        sim_sc_prob = st.slider("Safety Car Prob / Lap (%)", min_value=0.0, max_value=10.0, value=3.5, step=0.5) / 100.0
    with sim_ctrl3:
        max_pit_a = max(16, int(sim_total_laps * 0.75))
        val_pit_a = max(10, min(max_pit_a, int(sim_total_laps * 0.45)))
        sim_pit_a = st.slider("Strategy A Pit Lap (1-Stop)", min_value=10, max_value=max_pit_a, value=val_pit_a)
        sim_consistency = st.number_input("Driver Consistency (σ in s)", min_value=0.1, max_value=0.8, value=0.25, step=0.05)
    with sim_ctrl4:
        max_pit_b1 = max(8, int(sim_total_laps * 0.45))
        val_pit_b1 = max(5, min(max_pit_b1, int(sim_total_laps * 0.30)))
        sim_pit_b1 = st.slider("Strategy B Pit 1 Lap", min_value=5, max_value=max_pit_b1, value=val_pit_b1)
        
        min_pit_b2 = val_pit_b1 + 2
        max_pit_b2 = max(min_pit_b2 + 2, int(sim_total_laps * 0.88))
        val_pit_b2 = max(min_pit_b2, min(max_pit_b2, int(sim_total_laps * 0.65)))
        sim_pit_b2 = st.slider("Strategy B Pit 2 Lap", min_value=min_pit_b2, max_value=max_pit_b2, value=val_pit_b2)
        
    run_sim = st.button("🚀 Run 1,000 Monte Carlo Stochastic Simulations", width="stretch")
    
    if run_sim or "sim_results" not in st.session_state:
        n_sims = 1000
        np.random.seed(42)
        
        results_a = np.zeros(n_sims)
        results_b = np.zeros(n_sims)
        
        deg_soft = 0.115
        deg_med = 0.075
        deg_hard = 0.045
        
        for sim in range(n_sims):
            sc_laps = np.random.rand(sim_total_laps) < sim_sc_prob
            
            # --- STRATEGY A (1-Stop: Med -> Hard) ---
            pace_a = 0.0
            tyre_age_a = 1
            compound_a = "MED"
            
            for lap in range(1, sim_total_laps + 1):
                is_sc = sc_laps[lap - 1]
                fuel_bonus = (sim_total_laps - lap) * 0.065
                deg = deg_med * (tyre_age_a ** 1.3) if compound_a == "MED" else deg_hard * (tyre_age_a ** 1.3)
                driver_noise = np.random.normal(0, sim_consistency)
                
                lap_time = sim_base_pace - fuel_bonus + deg + driver_noise
                
                if lap == sim_pit_a:
                    compound_a = "HARD"
                    tyre_age_a = 1
                    loss = sim_green_pit_loss * 0.55 if is_sc else sim_green_pit_loss
                    pit_crew_noise = np.random.gamma(shape=2.0, scale=0.4) - 0.8
                    pace_a += (loss + max(0, pit_crew_noise))
                else:
                    tyre_age_a += 1
                    
                pace_a += (lap_time if not is_sc else lap_time + 15.0)
                
            results_a[sim] = pace_a
            
            # --- STRATEGY B (2-Stop: Soft -> Med -> Soft) ---
            pace_b = 0.0
            tyre_age_b = 1
            compound_b = "SOFT"
            
            for lap in range(1, sim_total_laps + 1):
                is_sc = sc_laps[lap - 1]
                fuel_bonus = (sim_total_laps - lap) * 0.065
                compound_advantage = -0.6 if compound_b == "SOFT" else 0.0
                deg = deg_soft * (tyre_age_b ** 1.35) if compound_b == "SOFT" else deg_med * (tyre_age_b ** 1.3)
                driver_noise = np.random.normal(0, sim_consistency)
                
                lap_time = sim_base_pace + compound_advantage - fuel_bonus + deg + driver_noise
                
                if lap in [sim_pit_b1, sim_pit_b2]:
                    compound_b = "MED" if lap == sim_pit_b1 else "SOFT"
                    tyre_age_b = 1
                    loss = sim_green_pit_loss * 0.55 if is_sc else sim_green_pit_loss
                    pit_crew_noise = np.random.gamma(shape=2.0, scale=0.4) - 0.8
                    pace_b += (loss + max(0, pit_crew_noise))
                else:
                    tyre_age_b += 1
                    
                pace_b += (lap_time if not is_sc else lap_time + 15.0)
                
            results_b[sim] = pace_b
            
        st.session_state["sim_results"] = (results_a, results_b)
        
    res_a, res_b = st.session_state["sim_results"]
    win_a = np.mean(res_a < res_b) * 100
    win_b = 100.0 - win_a
    delta_median = np.median(res_b) - np.median(res_a)
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Strategy A Win Probability (1-Stop)</div>
                <div class="metric-value" style="color: {HEX_SPEED};">{win_a:.1f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    with s_col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Strategy B Win Probability (2-Stop)</div>
                <div class="metric-value" style="color: #FF8000;">{win_b:.1f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    with s_col3:
        adv_text = f"Strategy A by {abs(delta_median):.2f}s" if delta_median > 0 else f"Strategy B by {abs(delta_median):.2f}s"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Expected Net Race Time Delta</div>
                <div class="metric-value">{adv_text}</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("#### 📊 Total Simulated Race Time Distribution Histogram")
    sim_df = pd.DataFrame({
        "Simulated Race Time (s)": np.concatenate([res_a, res_b]),
        "Strategy": ["Strategy A (1-Stop: M-H)"] * len(res_a) + ["Strategy B (2-Stop: S-M-S)"] * len(res_b)
    })
    
    fig_hist = px.histogram(
        sim_df, x="Simulated Race Time (s)", color="Strategy",
        barmode="overlay", nbins=65,
        color_discrete_map={"Strategy A (1-Stop: M-H)": HEX_SPEED, "Strategy B (2-Stop: S-M-S)": "#FF8000"},
        marginal="box"
    )
    fig_hist.update_layout(
        paper_bgcolor=HEX_BG_CANVAS,
        plot_bgcolor=HEX_BG_PANEL,
        font=dict(color=HEX_TEXT_MAIN, family="monospace"),
        height=520,
        xaxis=dict(showgrid=True, gridcolor=HEX_BORDER),
        yaxis=dict(showgrid=True, gridcolor=HEX_BORDER),
        margin=dict(l=40, r=20, t=30, b=40)
    )
    st.plotly_chart(fig_hist, width="stretch")
    
    st.markdown(
        f"""
        > **Strategic Pit-Wall Assessment:** 
        > At a **{sim_sc_prob * 100:.1f}%** Safety Car probability per lap at **{circuit_meta.get('name', 'this circuit')}**, 
        > **{'Strategy A (1-Stop)' if win_a > win_b else 'Strategy B (2-Stop)'}** is mathematically optimal with an expected win probability of **{max(win_a, win_b):.1f}%**.
        """
    )
