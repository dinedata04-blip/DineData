"""
================================================================================
DINEDATA — KÔFĒTALA BISTRO WASTE REDUCTION SYSTEM
File: dashboard_kofetala.py
Purpose: Streamlit dashboard with all revisions:
         1. Earth tones color scheme (coffee shop vibe)
         2. "Spoilage Risk" → "Alert Level"
         3. Waste Data — track quantities properly
         4. Admin imports, user downloads predictions
         5. Show columns, minimum 10 records displayed
         6. Random Forest runs automatically on CSV upload
         7. Code comments added throughout
         8. Performance matrix graph, top performing per day
         9. Detailed Performance — sort + filter by year/quarter/month
         10. Inventory — "Out of Stock" status added
Author: BPSU Data Science Team
Date: June 2026
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import io
import warnings
import sqlite3
import re
from datetime import datetime, date
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Kôfētala — DineData",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# EARTH TONES COLOR PALETTE — matches coffee shop vibe
# Primary:   #6F4E37 (coffee brown)
# Secondary: #A0826D (latte)
# Accent:    #D2691E (cinnamon)
# Light:     #C4A882 (cream)
# Bg:        #FAF6F1 (warm white)
# ============================================================================

EARTH = {
    'primary':   '#6F4E37',
    'secondary': '#A0826D',
    'accent':    '#D2691E',
    'light':     '#C4A882',
    'bg':        '#FAF6F1',
    'dark':      '#3E2723',
    'success':   '#6B8E4E',
    'warning':   '#B8860B',
    'danger':    '#C62828',
}

# Plotly color sequence for charts — all earth tones
CHART_COLORS = ['#6F4E37','#A0826D','#D2691E','#C4A882','#8B5E3C','#5D4037','#BCAAA4']

st.markdown(f'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&display=swap');

html, body {{ font-family: 'Fraunces', Georgia, 'Times New Roman', serif; }}
[data-testid="stMarkdownContainer"], [data-testid="stText"], [data-testid="stCaptionContainer"] {{
    font-family: 'Fraunces', Georgia, 'Times New Roman', serif;
}}
/* Protect Streamlit's icon glyphs — a blanket "*" font-family rule makes
   icon ligature text (e.g. "keyboard_double_arrow_up") render as literal
   text instead of the icon glyph. Keep icons on their own icon font. */
[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"],
[class*="material-symbols"],
.material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
}}

/* -- Background -- */
.main {{ background-color: #F7F4EF; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
section[data-testid="stSidebar"] {{ background-color: #2B1F1A; border-right: none; }}
section[data-testid="stSidebar"] * {{ color: #EFE6DA !important; }}

/* -- Hero / greeting banner -- */
.hero-banner {{
    background: linear-gradient(120deg, {EARTH["dark"]} 0%, {EARTH["primary"]} 55%, {EARTH["secondary"]} 100%);
    color: white; padding: 30px 34px; border-radius: 18px; margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(62,39,35,0.18);
    position: relative; overflow: hidden;
}}
.hero-banner::after {{
    content: ""; position: absolute; right: -40px; top: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.06);
}}
.hero-eyebrow {{ font-size: 13px; font-weight: 500; opacity: 0.85; letter-spacing: 0.3px; margin: 0 0 4px 0; }}
.hero-title {{ font-size: 26px; font-weight: 800; margin: 0 0 8px 0; }}
.hero-sub {{ font-size: 14px; font-weight: 400; opacity: 0.9; max-width: 620px; line-height: 1.5; margin: 0; }}

/* -- Legacy header (kept for pages still using it) -- */
.main-header {{
    background: linear-gradient(135deg, {EARTH["primary"]} 0%, {EARTH["secondary"]} 100%);
    color: white; padding: 24px 28px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(111,78,55,0.22);
}}

/* -- Stat / KPI cards -- */
.stat-card {{
    background: white; border-radius: 16px; padding: 16px 18px 14px 18px;
    border: 1px solid rgba(62,39,35,0.06);
    box-shadow: 0 1px 2px rgba(62,39,35,0.04), 0 4px 14px rgba(62,39,35,0.05);
    height: 100%; min-height: 100px;
    display: flex; flex-direction: column; justify-content: center;
}}
.stat-label {{
    font-size: 11.5px; font-weight: 600; color: #9C8B7A;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.stat-value {{
    font-size: 20px; font-weight: 700; color: {EARTH["dark"]}; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.stat-row {{ display: flex; align-items: center; gap: 8px; margin-top: 6px; min-height: 20px; }}
.stat-delta {{
    display: inline-flex; align-items: center; gap: 3px;
    font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 20px;
    white-space: nowrap;
}}
.stat-delta-up   {{ background: #E9EFE1; color: #556B2F; }}
.stat-delta-down {{ background: #FBEAEA; color: #C62828; }}
.stat-delta-flat {{ background: #F1ECE6; color: #8A7968; }}
.stat-delta-note {{ font-size: 11.5px; color: #B0A392; white-space: nowrap; }}

/* -- Generic content card wrapper -- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: white; border-radius: 16px !important;
    border: 1px solid rgba(62,39,35,0.06) !important;
    box-shadow: 0 1px 2px rgba(62,39,35,0.04), 0 4px 14px rgba(62,39,35,0.05);
}}

/* -- Status / label chips -- */
.chip {{ display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:20px; font-size:12.5px; font-weight:600; }}
.chip-success {{ background:#E7F3E8; color:#2E7D32; }}
.chip-warning {{ background:#FDF1DE; color:#B85C00; }}
.chip-danger  {{ background:#FBEAEA; color:#C62828; }}
.chip-neutral {{ background:#F1ECE6; color:#6F5B4B; }}

/* -- Alert level badges (kept for existing table use) -- */
.badge-high   {{ background:#C62828; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:bold; }}
.badge-medium {{ background:#B85C00; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:bold; }}
.badge-low    {{ background:#2E7D32; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:bold; }}

/* -- Buttons -- */
.stButton>button {{
    background-color: {EARTH["dark"]} !important; color: white !important;
    border-radius: 10px; padding: 10px 22px; font-weight: 600;
    border: none !important; font-size: 14px;
    transition: background-color 0.15s ease;
}}
.stButton>button:hover {{ background-color: {EARTH["primary"]} !important; }}

/* -- Sidebar brand -- */
.sidebar-brand {{ padding: 4px 4px 18px 4px; }}
.sidebar-brand-name {{ font-size: 19px; font-weight: 800; color: #FAF6F1; letter-spacing: 0.2px; }}
.sidebar-brand-sub  {{ font-size: 12.5px; color: #B8A896; font-weight: 500; margin-top: 2px; }}
.sidebar-section-label {{
    font-size: 11px; font-weight: 700; color: #8A7460; text-transform: uppercase;
    letter-spacing: 0.8px; margin: 18px 4px 8px 4px;
}}

/* -- Sidebar nav buttons -- rounded pill, active state highlighted -- */
section[data-testid="stSidebar"] .stButton {{ width: 100% !important; }}
section[data-testid="stSidebar"] .stButton>button {{
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    background-color: transparent !important;
    color: #D8C9B8 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 11px 16px !important;
    font-weight: 500 !important;
    font-size: 14.5px !important;
    text-align: left !important;
    box-shadow: none !important;
    margin: 3px 0 !important;
}}
section[data-testid="stSidebar"] .stButton>button p,
section[data-testid="stSidebar"] .stButton>button div,
section[data-testid="stSidebar"] .stButton>button span {{
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin: 0 !important;
}}
section[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: rgba(255,255,255,0.06) !important;
    color: #FAF6F1 !important;
}}
section[data-testid="stSidebar"] button[kind="primary"] {{
    background-color: {EARTH["accent"]} !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 10px rgba(216,111,25,0.35) !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:hover {{
    background-color: {EARTH["accent"]} !important;
}}

/* -- Sidebar data-status rows -- */
.status-row {{ display:flex; align-items:center; justify-content:space-between; padding:6px 4px; font-size:13px; }}
.status-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; }}
.status-dot-ok   {{ background:#7A9B5C; }}
.status-dot-bad  {{ background:#C97B63; }}

/* -- Section headers -- */
h1, h2, h3 {{ color: {EARTH["dark"]}; font-weight: 700; }}
h4 {{ color: {EARTH["dark"]}; font-weight: 600; }}

/* -- Dataframe header -- */
.stDataFrame thead {{ background-color: {EARTH["dark"]} !important; color: white !important; }}
.stDataFrame {{ border-radius: 12px !important; overflow: hidden; }}

/* -- Select boxes / inputs -- */
.stSelectbox > div > div, .stTextInput > div > div, .stMultiSelect > div > div {{
    border-radius: 10px !important;
}}

/* -- Replace the default top-right "running" spinner with a 5-step coffee-brewing animation -- */
[data-testid="stStatusWidget"] {{
    position: relative;
    min-width: 26px;
    min-height: 24px;
}}
[data-testid="stStatusWidget"] svg,
[data-testid="stStatusWidget"] > *:not(button) {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}}
[data-testid="stStatusWidget"]::before {{
    content: "";
    position: absolute;
    left: 2px;
    top: 50%;
    transform: translateY(-50%);
    width: 22px;
    height: 22px;
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyMDAgNDAiPgogIDwhLS0gRnJhbWUgMTogQmVhbnMgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwwKSI+CiAgICA8ZWxsaXBzZSBjeD0iMTQiIGN5PSIyMiIgcng9IjciIHJ5PSI5IiBmaWxsPSIjNkY0RTM3IiB0cmFuc2Zvcm09InJvdGF0ZSgtMjAgMTQgMjIpIi8+CiAgICA8bGluZSB4MT0iMTQiIHkxPSIxNSIgeDI9IjE0IiB5Mj0iMjkiIHN0cm9rZT0iI0ZBRjZGMSIgc3Ryb2tlLXdpZHRoPSIxLjIiIHRyYW5zZm9ybT0icm90YXRlKC0yMCAxNCAyMikiLz4KICAgIDxlbGxpcHNlIGN4PSIyNiIgY3k9IjE4IiByeD0iNiIgcnk9IjgiIGZpbGw9IiM4QjVFMzQiIHRyYW5zZm9ybT0icm90YXRlKDE1IDI2IDE4KSIvPgogICAgPGxpbmUgeDE9IjI2IiB5MT0iMTIiIHgyPSIyNiIgeTI9IjI0IiBzdHJva2U9IiNGQUY2RjEiIHN0cm9rZS13aWR0aD0iMS4yIiB0cmFuc2Zvcm09InJvdGF0ZSgxNSAyNiAxOCkiLz4KICAgIDxlbGxpcHNlIGN4PSIyMCIgY3k9IjMwIiByeD0iNiIgcnk9IjgiIGZpbGw9IiM2RjRFMzciIHRyYW5zZm9ybT0icm90YXRlKC01IDIwIDMwKSIvPgogICAgPGxpbmUgeDE9IjIwIiB5MT0iMjQiIHgyPSIyMCIgeTI9IjM2IiBzdHJva2U9IiNGQUY2RjEiIHN0cm9rZS13aWR0aD0iMS4yIiB0cmFuc2Zvcm09InJvdGF0ZSgtNSAyMCAzMCkiLz4KICA8L2c+CiAgPCEtLSBGcmFtZSAyOiBHcmluZGVyIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQwLDApIj4KICAgIDxwYXRoIGQ9Ik01MyAxMCBMNjcgMTAgTDY0IDIyIEw1NiAyMiBaIiBmaWxsPSIjNkY0RTM3Ii8+CiAgICA8cmVjdCB4PSI1NyIgeT0iMjIiIHdpZHRoPSI2IiBoZWlnaHQ9IjgiIGZpbGw9IiNDNEE4ODIiLz4KICAgIDxyZWN0IHg9IjU1IiB5PSIzMCIgd2lkdGg9IjEwIiBoZWlnaHQ9IjQiIHJ4PSIxIiBmaWxsPSIjM0UyNzIzIi8+CiAgICA8Y2lyY2xlIGN4PSI1OCIgY3k9IjM0IiByPSIxLjQiIGZpbGw9IiM4QjVFMzQiLz4KICAgIDxjaXJjbGUgY3g9IjYyIiBjeT0iMzYiIHI9IjEuNCIgZmlsbD0iIzhCNUUzNCIvPgogICAgPGNpcmNsZSBjeD0iNjAiIGN5PSIzOCIgcj0iMS4yIiBmaWxsPSIjOEI1RTM0Ii8+CiAgPC9nPgogIDwhLS0gRnJhbWUgMzogS2V0dGxlIHBvdXJpbmcgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODAsMCkiPgogICAgPHBhdGggZD0iTTk2IDE0IGgxMiBhMiAyIDAgMCAxIDIgMiB2OCBhNiA2IDAgMCAxIC02IDYgaC00IGE2IDYgMCAwIDEgLTYgLTYgdi04IGEyIDIgMCAwIDEgMiAtMiB6IiBmaWxsPSIjNkY0RTM3Ii8+CiAgICA8cGF0aCBkPSJNMTEwIDE2IGw2IC0yIiBzdHJva2U9IiM2RjRFMzciIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDxwYXRoIGQ9Ik0xMDIgMTAgdi0zIiBzdHJva2U9IiM2RjRFMzciIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDxwYXRoIGQ9Ik0xMTYgMTUgcTEgNCAwIDgiIHN0cm9rZT0iI0M0QTg4MiIgc3Ryb2tlLXdpZHRoPSIxLjYiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogIDwvZz4KICA8IS0tIEZyYW1lIDQ6IERyaXBwZXIgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTIwLDApIj4KICAgIDxwYXRoIGQ9Ik0xMzIgMTIgaDE2IGwtOCAxNCB6IiBmaWxsPSJub25lIiBzdHJva2U9IiM2RjRFMzciIHN0cm9rZS13aWR0aD0iMiIvPgogICAgPHBhdGggZD0iTTE0MCAyNiB2NCIgc3Ryb2tlPSIjOEI1RTM0IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogICAgPHBhdGggZD0iTTE0MCAzMiB2MyIgc3Ryb2tlPSIjQzRBODgyIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogICAgPHJlY3QgeD0iMTMzIiB5PSIzMyIgd2lkdGg9IjE0IiBoZWlnaHQ9IjUiIHJ4PSIxIiBmaWxsPSIjM0UyNzIzIi8+CiAgPC9nPgogIDwhLS0gRnJhbWUgNTogQ3VwIHdpdGggc3RlYW0gLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTYwLDApIj4KICAgIDxwYXRoIGQ9Ik02IDEwIHEyIC0zIDQgMCBNMTMgMTAgcTIgLTMgNCAwIiBzdHJva2U9IiNDNEE4ODIiIHN0cm9rZS13aWR0aD0iMS42IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2NSwwKSIvPgogICAgPHJlY3QgeD0iMTcyIiB5PSIxOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMiIgZmlsbD0iIzZGNEUzNyIvPgogICAgPHBhdGggZD0iTTE4OCAyMCBxNCAwIDQgNCB0LTQgNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNkY0RTM3IiBzdHJva2Utd2lkdGg9IjIiLz4KICAgIDxyZWN0IHg9IjE3MiIgeT0iMzAiIHdpZHRoPSIxNiIgaGVpZ2h0PSIzIiBmaWxsPSIjM0UyNzIzIi8+CiAgPC9nPgo8L3N2Zz4K");
    background-repeat: no-repeat;
    background-size: 110px 22px;   /* 5 frames * 22px each */
    animation: coffee-frames 2s steps(5) infinite;
}}

@keyframes coffee-frames {{
    from {{ background-position: 0 0; }}
    to   {{ background-position: -110px 0; }}
}}
</style>
''', unsafe_allow_html=True)

# ============================================================================
# STAT CARD COMPONENT -- replaces st.metric() with a professional card
# (label, big value, optional delta chip). Earth-tone, no emoji.
# ============================================================================
def stat_card(label, value, delta=None, delta_dir='up', note=None):
    """Render one KPI stat card. delta_dir: 'up' | 'down' | 'flat'."""
    delta_html = ""
    if delta is not None:
        cls = {'up': 'stat-delta-up', 'down': 'stat-delta-down', 'flat': 'stat-delta-flat'}[delta_dir]
        arrow = {'up': '\u25B2', 'down': '\u25BC', 'flat': '\u2013'}[delta_dir]
        delta_html = f'<span class="stat-delta {cls}">{arrow} {delta}</span>'
    note_html = f'<span class="stat-delta-note">{note}</span>' if note else ""
    st.markdown(f'''
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-row">{delta_html}{note_html}</div>
    </div>
    ''', unsafe_allow_html=True)

def hero_banner(eyebrow, title, subtitle):
    st.markdown(f'''
    <div class="hero-banner">
        <p class="hero-eyebrow">{eyebrow}</p>
        <p class="hero-title">{title}</p>
        <p class="hero-sub">{subtitle}</p>
    </div>
    ''', unsafe_allow_html=True)

def chart_insight(html_text, kind='info'):
    """Plain-language takeaway box shown under a chart, for users who don't
    read visualizations easily. kind: 'info' (brown) | 'good' (olive) | 'warn' (ochre)."""
    colors = {
        'info': ('#FFF8F3', EARTH['primary']),
        'good': ('#F0EDE2', EARTH['success']),
        'warn': ('#FDF1DE', EARTH['warning']),
    }
    bg, border = colors.get(kind, colors['info'])
    st.markdown(f'''
    <div style='background:{bg};border-left:4px solid {border};
                padding:14px 18px;border-radius:8px;margin-top:8px;font-size:14.5px;line-height:1.5'>
        {html_text}
    </div>
    ''', unsafe_allow_html=True)

def full_breakdown_html(pairs, prefix='', suffix='', decimals=2, show_pct=True):
    """Formats an ordered list of (label, value) pairs into a single
    '<b>A</b>: 10 (40%), <b>B</b>: 8 (32%), <b>C</b>: 7 (28%)' string,
    so a chart_insight box can state every category's value, not just the
    single highest one. `pairs` should already be sorted the way it's
    meant to display (e.g. largest to smallest)."""
    total = sum(v for _, v in pairs) or 1
    parts = []
    for label, val in pairs:
        pct_txt = f" ({val/total*100:.1f}%)" if show_pct else ""
        parts.append(f"<b>{label}</b>: {prefix}{val:,.{decimals}f}{suffix}{pct_txt}")
    return ", ".join(parts)


# ============================================================================
# FILE PATHS — relative to project root where dashboard.py lives
# ============================================================================

DATA_CLEANING_DIR = 'Data_Cleaning'
ML_MODELS_DIR     = 'ML_models'
FEAT_ENG_DIR      = 'Feature_Engineering'

DATA_PATHS = {
    'sales':     os.path.join(DATA_CLEANING_DIR, 'kofe_tala_sales_data.csv'),
    'menu':      os.path.join(DATA_CLEANING_DIR, 'kofe_tala_menu_data.csv'),
    'inventory': os.path.join(DATA_CLEANING_DIR, 'kofe_tala_inventory_data.csv'),
    'waste':     os.path.join(DATA_CLEANING_DIR, 'kofe_tala_waste_data.csv'),
    'features':  os.path.join(FEAT_ENG_DIR,      'kofe_tala_features_final.csv'),
}

MODEL_PATHS = {
    'demand':   os.path.join(ML_MODELS_DIR, 'model_demand_forecast.pkl'),
    'waste':    os.path.join(ML_MODELS_DIR, 'model_waste_prediction.pkl'),
    'menu':     os.path.join(ML_MODELS_DIR, 'model_menu_performance.pkl'),
    'spoilage': os.path.join(ML_MODELS_DIR, 'model_spoilage_risk.pkl'),
}

METRICS_PATH = os.path.join(ML_MODELS_DIR, 'model_metrics.json')

DATABASE_DIR = 'Database'
DB_PATH      = os.path.join(DATABASE_DIR, 'dinedata.db')


def style_map(styler, func, subset=None):
    """Version-safe Styler element-wise coloring — pandas >=2.1 renamed
    Styler.applymap() to Styler.map(); older pandas doesn't have .map()."""
    try:
        return styler.map(func, subset=subset)
    except AttributeError:
        return styler.applymap(func, subset=subset)


def add_normalized_keys(df, item_col=None, cat_col=None):
    """Adds lowercased/stripped '_item_key' and '_cat_key' columns for
    grouping, plus a mapping back to each key's most common original-case
    spelling. Inconsistent casing/whitespace in uploaded data (e.g.
    "Caramel Macchiato (iced)" vs "caramel macchiato (iced)") otherwise
    splits one item's volume across multiple rows in every chart that
    groups by item or category name, making popular items look far less
    popular than they really are. Returns (df, item_display_map, cat_display_map);
    either map is None if the corresponding column wasn't provided."""
    df = df.copy()
    item_display = None
    cat_display = None
    if item_col and item_col in df.columns:
        df['_item_key'] = df[item_col].astype(str).str.strip().str.lower()
        item_display = df.groupby('_item_key')[item_col].agg(lambda s: s.value_counts().idxmax())
    if cat_col and cat_col in df.columns:
        df['_cat_key'] = df[cat_col].astype(str).str.strip().str.lower()
        cat_display = df.groupby('_cat_key')[cat_col].agg(lambda s: s.value_counts().idxmax())
    return df, item_display, cat_display


def normalize_text_columns(df, cols):
    """Remaps each of the given text columns to a single canonical spelling
    per value (the most common original-case form for that value's
    lowercased+stripped version). Meant to be called on newly-uploaded data
    merged with existing data, BEFORE de-duplication — so casing/whitespace
    differences on upload (e.g. "caramel macchiato (iced)" vs "Caramel
    Macchiato (iced)") converge into one spelling instead of silently
    creating a second, permanent variant of the same item every time
    someone uploads with slightly different capitalization."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        key = df[col].astype(str).str.strip().str.lower()
        canonical = df.groupby(key)[col].agg(lambda s: s.value_counts().idxmax())
        df[col] = key.map(canonical)
    return df

# ── Known placeholder/template text that sometimes ends up as a real row —
# e.g. someone uploads a CSV where the template's example item name
# ("Product Name (regular)") was left un-edited. Stripped out ONCE, right
# after loading each dataframe, so it never appears anywhere on the
# dashboard (every chart, KPI, and table) instead of having to be
# special-cased on each page separately.
#
# Matching is done on a "squashed" version of the text — lowercased with
# every non-letter/non-digit character (spaces, parentheses, dashes,
# underscores, double spaces, trailing spaces, etc.) removed — so
# "Product Name (regular)", "Product Name(Regular)", "product_name -
# regular", and "PRODUCT NAME  REGULAR " all collapse to the same key and
# get caught, instead of only an exact-punctuation match.
_PLACEHOLDER_ITEM_NAMES_RAW = {
    'product name (regular)', 'product name', 'item name (regular)',
    'item name', 'item_name', 'sample item', 'example item', 'item',
    'product name (iced)', 'product name (hot)',
}

def _squash_text(s):
    """Lowercase and strip out everything except letters and digits, so
    spacing/punctuation/case differences don't defeat a placeholder match."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

_PLACEHOLDER_ITEM_KEYS = {_squash_text(name) for name in _PLACEHOLDER_ITEM_NAMES_RAW}

def strip_placeholder_rows(df, col_candidates=('item', 'item_name')):
    """Drops rows whose item/item_name value matches a known placeholder
    string, ignoring case, spacing, and punctuation differences. Safe
    no-op if df is None/empty or none of the candidate columns exist."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    for col in col_candidates:
        if col in df.columns:
            key = df[col].astype(str).map(_squash_text)
            df = df[~key.isin(_PLACEHOLDER_ITEM_KEYS)]
    return df

# ── Quarter → Month mapping, shared by every Year/Quarter/Month filter row ──
QUARTER_MONTHS = {
    'Q1': ['January', 'February', 'March'],
    'Q2': ['April', 'May', 'June'],
    'Q3': ['July', 'August', 'September'],
    'Q4': ['October', 'November', 'December'],
}
ALL_MONTHS = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']

def render_year_quarter_month_filter(df, key_prefix, date_col='date'):
    """Renders one Year / Quarter / Month filter row (plus a 'Custom Date'
    mode that swaps it for an exact date-range picker) and returns the
    filtered dataframe. The Month dropdown is CASCADED off the selected
    Quarter — picking Q1 narrows the Month options down to just January,
    February, March, instead of showing all 12 months (which let someone
    pick a month outside the quarter they just chose, silently overriding
    it). Used by every chart on the dashboard that filters by Year/Quarter/
    Month, so the cascading behavior — and the Custom Date option — is
    consistent everywhere."""
    df_f = df.copy()
    if date_col not in df_f.columns:
        return df_f
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    with c0:
        filter_mode = st.selectbox(
            "Time Filter", ['Year/Quarter/Month', 'Custom Date'], key=f'{key_prefix}_mode'
        )
    if filter_mode == 'Custom Date':
        with c1:
            if len(df_f) > 0:
                min_d = df_f[date_col].min().date()
                max_d = df_f[date_col].max().date()
                date_range = st.date_input(
                    "Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                    key=f'{key_prefix}_customrange'
                )
            else:
                date_range = None
        with c2:
            st.empty()
        with c3:
            st.empty()
        if date_range is not None and isinstance(date_range, tuple) and len(date_range) == 2:
            df_f = df_f[(df_f[date_col].dt.date >= date_range[0]) & (df_f[date_col].dt.date <= date_range[1])]
        return df_f

    yrs = sorted(df_f[date_col].dt.year.dropna().unique().tolist(), reverse=True)
    with c1:
        yr = st.selectbox("Year", ['All'] + [str(y) for y in yrs], key=f'{key_prefix}_year')
    with c2:
        qtr = st.selectbox("Quarter", ['All','Q1','Q2','Q3','Q4'], key=f'{key_prefix}_qtr')
    with c3:
        month_opts = ['All'] + (QUARTER_MONTHS[qtr] if qtr != 'All' else ALL_MONTHS)
        mon = st.selectbox("Month", month_opts, key=f'{key_prefix}_month')
    if yr != 'All':
        df_f = df_f[df_f[date_col].dt.year == int(yr)]
    if qtr != 'All':
        qm = {'Q1':[1,2,3],'Q2':[4,5,6],'Q3':[7,8,9],'Q4':[10,11,12]}
        df_f = df_f[df_f[date_col].dt.month.isin(qm[qtr])]
    if mon != 'All':
        df_f = df_f[df_f[date_col].dt.month_name() == mon]
    return df_f

def render_daily_weekly_yearly_filter(df, key_prefix, date_col='date', show_granularity=True):
    """Renders a Daily / Weekly / Yearly 'View by' selector (optional) plus
    a custom date-range picker — the same pattern used on the Database
    page — and returns (filtered_df, granularity). Replaces the old
    'Last 30 days / Last 90 days / This year / All time'-style shortcut
    filters, which only covered a few fixed windows and couldn't be
    adjusted to an exact range. Passing show_granularity=False renders
    only the date-range picker (for charts that don't need a granularity
    choice)."""
    df_f = df.copy()
    if date_col not in df_f.columns or len(df_f) == 0:
        return df_f, 'Daily'
    min_d = df_f[date_col].min().date()
    max_d = df_f[date_col].max().date()
    if show_granularity:
        g1, g2 = st.columns([1, 2])
        with g1:
            granularity = st.selectbox("View by", ['Daily', 'Weekly', 'Yearly', 'Custom Date'], index=1, key=f'{key_prefix}_gran')
        with g2:
            date_range = st.date_input(
                "Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                key=f'{key_prefix}_daterange'
            )
    else:
        granularity = 'Daily'
        date_range = st.date_input(
            "Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
            key=f'{key_prefix}_daterange'
        )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        df_f = df_f[(df_f[date_col].dt.date >= start_d) & (df_f[date_col].dt.date <= end_d)]
    # 'Custom Date' buckets the same as 'Daily' — the date-range picker
    # above already lets the user narrow to any exact range regardless of
    # which option is chosen, so the two behave identically day-to-day.
    if granularity == 'Custom Date':
        granularity = 'Daily'
    return df_f, granularity

def render_custom_range_picker(df, date_col, key_prefix, bucket='day'):
    """Renders the date-range picker shown when 'Custom Date' is chosen in
    a 'View by' dropdown. Filters df to the picked range and adds a
    'period' column bucketed by day/week/month. Returns (filtered_df,
    x_axis_label)."""
    df = df.copy()
    if date_col not in df.columns or len(df) == 0:
        st.date_input("Custom Date Range", value=(date.today(), date.today()), key=f'{key_prefix}_customrange')
        df['period'] = df[date_col] if date_col in df.columns else []
        return df, 'Date'
    min_d = df[date_col].min().date()
    max_d = df[date_col].max().date()
    sel_range = st.date_input(
        "Custom Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
        key=f'{key_prefix}_customrange'
    )
    if isinstance(sel_range, tuple) and len(sel_range) == 2:
        start_d, end_d = sel_range
        df = df[(df[date_col].dt.date >= start_d) & (df[date_col].dt.date <= end_d)]
    if bucket == 'week':
        df['period'] = df[date_col].dt.to_period('W').dt.start_time
        x_label = 'Week'
    elif bucket == 'month':
        df['period'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        x_label = 'Month'
    else:
        df['period'] = df[date_col].dt.date
        x_label = 'Date'
    return df, x_label

# ============================================================================
# DATA LOADERS — cached so they only load once per session
# ============================================================================

@st.cache_data
def load_csv(path):
    """Load a CSV file and auto-parse date columns."""
    try:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception:
        return None

@st.cache_resource
def load_model(path):
    """Load a trained joblib model bundle."""
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data
def load_metrics():
    """Load saved model performance metrics from JSON."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

# ============================================================================
# DATABASE LAYER — reads from dinedata.db (Galaxy Schema) when available,
# reconstructing flat DataFrames identical in shape to the old CSV-based
# loaders so every chart/page below works unchanged either way.
# ============================================================================

def db_available():
    return os.path.exists(DB_PATH)

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data
def load_sales_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT s.sale_key, d.date AS date, i.item_name AS item, i.item_name,
                   i.category, s.quantity, s.price, s.total, s.cost
            FROM FACT_SALES s
            JOIN DIM_DATE d ON s.date_key = d.date_key
            JOIN DIM_ITEM i ON s.item_key = i.item_key
        """, conn)
        conn.close()
        # de-dup the repeated item_name column from SELECT *-style aliasing above
        df = df.loc[:, ~df.columns.duplicated()]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception:
        return None

@st.cache_data
def load_menu_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT item_name, category, price, cost, profit_margin FROM DIM_ITEM", conn)
        conn.close()
        return df
    except Exception:
        return None

@st.cache_data
def load_waste_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT w.waste_key, d.date AS date, i.item_name, i.category,
                   w.quantity_wasted, r.reason_name AS waste_reason,
                   w.cost_per_item, w.total_waste_cost,
                   d.month, d.month_name
            FROM FACT_WASTE w
            JOIN DIM_DATE d ON w.date_key = d.date_key
            JOIN DIM_ITEM i ON w.item_key = i.item_key
            JOIN DIM_WASTE_REASON r ON w.reason_key = r.reason_key
        """, conn)
        conn.close()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception:
        return None

@st.cache_data
def load_inventory_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT inv.inventory_key, d.date AS purchase_date, ing.ingredient_name AS ingredient,
                   inv.quantity, ing.unit, inv.cost_per_unit, inv.total_cost,
                   inv.expiration_date, inv.shelf_life_days, inv.days_until_expiration,
                   al.alert_name AS alert_level
            FROM FACT_INVENTORY inv
            JOIN DIM_DATE d ON inv.date_key = d.date_key
            JOIN DIM_INGREDIENT ing ON inv.ingredient_key = ing.ingredient_key
            JOIN DIM_ALERT_LEVEL al ON inv.alert_key = al.alert_key
        """, conn)
        conn.close()
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
        return df
    except Exception:
        return None

def get_or_create_item_key(conn, item_name, category=None):
    row = conn.execute("SELECT item_key FROM DIM_ITEM WHERE item_name = ?", (item_name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO DIM_ITEM (item_name, category) VALUES (?, ?)", (item_name, category))
    return cur.lastrowid

def get_or_create_reason_key(conn, reason_name):
    row = conn.execute("SELECT reason_key FROM DIM_WASTE_REASON WHERE reason_name = ?", (reason_name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO DIM_WASTE_REASON (reason_name) VALUES (?)", (reason_name,))
    return cur.lastrowid

def get_or_create_ingredient_key(conn, ingredient_name, unit=None, shelf_life_days=None):
    row = conn.execute("SELECT ingredient_key FROM DIM_INGREDIENT WHERE ingredient_name = ?", (ingredient_name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO DIM_INGREDIENT (ingredient_name, unit, shelf_life_days) VALUES (?, ?, ?)",
        (ingredient_name, unit, shelf_life_days)
    )
    return cur.lastrowid

def get_or_create_alert_key(conn, alert_name):
    row = conn.execute("SELECT alert_key FROM DIM_ALERT_LEVEL WHERE alert_name = ?", (alert_name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO DIM_ALERT_LEVEL (alert_name) VALUES (?)", (alert_name,))
    return cur.lastrowid

def get_or_create_date_key(conn, date_val):
    d = pd.Timestamp(date_val)
    dk = int(d.strftime('%Y%m%d'))
    row = conn.execute("SELECT date_key FROM DIM_DATE WHERE date_key = ?", (dk,)).fetchone()
    if row:
        return dk
    conn.execute(
        "INSERT INTO DIM_DATE VALUES (?,?,?,?,?,?,?,?,?,?)",
        (dk, d.strftime('%Y-%m-%d'), int(d.dayofweek), d.day_name(),
         int(d.isocalendar()[1]), int(d.month), d.month_name(),
         int((d.month - 1)//3 + 1), int(d.year), int(d.dayofweek >= 5))
    )
    return dk

def insert_sales_records_to_db(new_sales_df):
    """Insert new sales rows into FACT_SALES (+ any new dimension rows needed).
    Skips a row if an identical one (same date + item + qty + total) already
    exists, to prevent accidental double-inserts."""
    if not db_available():
        return 0
    conn = get_db_connection()
    inserted = 0
    for _, row in new_sales_df.iterrows():
        try:
            dk = get_or_create_date_key(conn, row['date'])
            ik = get_or_create_item_key(conn, row['item'], row.get('category'))
            dupe = conn.execute(
                "SELECT 1 FROM FACT_SALES WHERE date_key=? AND item_key=? AND quantity=? AND total=? LIMIT 1",
                (dk, ik, row.get('quantity'), row.get('total'))
            ).fetchone()
            if dupe:
                continue
            conn.execute(
                "INSERT INTO FACT_SALES (date_key, item_key, quantity, price, total, cost) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (dk, ik, row.get('quantity'), row.get('price'), row.get('total'), row.get('cost'))
            )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    load_sales_from_db.clear()
    return inserted

def insert_menu_records_to_db(new_menu_df):
    """Insert or update DIM_ITEM rows for new/changed menu items."""
    if not db_available():
        return 0
    conn = get_db_connection()
    inserted = 0
    for _, row in new_menu_df.iterrows():
        try:
            price = row.get('price')
            cost = row.get('cost')
            margin = (price - cost) if (pd.notna(price) and pd.notna(cost)) else None
            existing = conn.execute(
                "SELECT item_key FROM DIM_ITEM WHERE item_name = ?", (row['item_name'],)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE DIM_ITEM SET category=?, price=?, cost=?, profit_margin=? WHERE item_key=?",
                    (row.get('category'), price, cost, margin, existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO DIM_ITEM (item_name, category, price, cost, profit_margin) VALUES (?,?,?,?,?)",
                    (row['item_name'], row.get('category'), price, cost, margin)
                )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    load_menu_from_db.clear()
    return inserted

def insert_waste_records_to_db(new_waste_df):
    """Insert new waste rows into FACT_WASTE (+ any new dimension rows needed).
    Skips a row if an identical one (same date + item + reason + qty) already
    exists, so clicking 'Add to Existing Data' twice by accident can't
    double the database while the CSV stays correctly deduped."""
    if not db_available():
        return 0
    conn = get_db_connection()
    inserted = 0
    for _, row in new_waste_df.iterrows():
        try:
            dk = get_or_create_date_key(conn, row['date'])
            ik = get_or_create_item_key(conn, row['item_name'], row.get('category'))
            rk = get_or_create_reason_key(conn, row['waste_reason'])
            dupe = conn.execute(
                "SELECT 1 FROM FACT_WASTE WHERE date_key=? AND item_key=? AND reason_key=? AND quantity_wasted=? LIMIT 1",
                (dk, ik, rk, row.get('quantity_wasted'))
            ).fetchone()
            if dupe:
                continue
            conn.execute(
                "INSERT INTO FACT_WASTE (date_key, item_key, reason_key, quantity_wasted, cost_per_item, total_waste_cost) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (dk, ik, rk, row.get('quantity_wasted'), row.get('cost_per_item'), row.get('total_waste_cost'))
            )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    load_waste_from_db.clear()
    return inserted

def insert_inventory_records_to_db(new_inv_df):
    """Insert new inventory rows into FACT_INVENTORY (+ any new dimension rows needed).
    Skips a row if an identical one (same date + ingredient + quantity)
    already exists, to prevent accidental double-inserts."""
    if not db_available():
        return 0
    conn = get_db_connection()
    inserted = 0
    for _, row in new_inv_df.iterrows():
        try:
            dk = get_or_create_date_key(conn, row['purchase_date'])
            ing_k = get_or_create_ingredient_key(conn, row['ingredient'], row.get('unit'), row.get('shelf_life_days'))
            alert_k = get_or_create_alert_key(conn, row.get('alert_level', 'Low Alert'))
            dupe = conn.execute(
                "SELECT 1 FROM FACT_INVENTORY WHERE date_key=? AND ingredient_key=? AND quantity=? LIMIT 1",
                (dk, ing_k, row.get('quantity'))
            ).fetchone()
            if dupe:
                continue
            _exp_val = row.get('expiration_date')
            try:
                _exp_str = pd.to_datetime(_exp_val).strftime('%Y-%m-%d') if pd.notna(_exp_val) else None
            except Exception:
                _exp_str = str(_exp_val).split(' ')[0] if _exp_val is not None else None
            conn.execute(
                "INSERT INTO FACT_INVENTORY (date_key, ingredient_key, alert_key, quantity, cost_per_unit, "
                "total_cost, expiration_date, shelf_life_days, days_until_expiration) VALUES (?,?,?,?,?,?,?,?,?)",
                (dk, ing_k, alert_k, row.get('quantity'), row.get('cost_per_unit'), row.get('total_cost'),
                 _exp_str, row.get('shelf_life_days'), row.get('days_until_expiration'))
            )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    load_inventory_from_db.clear()
    return inserted

# ── Load all data and models — database first, CSV fallback ────────────────
if db_available():
    sales_df     = load_sales_from_db()
    menu_df      = load_menu_from_db()
    inventory_df = load_inventory_from_db()
    waste_df     = load_waste_from_db()
else:
    sales_df     = load_csv(DATA_PATHS['sales'])
    menu_df      = load_csv(DATA_PATHS['menu'])
    inventory_df = load_csv(DATA_PATHS['inventory'])
    waste_df     = load_csv(DATA_PATHS['waste'])

# features_df loaded outside if/else — always available to all pages
features_df  = load_csv(DATA_PATHS['features'])
models       = {name: load_model(path) for name, path in MODEL_PATHS.items()}
metrics      = load_metrics()

# ── Drop known placeholder/template rows (e.g. "Product Name (regular)")
# from every dataset that carries an item name, right after loading — so
# they're gone from every page, chart, and KPI at once. See
# strip_placeholder_rows() above for the full explanation.
sales_df = strip_placeholder_rows(sales_df, col_candidates=('item', 'item_name'))
menu_df  = strip_placeholder_rows(menu_df,  col_candidates=('item_name',))
waste_df = strip_placeholder_rows(waste_df, col_candidates=('item_name',))

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown('''
    <div class="sidebar-brand">
        <div class="sidebar-brand-name">DineData</div>
        <div class="sidebar-brand-sub">Kôfētala Bistro</div>
    </div>
    ''', unsafe_allow_html=True)

    NAV_ITEMS = [
        'Dashboard Overview',
        'Sales Analytics',
        'Waste Analytics',
        'Menu Performance',
        'Inventory Status',
        'Forecast & Predictions',
        'Database',
    ]

    if 'page' not in st.session_state:
        st.session_state.page = NAV_ITEMS[0]

    st.markdown('<div class="sidebar-section-label">Navigate</div>', unsafe_allow_html=True)
    for item in NAV_ITEMS:
        is_active = st.session_state.page == item
        if st.button(
            item, key=f'nav_{item}', use_container_width=True,
            type='primary' if is_active else 'secondary'
        ):
            st.session_state.page = item
            st.rerun()

    page = st.session_state.page

    st.markdown('<div class="sidebar-section-label">Data Status</div>', unsafe_allow_html=True)
    status_rows = ""
    for label, df_obj in [('Sales', sales_df), ('Menu', menu_df),
                           ('Inventory', inventory_df), ('Waste', waste_df)]:
        dot_cls = 'status-dot-ok' if df_obj is not None else 'status-dot-bad'
        count = f'{len(df_obj):,} records' if df_obj is not None else 'Not found'
        status_rows += f'''
        <div class="status-row">
            <span><span class="status-dot {dot_cls}"></span>{label}</span>
            <span style="color:#B8A896;">{count}</span>
        </div>'''
    st.markdown(status_rows, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Model Status</div>', unsafe_allow_html=True)
    model_labels = {
        'demand':   'Demand Forecast',
        'waste':    'Waste Prediction',
        'menu':     'Menu Classifier',
        'spoilage': 'Alert Level Classifier',
    }
    model_rows = ""
    for key, label in model_labels.items():
        dot_cls = 'status-dot-ok' if models[key] is not None else 'status-dot-bad'
        state = 'Loaded' if models[key] is not None else 'Missing'
        model_rows += f'''
        <div class="status-row">
            <span><span class="status-dot {dot_cls}"></span>{label}</span>
            <span style="color:#B8A896;">{state}</span>
        </div>'''
    st.markdown(model_rows, unsafe_allow_html=True)

# ── Stop if no sales data ─────────────────────────────────────────────────
if sales_df is None:
    st.error("Sales data not found!")
    st.info("Go to the Database page to upload your sales log, or check that Data_Cleaning/kofe_tala_sales_data.csv exists.")
    st.stop()

# ============================================================================
# HELPER: APPLY RF MODELS ON UPLOADED DATAFRAME
# Runs all 4 Random Forest models and returns prediction dataframe
# ============================================================================

def run_rf_predictions(df):
    """
    Apply all trained Random Forest models on a new uploaded Peddlr CSV.
    Returns a dataframe with 4 prediction columns added.
    """
    results = df.copy()

    # Parse datetime if available
    # fillna(0) used on all time columns to handle NaT (null dates) gracefully
    if 'DATETIME' in results.columns:
        results['DATETIME'] = pd.to_datetime(results['DATETIME'], format='mixed', errors='coerce')
        results['hour']        = results['DATETIME'].dt.hour.fillna(0).astype(int)
        results['day_of_week'] = results['DATETIME'].dt.dayofweek.fillna(0).astype(int)
        results['month']       = results['DATETIME'].dt.month.fillna(1).astype(int)
        results['quarter']     = results['DATETIME'].dt.quarter.fillna(1).astype(int)
        # week_number: use fillna before astype to avoid NA → int error
        results['week_number'] = results['DATETIME'].dt.isocalendar().week.fillna(1).astype(int)
        results['is_weekend']  = results['day_of_week'].isin([5, 6]).astype(int)
        results['is_peak_hour']= results['hour'].isin([7,8,9,12,13,15,16]).astype(int)
        results['is_dry_season']= results['month'].isin([12,1,2,3,4,5]).astype(int)

    # Encode item
    if 'PRODUCT' in results.columns:
        results['item_encoded'] = pd.factorize(results['PRODUCT'])[0]

    # Cost and profit margin estimation
    if 'PRICE' in results.columns:
        results['price'] = pd.to_numeric(results['PRICE'], errors='coerce').fillna(0)
        results['cost']  = results['price'] * 0.30
        results['profit_margin'] = (results['price'] - results['cost']) / (results['price'] + 1e-6)
        results['gross_profit']  = results['price'] - results['cost']

    # Quantity
    if 'QTY' in results.columns:
        results['quantity'] = pd.to_numeric(results['QTY'], errors='coerce').fillna(1)

    # Rolling features — set to item mean as proxy for new data
    for col in ['rolling_7d_avg','rolling_30d_avg','lag_1d','demand_trend',
                'waste_rate','estimated_waste_qty','transaction_profit',
                'total_cost','revenue','is_high_waste_item']:
        results[col] = 0.0

    # Category one-hot columns — must match exactly what was used during model training
    # Updated to match actual Kôfētala menu categories
    for cat in ['cat_Coffee Based','cat_Kôfē Frappé','cat_Mini Bites',
                'cat_Sans Coffee','cat_Signature Kôfē']:
        results[cat] = 0

    # ── Run each model ───────────────────────────────────────────────────
    for model_key, col_name, label_map in [
        ('demand',   'Predicted Demand (units)', None),
        ('waste',    'Predicted Waste (units)',  None),
        ('menu',     'Menu Performance',  {0:'Reconsider', 1:'Improve', 2:'Keep'}),
        ('spoilage', 'Alert Level',       {0:'Low', 1:'Medium', 2:'High'}),
    ]:
        bundle = models.get(model_key)
        if bundle is None:
            results[col_name] = 'Model not trained'
            continue

        model    = bundle['model']
        features = [f for f in bundle['features'] if f in results.columns]

        if not features:
            results[col_name] = 'Features missing'
            continue

        X = results[features].fillna(0)
        preds = model.predict(X)

        if label_map:
            results[col_name] = [label_map.get(int(p), str(p)) for p in preds]
        else:
            results[col_name] = np.round(preds, 2)

    return results

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================

if page == 'Dashboard Overview':

    # ── Reuse the same DB-first dataframes loaded once near the top of this
    # file (identical source used by every other page), instead of separate
    # loaders — keeps record counts always in sync with the Database page
    # after an upload or delete.
    overview_waste = waste_df.copy() if waste_df is not None else pd.DataFrame()
    overview_sales = sales_df.copy() if sales_df is not None else pd.DataFrame()
    overview_inv   = inventory_df.copy() if inventory_df is not None else pd.DataFrame()
    overview_menu  = menu_df.copy() if menu_df is not None else pd.DataFrame()

    # ── Greeting hero banner — with today's date and day ────────────────
    _now      = datetime.now()
    _hour     = _now.hour
    _greeting = "Good Morning" if _hour < 12 else ("Good Afternoon" if _hour < 18 else "Good Evening")
    _day_name = _now.strftime('%A')           # e.g. Wednesday
    _date_str = _now.strftime('%B %d, %Y')   # e.g. September 03, 2026
    hero_banner(
        f"{_greeting}, Kôfētala Bistro — {_day_name}, {_date_str}",
        "Business Overview",
        f"A quick snapshot across Sales, Waste, Inventory, and Menu — "
        f"{len(overview_sales):,} sales, {len(overview_waste):,} waste, "
        f"{len(overview_inv):,} inventory, and {len(overview_menu):,} menu records tracked. "
        f"Visit each dedicated page in the sidebar for the full breakdown."
    )

    # ── KPI cards — one primary + one secondary metric per data area ────
    st.markdown("### Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rev = overview_sales['total'].sum() if 'total' in overview_sales.columns else 0
        stat_card("Sales · Revenue", f"₱{rev:,.2f}")
    with col2:
        total_waste_cost = overview_waste['total_waste_cost'].sum() if 'total_waste_cost' in overview_waste.columns else 0
        stat_card("Waste · Total Cost", f"₱{total_waste_cost:,.2f}")
    with col3:
        inv_value = overview_inv['total_cost'].sum() if 'total_cost' in overview_inv.columns else 0
        stat_card("Inventory · Value", f"₱{inv_value:,.2f}")
    with col4:
        stat_card("Menu · Items", f"{len(overview_menu):,}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        stat_card("Sales · Records", f"{len(overview_sales):,}")
    with col6:
        total_units_wasted = overview_waste['quantity_wasted'].sum() if 'quantity_wasted' in overview_waste.columns else 0
        stat_card("Waste · Units Wasted", f"{int(total_units_wasted):,}")
    with col7:
        oos = int((overview_inv['quantity'] <= 0).sum()) if 'quantity' in overview_inv.columns else 0
        stat_card("Inventory · Out of Stock", f"{oos:,}")
    with col8:
        n_cats_menu = overview_menu['category'].nunique() if 'category' in overview_menu.columns else 0
        stat_card("Menu · Categories", f"{n_cats_menu:,}")

    st.markdown("")
    st.markdown("### Snapshots")
    st.caption(
        "One quick chart per area — open **Sales Analytics**, **Waste Analytics**, "
        "**Inventory Status**, or **Menu Performance** in the sidebar for the full breakdown and filters."
    )

    # ============================================================================
    # SALES SNAPSHOT
    # ============================================================================
    with st.container(border=True):
        st.markdown("#### Sales Snapshot")
        if len(overview_sales) > 0 and 'date' in overview_sales.columns and 'total' in overview_sales.columns:
            # Same Year/Quarter/Month (+ Custom Date) filter used by the
            # Waste, Inventory, and Menu snapshots below — this chart was
            # missing it before, so it always showed all-time data
            # regardless of period, unlike every other snapshot on this page.
            sm_f = render_year_quarter_month_filter(overview_sales, 'ov_sales', date_col='date')
            if len(sm_f) == 0:
                st.info("No sales records for this selection.")
            else:
                sm = sm_f.copy()
                sm['month'] = sm['date'].dt.to_period('M').dt.to_timestamp()
                sm_rev = sm.groupby('month')['total'].sum().reset_index()
                sm_rev.columns = ['Month', 'Revenue']
                fig = px.area(sm_rev, x='Month', y='Revenue', color_discrete_sequence=[EARTH['primary']])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                if len(sm_rev) >= 2:
                    sales_dir = "growing" if sm_rev['Revenue'].iloc[-1] > sm_rev['Revenue'].iloc[0] else "declining"
                    chart_insight(
                        f"Monthly revenue is <b>{sales_dir}</b>. Latest month "
                        f"({sm_rev.iloc[-1]['Month'].strftime('%B %Y')}) recorded "
                        f"<b>₱{sm_rev.iloc[-1]['Revenue']:,.2f}</b> in revenue. "
                        f"See <b>Sales Analytics</b> for the full breakdown.",
                        'good' if sales_dir == 'growing' else 'warn'
                    )
                elif len(sm_rev) == 1:
                    chart_insight(
                        f"Only one month of data in this selection: "
                        f"<b>{sm_rev.iloc[0]['Month'].strftime('%B %Y')}</b> recorded "
                        f"<b>₱{sm_rev.iloc[0]['Revenue']:,.2f}</b> in revenue. "
                        f"See <b>Sales Analytics</b> for the full breakdown."
                    )
        else:
            st.info("No sales data yet. Go to the Database page to upload your sales log.")

    st.markdown("")

    # ============================================================================
    # WASTE SNAPSHOT
    # ============================================================================
    with st.container(border=True):
        st.markdown("#### Waste Snapshot")
        if len(overview_waste) > 0 and 'date' in overview_waste.columns and 'total_waste_cost' in overview_waste.columns:
            ow_years = sorted(overview_waste['date'].dt.year.dropna().unique().tolist(), reverse=True)

            # Filters — View By controls what other filters appear
            tf1, tf2, tf3 = st.columns(3)
            with tf1:
                ow_granularity = st.selectbox("View By", ['Weekly','Monthly','Yearly','Custom Date'], index=1, key='ov_trend_gran')
            with tf2:
                if ow_granularity == 'Custom Date':
                    ow_custom_range = st.date_input(
                        "Date Range",
                        value=(overview_waste['date'].min().date(), overview_waste['date'].max().date()),
                        min_value=overview_waste['date'].min().date(), max_value=overview_waste['date'].max().date(),
                        key='ov_trend_customrange'
                    )
                    ow_sel_year = 'All'
                elif ow_granularity != 'Yearly':
                    ow_sel_year = st.selectbox("Year", ['All'] + [str(y) for y in ow_years], key='ov_trend_year')
                else:
                    st.empty()
                    ow_sel_year = 'All'  # Yearly always shows all years
            with tf3:
                if ow_granularity == 'Weekly':
                    ow_month_opts = ['All months','January','February','March','April','May','June',
                                     'July','August','September','October','November','December']
                    ow_sel_month = st.selectbox("Month (Week 1-4)", ow_month_opts, key='ov_trend_month')
                else:
                    st.empty()
                    ow_sel_month = 'All months'

            # Apply filters
            ow = overview_waste.copy()
            if ow_granularity != 'Custom Date' and ow_sel_year != 'All':
                ow = ow[ow['date'].dt.year == int(ow_sel_year)]

            if ow_granularity == 'Custom Date':
                if isinstance(ow_custom_range, tuple) and len(ow_custom_range) == 2:
                    ow = ow[(ow['date'].dt.date >= ow_custom_range[0]) & (ow['date'].dt.date <= ow_custom_range[1])]
                ow['period'] = ow['date'].dt.date
                x_lbl = 'Date'
            elif ow_granularity == 'Weekly':
                if ow_sel_month != 'All months':
                    ow = ow[ow['date'].dt.month_name() == ow_sel_month]
                    ow['period'] = 'Week ' + (((ow['date'].dt.day - 1) // 7) + 1).clip(upper=4).astype(str)
                    x_lbl = f'Week of {ow_sel_month}'
                else:
                    ow['period'] = ow['date'].dt.to_period('W').apply(lambda r: r.start_time)
                    x_lbl = 'Week'
            elif ow_granularity == 'Yearly':
                ow['period'] = ow['date'].dt.year
                x_lbl = 'Year'
            else:
                ow['period'] = ow['date'].dt.to_period('M').dt.to_timestamp()
                x_lbl = 'Month'

            trend_df = ow.groupby('period')['total_waste_cost'].sum().reset_index()
            trend_df.columns = [x_lbl, 'Waste Cost']
            if ow_granularity == 'Weekly' and ow_sel_month != 'All months':
                trend_df = trend_df.sort_values(x_lbl, key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))

            if len(trend_df) == 0:
                st.info("No records for this selection.")
            else:
                fig = px.line(trend_df, x=x_lbl, y='Waste Cost', markers=True,
                              color_discrete_sequence=[EARTH['accent']])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    xaxis_title=x_lbl,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                if len(trend_df) >= 2:
                    first_half_avg  = trend_df['Waste Cost'].iloc[:len(trend_df)//2].mean()
                    second_half_avg = trend_df['Waste Cost'].iloc[len(trend_df)//2:].mean()
                    _direction = "increasing" if second_half_avg > first_half_avg else "decreasing"
                    latest_p   = trend_df.iloc[-1]
                    try:
                        if ow_granularity == 'Yearly':
                            _lbl = str(latest_p[x_lbl])
                        elif ow_granularity == 'Weekly' and ow_sel_month != 'All months':
                            _lbl = str(latest_p[x_lbl])  # e.g. "Week 4"
                        elif ow_granularity == 'Custom Date':
                            _lbl = pd.to_datetime(str(latest_p[x_lbl])).strftime('%b %d, %Y')
                        else:
                            _lbl = pd.to_datetime(str(latest_p[x_lbl])).strftime('%B %Y')
                    except Exception:
                        _lbl = str(latest_p[x_lbl])
                    _period_word = 'day' if ow_granularity == 'Custom Date' else (
                        ow_granularity.lower()[:-2] if ow_granularity.endswith('ly') else ow_granularity.lower()
                    )
                    chart_insight(
                        f"Waste cost is <b>{_direction}</b> over time. The most recent {_period_word} "
                        f"({_lbl}) recorded <b>₱{latest_p['Waste Cost']:,.2f}</b> in waste. "
                        f"See <b>Waste Analytics</b> for category and item-level breakdowns.",
                        'warn' if _direction == 'increasing' else 'good'
                    )
        else:
            st.info("No waste data yet. Go to the Database page to upload your waste log.")

    st.markdown("")

    # ============================================================================
    # INVENTORY SNAPSHOT
    # ============================================================================
    with st.container(border=True):
        st.markdown("#### Inventory Snapshot")
        if len(overview_inv) > 0 and 'purchase_date' in overview_inv.columns:
            inv_ov = overview_inv.copy()
            inv_ov['purchase_date'] = pd.to_datetime(inv_ov['purchase_date'], errors='coerce')
            inv_ov_f = render_year_quarter_month_filter(inv_ov, 'ov_inv', date_col='purchase_date')

            if 'alert_level' not in inv_ov_f.columns:
                if 'spoilage_risk' in inv_ov_f.columns:
                    inv_ov_f['alert_level'] = inv_ov_f['spoilage_risk'].replace({
                        'High Risk': 'High Alert', 'Medium Risk': 'Medium Alert',
                        'Low Risk': 'Low Alert', 'Expired': 'Expired',
                    })
                else:
                    inv_ov_f['alert_level'] = 'Low Alert'
            if 'quantity' in inv_ov_f.columns:
                inv_ov_f.loc[inv_ov_f['quantity'] <= 0, 'alert_level'] = 'Out of Stock'

            ALERT_COLORS_OV = {
                'High Alert':   EARTH['danger'],
                'Medium Alert': EARTH['warning'],
                'Low Alert':    EARTH['success'],
                'Expired':      '#4E342E',
                'Out of Stock': '#8D6E63',
            }
            alert_order_ov = ['High Alert','Medium Alert','Low Alert','Expired','Out of Stock']

            if len(inv_ov_f) == 0:
                st.info("No inventory records for this selection.")
            else:
                dist_inv = inv_ov_f['alert_level'].value_counts().reindex(alert_order_ov).fillna(0).reset_index()
                dist_inv.columns = ['Alert Level', 'Count']
                fig = px.bar(dist_inv, x='Alert Level', y='Count',
                             color='Alert Level', color_discrete_map=ALERT_COLORS_OV,
                             text_auto=True, category_orders={'Alert Level': alert_order_ov})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                urgent_ct = int(dist_inv.loc[dist_inv['Alert Level'].isin(['High Alert','Out of Stock']), 'Count'].sum())
                chart_insight(
                    f"<b>{urgent_ct}</b> ingredient record(s) in this selection are High Alert or Out of Stock. "
                    f"See <b>Inventory Status</b> for the full restock guide.",
                    'warn' if urgent_ct > 0 else 'good'
                )
        else:
            st.info("No inventory data yet. Go to the Database page to upload your inventory records.")

    st.markdown("")

    # ============================================================================
    # MENU SNAPSHOT
    # ============================================================================
    with st.container(border=True):
        st.markdown("#### Menu Snapshot")
        # Menu items (DIM_ITEM) don't carry a date, so the date filter here
        # runs on the sales log instead — showing which menu categories
        # actually sold within the selected period.
        if len(overview_sales) > 0 and 'category' in overview_sales.columns and 'date' in overview_sales.columns:
            menu_sales_f = render_year_quarter_month_filter(overview_sales, 'ov_menu', date_col='date')
            if len(menu_sales_f) == 0 or 'total' not in menu_sales_f.columns:
                st.info("No sales records for this selection.")
            else:
                menu_cat_dist = menu_sales_f.groupby('category')['total'].sum().reset_index()
                menu_cat_dist.columns = ['Category', 'Revenue']
                fig = px.bar(menu_cat_dist.sort_values('Revenue', ascending=False),
                             x='Category', y='Revenue',
                             color_discrete_sequence=[EARTH['secondary']], text_auto=',.0f')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                top_cat_menu = menu_cat_dist.sort_values('Revenue', ascending=False).iloc[0]
                n_menu_items = len(overview_menu) if len(overview_menu) > 0 else 0
                chart_insight(
                    f"<b>{top_cat_menu['Category']}</b> generated the most revenue in this period "
                    f"(₱{top_cat_menu['Revenue']:,.2f}). Your menu has <b>{n_menu_items}</b> items total. "
                    f"See <b>Menu Performance</b> for Keep/Improve/Reconsider recommendations."
                )
        elif len(overview_menu) > 0 and 'category' in overview_menu.columns:
            menu_cat_dist = overview_menu['category'].value_counts().reset_index()
            menu_cat_dist.columns = ['Category', 'Items']
            fig = px.bar(menu_cat_dist, x='Category', y='Items',
                         color_discrete_sequence=[EARTH['secondary']], text_auto=True)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("No dated sales records available yet, so this shows item counts by category instead of revenue.")

            top_cat_menu = menu_cat_dist.sort_values('Items', ascending=False).iloc[0]
            chart_insight(
                f"<b>{len(overview_menu)}</b> menu items across <b>{len(menu_cat_dist)}</b> categories. "
                f"<b>{top_cat_menu['Category']}</b> has the most items ({int(top_cat_menu['Items'])}). "
                f"See <b>Menu Performance</b> for Keep/Improve/Reconsider recommendations."
            )
        else:
            st.info("No menu data yet. Go to the Database page to upload your menu items.")

# ============================================================================
# PAGE 2: SALES ANALYTICS
# ============================================================================

elif page == 'Sales Analytics':

    st.title("Sales Analytics")
    st.caption("To add new sales records, go to the **Database** page — all data uploads now happen there.")

    def _sa_apply_filters(base_df, key_prefix):
        """Renders its own Year/Quarter/Month filter row (Month options
        cascade off the chosen Quarter) and returns the filtered
        dataframe — kept local to each chart so no two visualizations on
        this page share the same filter controls."""
        return render_year_quarter_month_filter(base_df, key_prefix, date_col='date')

    # ── Business Summary — boxed and filterable, uniform with the Waste
    # Summary box on Waste Analytics and the Performance Distribution box
    # on Menu Performance. This absorbs what used to be a separate
    # "Transaction KPIs" section further down the page, so the headline
    # numbers for Sales Analytics all live together at the top now.
    with st.container(border=True):
        st.markdown("### Business Summary")
        kpi_src = _sa_apply_filters(sales_df, 'sa_summary')
        st.caption(f"Showing **{len(kpi_src):,}** of {len(sales_df):,} records")

        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
        with bcol1:
            stat_card("Total Transactions", f"{len(kpi_src):,}")
        with bcol2:
            rev = kpi_src['total'].sum() if 'total' in kpi_src.columns else 0
            stat_card("Total Revenue", f"₱{rev:,.2f}")
        with bcol3:
            qty = kpi_src['quantity'].sum() if 'quantity' in kpi_src.columns else 0
            stat_card("Units Sold", f"{qty:,.0f}")
        with bcol4:
            avg = kpi_src['total'].mean() if 'total' in kpi_src.columns else 0
            stat_card("Avg Transaction", f"₱{avg:,.2f}")
        with bcol5:
            # Count total menu items from the menu master list
            n_menu = len(menu_df) if menu_df is not None else 0
            stat_card("Menu Items", n_menu)

    st.markdown("")
    st.markdown("### Sales Overview")

    st.markdown("")

    with st.container(border=True):
        st.markdown("#### Revenue by Category")
        cat_rev_src = _sa_apply_filters(sales_df, 'sa_catrev')
        cat_col = 'category' if 'category' in cat_rev_src.columns else None
        if cat_col and 'total' in cat_rev_src.columns and len(cat_rev_src) > 0:
            cat_rev = cat_rev_src.groupby(cat_col)['total'].sum().reset_index()
            fig = px.pie(cat_rev, values='total', names=cat_col,
                         hole=0.45,
                         color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textposition='outside', textinfo='percent+label')
            fig.update_layout(
                showlegend=True, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            if len(cat_rev) > 0:
                cat_rev_sorted = cat_rev.sort_values('total', ascending=False)
                cat_rev_breakdown = full_breakdown_html(
                    list(zip(cat_rev_sorted[cat_col], cat_rev_sorted['total'])), prefix='₱'
                )
                chart_insight(
                    f"Revenue by category — {cat_rev_breakdown}. "
                    f"<b>{cat_rev_sorted.iloc[0][cat_col]}</b> is your top revenue category."
                )
        else:
            st.info("No sales records for this filter combination.")

    st.markdown("")

    with st.container(border=True):
        st.markdown("#### Performance by Day of Week")
        dow_metric = st.selectbox("Metric", ['Revenue', 'Transactions'], key='sa_dow_metric')
        dow_src = _sa_apply_filters(sales_df, 'sa_dow')
        if 'date' in dow_src.columns and len(dow_src) > 0:
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            fc = dow_src.copy()
            fc['day'] = fc['date'].dt.day_name()

            if dow_metric == 'Revenue' and 'total' in fc.columns:
                dow_tbl = fc.groupby('day')['total'].sum().reindex(day_order).reset_index()
                dow_tbl.columns = ['day', 'value']
                y_label, y_prefix, text_fmt = 'Revenue (₱)', '₱', '.2s'
            else:
                dow_tbl = fc['day'].value_counts().reindex(day_order).reset_index()
                dow_tbl.columns = ['day', 'value']
                y_label, y_prefix, text_fmt = 'Transactions', '', True

            if dow_tbl['value'].notna().any():
                max_day = dow_tbl['value'].idxmax()
                colors  = [EARTH['primary'] if i == max_day else EARTH['light']
                           for i in range(len(dow_tbl))]
                fig = px.bar(dow_tbl, x='day', y='value',
                             color_discrete_sequence=[EARTH['primary']],
                             text_auto=text_fmt)
                fig.update_traces(marker_color=colors)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix=y_prefix, tickformat=',.0f'),
                    xaxis_title='', yaxis_title=y_label,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                dow_sorted = dow_tbl.dropna(subset=['value']).sort_values('value', ascending=False)
                if dow_metric == 'Revenue':
                    dow_breakdown = full_breakdown_html(
                        list(zip(dow_sorted['day'], dow_sorted['value'])), prefix='₱'
                    )
                    chart_insight(
                        f"Revenue by day of week — {dow_breakdown}. "
                        f"<b>{dow_sorted.iloc[0]['day']}</b> generates the most revenue — "
                        f"consider extra staffing or promos on this day."
                    )
                else:
                    dow_breakdown = full_breakdown_html(
                        list(zip(dow_sorted['day'], dow_sorted['value'])), suffix=' transactions', decimals=0, show_pct=False
                    )
                    chart_insight(
                        f"Transactions by day — {dow_breakdown}. "
                        f"<b>{dow_sorted.iloc[0]['day']}</b> is the busiest day."
                    )
            else:
                st.info("No sales data for this filter combination.")
        else:
            st.info("No sales data for this filter combination.")

    st.markdown("")

    with st.container(border=True):
        st.markdown("#### Sales Heatmap (Hour × Day)")
        if 'date' in sales_df.columns and len(sales_df) > 0:
            hm = render_year_quarter_month_filter(sales_df, 'sales_heatmap', date_col='date')
            hm['hour'] = hm['date'].dt.hour
            hm['day']  = hm['date'].dt.day_name()
            day_order2 = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pivot = hm.groupby(['day','hour']).size().unstack(fill_value=0)
            pivot = pivot.reindex([d for d in day_order2 if d in pivot.index])
            if len(pivot) == 0:
                st.info("No sales records for this selection.")
            else:
                fig = px.imshow(pivot,
                                color_continuous_scale=['#FAF6F1','#C4A882','#6F4E37'],
                                aspect='auto',
                                labels=dict(x='Hour of Day', y='Day', color='Transactions'))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                if pivot.values.sum() > 0:
                    by_day_totals = pivot.sum(axis=1).sort_values(ascending=False)
                    busiest_day_sa = by_day_totals.index[0]
                    busiest_hour = pivot.sum(axis=0).idxmax()
                    day_txn_breakdown = full_breakdown_html(
                        list(by_day_totals.items()), suffix=' transactions', decimals=0, show_pct=False
                    )
                    chart_insight(
                        f"Transactions by day — {day_txn_breakdown}. "
                        f"<b>{busiest_day_sa}</b> is the busiest day overall, and "
                        f"<b>{busiest_hour}:00</b> is the busiest hour across the week. "
                        f"Darker cells mark your peak traffic windows."
                    )

    st.markdown("---")

    # Transaction KPIs now live in the boxed "Business Summary" section at
    # the top of this page (see above) instead of here.
    item_col = 'item' if 'item' in sales_df.columns else \
               'item_name' if 'item_name' in sales_df.columns else None

    # Horizontal bar — top 15 items — own filter
    with st.container(border=True):
        st.markdown("#### Top 15 Items by Quantity")
        top15_src = _sa_apply_filters(sales_df, 'sa_top15')
        if item_col and 'quantity' in top15_src.columns and len(top15_src) > 0:
            top15_src['quantity'] = pd.to_numeric(top15_src['quantity'], errors='coerce')
            top15_src, top15_item_disp, _ = add_normalized_keys(top15_src, item_col=item_col)
            top = top15_src.groupby('_item_key')['quantity'].sum().nlargest(15).reset_index()
            top[item_col] = top['_item_key'].map(top15_item_disp)
            top = top[[item_col, 'quantity']]
            fig = px.bar(top, x='quantity', y=item_col, orientation='h',
                         color_discrete_sequence=[EARTH['primary']],
                         text_auto=True)
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            top5 = top.sort_values('quantity', ascending=False).head(5)
            top5_html = full_breakdown_html(
                list(zip(top5[item_col], top5['quantity'])), suffix=' units', decimals=0, show_pct=False
            )
            chart_insight(
                f"Top 5 sellers — {top5_html}. Together, all {len(top)} items shown total "
                f"<b>{top['quantity'].sum():,.0f}</b> units sold in this period."
            )
        else:
            st.info("No sales records for this filter combination.")

    st.markdown("")

    # ── Sales Trend — merges the old "Daily Revenue Trend" and "Sales
    # Trend" charts into one: pick any metric, any granularity (Daily up
    # to Yearly, or an exact Custom Date range), with a moving-average
    # overlay at the finer granularities where a single day/week can be
    # noisy on its own. ──────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Sales Trend")
        trend_src = _sa_apply_filters(sales_df, 'sa_trend')
        if 'date' in trend_src.columns and len(trend_src) > 0:
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                metric_choice = st.selectbox(
                    "Metric to plot", ['Revenue', 'Transactions', 'Units Sold'],
                    key='monthly_trend_metric'
                )
            with tcol2:
                granularity = st.selectbox(
                    "View by", ['Daily', 'Weekly', 'Monthly', 'Yearly', 'Custom Date'],
                    index=2, key='trend_granularity'
                )
            with tcol3:
                if granularity == 'Custom Date':
                    sa_trend_custom_range = st.date_input(
                        "Date Range",
                        value=(trend_src['date'].min().date(), trend_src['date'].max().date()),
                        min_value=trend_src['date'].min().date(), max_value=trend_src['date'].max().date(),
                        key='sa_trend_customrange'
                    )
                else:
                    st.empty()

            fc2 = trend_src.copy()
            if granularity == 'Custom Date':
                if isinstance(sa_trend_custom_range, tuple) and len(sa_trend_custom_range) == 2:
                    fc2 = fc2[(fc2['date'].dt.date >= sa_trend_custom_range[0]) & (fc2['date'].dt.date <= sa_trend_custom_range[1])]
                fc2['period'] = fc2['date'].dt.date
                x_title, ma_window = 'Date', 7
            elif granularity == 'Daily':
                fc2['period'] = fc2['date'].dt.date
                x_title, ma_window = 'Date', 7
            elif granularity == 'Weekly':
                fc2['period'] = fc2['date'].dt.to_period('W').dt.start_time
                x_title, ma_window = 'Week', 4
            elif granularity == 'Monthly':
                fc2['period'] = fc2['date'].dt.to_period('M').dt.to_timestamp()
                x_title, ma_window = 'Month', 3
            else:
                fc2['period'] = fc2['date'].dt.to_period('Y').dt.to_timestamp()
                x_title, ma_window = 'Year', 1  # no smoothing needed at yearly granularity

            agg_map = {}
            if 'total' in fc2.columns:    agg_map['total'] = 'sum'
            if 'quantity' in fc2.columns: agg_map['quantity'] = 'sum'

            trend_tbl = fc2.groupby('period').agg(
                transactions=('date', 'count'),
                **({'revenue': ('total', 'sum')} if 'total' in fc2.columns else {}),
                **({'units': ('quantity', 'sum')} if 'quantity' in fc2.columns else {})
            ).reset_index().sort_values('period')

            y_map = {
                'Revenue':      ('revenue', 'Revenue (₱)', '₱'),
                'Transactions': ('transactions', 'Transactions', ''),
                'Units Sold':   ('units', 'Units Sold', ''),
            }
            y_col, y_label, y_prefix = y_map[metric_choice]

            if y_col in trend_tbl.columns:
                show_ma = ma_window > 1 and len(trend_tbl) > 1
                if show_ma:
                    trend_tbl[f'{ma_window}-period MA'] = trend_tbl[y_col].rolling(ma_window, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_tbl['period'], y=trend_tbl[y_col],
                    mode='lines+markers', name=metric_choice,
                    line=dict(color=EARTH['light'] if show_ma else EARTH['accent'], width=1 if show_ma else 2.5),
                    opacity=0.6 if show_ma else 1
                ))
                if show_ma:
                    fig.add_trace(go.Scatter(
                        x=trend_tbl['period'], y=trend_tbl[f'{ma_window}-period MA'],
                        mode='lines', name=f'{ma_window}-period MA',
                        line=dict(color=EARTH['primary'], width=2.5)
                    ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=x_title, yaxis_title=y_label,
                    yaxis=dict(tickprefix=y_prefix, tickformat=',.0f'),
                    legend=dict(orientation='h', y=1.1),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                if len(trend_tbl) >= 2:
                    trend_delta = trend_tbl[y_col].iloc[-1] - trend_tbl[y_col].iloc[0]
                    trend_word = "grown" if trend_delta > 0 else "declined"
                    peak_row = trend_tbl.loc[trend_tbl[y_col].idxmax()]
                    low_row  = trend_tbl.loc[trend_tbl[y_col].idxmin()]
                    chart_insight(
                        f"{metric_choice} has <b>{trend_word}</b> from "
                        f"{y_prefix}{trend_tbl[y_col].iloc[0]:,.0f} to <b>{y_prefix}{trend_tbl[y_col].iloc[-1]:,.0f}</b> "
                        f"across this range ({len(trend_tbl)} {x_title.lower()}(s) shown). "
                        f"The highest point was <b>{y_prefix}{peak_row[y_col]:,.0f}</b> ({peak_row['period']}) and the "
                        f"lowest was <b>{y_prefix}{low_row[y_col]:,.0f}</b> ({low_row['period']}); average across the "
                        f"range was <b>{y_prefix}{trend_tbl[y_col].mean():,.0f}</b>."
                    )
            else:
                st.info(f"'{metric_choice}' isn't available in this dataset.")
        else:
            st.info("No sales records for this filter combination.")

# ============================================================================
# PAGE 3: WASTE ANALYSIS
# ============================================================================

elif page == 'Waste Analytics':

    st.title("Waste Analytics")
    st.caption("To add new waste records, go to the **Database** page — all data uploads now happen there.")

    # ── WASTE DATA FILE PATH — saved in Data_Cleaning folder ─────────────
    WASTE_FILE = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_waste_data.csv')

    @st.cache_data(ttl=300)
    def load_waste_data(_path=None):
        """Load existing waste CSV — cached for fast page switching."""
        path = _path or WASTE_FILE
        try:
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame(columns=[
                'date','item_name','category','quantity_wasted',
                'waste_reason','cost_per_item','total_waste_cost'
            ])

    # ── Load existing waste data (CSV fallback when DB isn't available) ────
    existing_waste = load_waste_data()

    # ── Current Waste Data + Analytics ─────────────────────────
    # Prefer the same DB-first waste_df used by Dashboard Overview and the
    # Database page, so a delete/edit made on the Database page (which only
    # touches the database, not this CSV) is reflected here immediately too.
    display_waste = waste_df if db_available() else existing_waste

    if display_waste is not None and len(display_waste) > 0:

        def _wa_apply_filters(base_df, key_prefix, show_reason=True):
            """Renders the same Year/Quarter/Month (+ Custom Date) filter row
            used by Transaction KPIs on Sales Analytics and Performance
            Distribution on Menu Performance, plus a Category select and
            (optionally) a Waste Reason select — kept local to each chart so
            no two visualizations on this page share the same filter
            controls."""
            df_f = render_year_quarter_month_filter(base_df, key_prefix, date_col='date') \
                if 'date' in base_df.columns else base_df.copy()
            if 'category' in df_f.columns:
                cats = sorted(df_f['category'].dropna().unique().tolist())
                sel_cat = st.selectbox("Category", ['All'] + cats, key=f'{key_prefix}_cat')
                if sel_cat != 'All':
                    df_f = df_f[df_f['category'] == sel_cat]
            if show_reason and 'waste_reason' in df_f.columns:
                reasons = sorted(df_f['waste_reason'].dropna().unique().tolist())
                sel_reason = st.selectbox("Waste Reason", ['All'] + reasons, key=f'{key_prefix}_reason')
                if sel_reason != 'All':
                    df_f = df_f[df_f['waste_reason'] == sel_reason]
            return df_f

        # ── Waste Summary (KPIs) — boxed, uniform with the Business Summary
        # box on Sales Analytics and the Performance Distribution box on
        # Menu Performance ───────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("### Waste Summary")
            kpi_waste = _wa_apply_filters(display_waste, 'wa_kpi')
            st.caption(f"Showing **{len(kpi_waste):,}** of {len(display_waste):,} waste records")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                total_qty = kpi_waste['quantity_wasted'].sum() if 'quantity_wasted' in kpi_waste.columns else 0
                stat_card("Total Units Wasted", f"{int(total_qty):,}")
            with k2:
                total_cost = kpi_waste['total_waste_cost'].sum() if 'total_waste_cost' in kpi_waste.columns else 0
                stat_card("Total Waste Cost", f"₱{total_cost:,.2f}")
            with k3:
                n_items = kpi_waste['item_name'].nunique() if 'item_name' in kpi_waste.columns else 0
                stat_card("Unique Items Wasted", n_items)
            with k4:
                n_days = kpi_waste['date'].nunique() if 'date' in kpi_waste.columns else 0
                stat_card("Days Tracked", n_days)

        st.markdown("")

        # ── Charts ────────────────────────────────────────────────────────
        # Bar — waste cost by item (top 10) — own filter
        with st.container(border=True):
            st.markdown("#### Top 10 Items by Waste Cost")
            top10_waste = _wa_apply_filters(display_waste, 'wa_top10')
            if 'item_name' in top10_waste.columns and 'total_waste_cost' in top10_waste.columns and len(top10_waste) > 0:
                top10_waste, top10_item_disp, _ = add_normalized_keys(top10_waste, item_col='item_name')
                top_items = (
                    top10_waste.groupby('_item_key')['total_waste_cost']
                    .sum().nlargest(10).reset_index()
                )
                top_items['item_name'] = top_items['_item_key'].map(top10_item_disp)
                top_items = top_items[['item_name', 'total_waste_cost']]
                top_items.columns = ['Item', 'Waste Cost']
                fig = px.bar(top_items, x='Waste Cost', y='Item',
                             orientation='h',
                             color='Waste Cost',
                             color_continuous_scale=['#C4A882','#6F4E37'],
                             text_auto=',.0f')
                fig.update_traces(texttemplate='₱%{x:,.0f}', textposition='outside')
                fig.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False,
                    xaxis_title='Waste Cost (₱)',
                    margin=dict(l=0,r=80,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                top_items_3 = top_items.sort_values('Waste Cost', ascending=False).head(3)
                top_items_html = full_breakdown_html(
                    list(zip(top_items_3['Item'], top_items_3['Waste Cost'])), prefix='₱', show_pct=False
                )
                chart_insight(
                    f"Top 3 by waste cost — {top_items_html}. Together, all {len(top_items)} items shown "
                    f"total <b>₱{top_items['Waste Cost'].sum():,.2f}</b> in this filtered view."
                )
            else:
                st.info("No waste records for this filter combination.")

        st.markdown("")

        # Bar — waste by reason — own filter (no Reason filter here, chart IS the reason breakdown)
        with st.container(border=True):
            st.markdown("#### Waste by Reason")
            reason_waste = _wa_apply_filters(display_waste, 'wa_reason', show_reason=False)
            if 'waste_reason' in reason_waste.columns and 'total_waste_cost' in reason_waste.columns and len(reason_waste) > 0:
                reason = (
                    reason_waste.groupby('waste_reason')['total_waste_cost']
                    .sum().sort_values().reset_index()
                )
                reason.columns = ['Reason', 'Waste Cost']
                reason_colors = {
                    'Over-preparation': EARTH['accent'],
                    'Over-prepared':    EARTH['accent'],
                    'Spoilage':         EARTH['danger'],
                    'Spoiled':          EARTH['danger'],
                    'Expired':          '#5D4037',
                    'Quality issue':    EARTH['warning'],
                }
                fig = px.bar(reason, x='Waste Cost', y='Reason',
                             orientation='h',
                             color='Reason',
                             color_discrete_map=reason_colors,
                             text_auto=',.0f')
                fig.update_traces(texttemplate='₱%{x:,.0f}', textposition='outside')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis_title='Waste Cost (₱)',
                    margin=dict(l=0,r=80,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                reason_sorted = reason.sort_values('Waste Cost', ascending=False)
                reason_breakdown = full_breakdown_html(
                    list(zip(reason_sorted['Reason'], reason_sorted['Waste Cost'])), prefix='₱'
                )
                chart_insight(
                    f"Waste cost by reason — {reason_breakdown}. "
                    f"<b>{reason_sorted.iloc[0]['Reason']}</b> is the leading cause of waste in this view."
                )

        st.markdown("")

        # Waste trend — its own filter (granularity + category + time period)
        with st.container(border=True):
            st.markdown("#### Waste Cost Trend")
            trend_waste = _wa_apply_filters(display_waste, 'wa_trend', show_reason=False)
            if 'date' in trend_waste.columns and 'total_waste_cost' in trend_waste.columns and len(trend_waste) > 0:
                wtc1, wtc2 = st.columns([1, 2])
                with wtc1:
                    trend_granularity = st.selectbox(
                        "View by", ['Daily', 'Weekly', 'Monthly', 'Custom Date'], index=2, key='waste_trend_granularity'
                    )
                with wtc2:
                    if trend_granularity == 'Custom Date':
                        wa_trend_custom_range = st.date_input(
                            "Date Range",
                            value=(trend_waste['date'].min().date(), trend_waste['date'].max().date()),
                            min_value=trend_waste['date'].min().date(), max_value=trend_waste['date'].max().date(),
                            key='wa_trend_customrange'
                        )
                    else:
                        st.empty()

                twaste = trend_waste.copy()

                if trend_granularity == 'Custom Date':
                    if isinstance(wa_trend_custom_range, tuple) and len(wa_trend_custom_range) == 2:
                        twaste = twaste[(twaste['date'].dt.date >= wa_trend_custom_range[0]) & (twaste['date'].dt.date <= wa_trend_custom_range[1])]
                    twaste['period'] = twaste['date'].dt.date
                    ma_window = 7
                elif trend_granularity == 'Daily':
                    twaste['period'] = twaste['date'].dt.date
                    ma_window = 7
                elif trend_granularity == 'Weekly':
                    twaste['period'] = twaste['date'].dt.to_period('W').dt.start_time
                    ma_window = 4
                else:
                    twaste['period'] = twaste['date'].dt.to_period('M').dt.to_timestamp()
                    ma_window = 3

                daily = twaste.groupby('period')['total_waste_cost'].sum().reset_index()
                daily.columns = ['date', 'waste_cost']
                daily[f'{ma_window}-period avg'] = daily['waste_cost'].rolling(ma_window, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily['date'], y=daily['waste_cost'],
                    name=f'{trend_granularity} Waste Cost',
                    mode='lines+markers',
                    line=dict(color=EARTH['light'], width=1.5)
                ))
                fig.add_trace(go.Scatter(
                    x=daily['date'], y=daily[f'{ma_window}-period avg'],
                    name=f'{ma_window}-period Average',
                    line=dict(color=EARTH['primary'], width=2.5),
                    mode='lines'
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    legend=dict(orientation='h', y=1.1),
                    margin=dict(l=0,r=0,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                if len(daily) >= 2:
                    trend_dir_wa = "rising" if daily[f'{ma_window}-period avg'].iloc[-1] > daily[f'{ma_window}-period avg'].iloc[0] else "falling"
                    peak_row_wa = daily.loc[daily['waste_cost'].idxmax()]
                    low_row_wa  = daily.loc[daily['waste_cost'].idxmin()]
                    chart_insight(
                        f"The waste cost trend is <b>{trend_dir_wa}</b> over this range ({len(daily)} "
                        f"{trend_granularity.lower()} periods). Latest {ma_window}-period average: "
                        f"<b>₱{daily[f'{ma_window}-period avg'].iloc[-1]:,.2f}</b>. The highest single period was "
                        f"<b>₱{peak_row_wa['waste_cost']:,.2f}</b> "
                        f"({pd.to_datetime(peak_row_wa['date']).strftime('%Y-%m-%d')}), the lowest was "
                        f"<b>₱{low_row_wa['waste_cost']:,.2f}</b> "
                        f"({pd.to_datetime(low_row_wa['date']).strftime('%Y-%m-%d')}), and the overall "
                        f"average was <b>₱{daily['waste_cost'].mean():,.2f}</b> per {trend_granularity.lower()[:-2]}.",
                        'warn' if trend_dir_wa == 'rising' else 'good'
                    )
            else:
                st.info("No waste records for this filter combination.")

        st.markdown("")

        # ── Waste by Day of Week — which day wastes the most — own filter ──
        with st.container(border=True):
            st.markdown("#### Waste by Day of Week")
            dow_waste_src = _wa_apply_filters(display_waste, 'wa_dow')
            if 'date' in dow_waste_src.columns and 'total_waste_cost' in dow_waste_src.columns and len(dow_waste_src) > 0:
                day_order_w = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                dw = dow_waste_src.copy()
                dw['day'] = dw['date'].dt.day_name()
                dow_waste = dw.groupby('day')['total_waste_cost'].sum().reindex(day_order_w).fillna(0).reset_index()
                dow_waste.columns = ['Day', 'Waste Cost']
                worst_day_idx = dow_waste['Waste Cost'].idxmax()
                bar_colors = [EARTH['danger'] if i == worst_day_idx else EARTH['light']
                              for i in range(len(dow_waste))]
                fig = px.bar(dow_waste, x='Day', y='Waste Cost', text_auto=',.0f')
                fig.update_traces(marker_color=bar_colors, texttemplate='₱%{y:,.0f}', textposition='outside')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    margin=dict(l=0,r=0,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                dow_waste_sorted = dow_waste.sort_values('Waste Cost', ascending=False)
                dow_waste_html = full_breakdown_html(
                    list(zip(dow_waste_sorted['Day'], dow_waste_sorted['Waste Cost'])), prefix='₱'
                )
                chart_insight(
                    f"Waste cost by day of week — {dow_waste_html}. "
                    f"<b>{dow_waste.loc[worst_day_idx,'Day']}</b> has the highest waste cost — "
                    f"consider adjusting prep quantities on that day.",
                    'warn'
                )
            else:
                st.info("No waste records for this filter combination.")

        st.markdown("")

        # ── Waste Heatmap — day of week × week/month, fully self-contained ──
        with st.container(border=True):
            st.markdown("#### Waste Heatmap")
            st.caption("Darker cells = higher waste cost.")
            if 'date' in display_waste.columns and 'total_waste_cost' in display_waste.columns:
                hcolw1, hcolw2, hcolw3, hcolw4 = st.columns(4)
                with hcolw1:
                    heatmap_granularity = st.selectbox(
                        "View by", ['Weekly', 'Monthly', 'Yearly', 'Custom Date'], index=1, key='waste_heatmap_granularity'
                    )
                with hcolw2:
                    if 'category' in display_waste.columns:
                        cats_hm = sorted(display_waste['category'].dropna().unique().tolist())
                        sel_cat_hm = st.selectbox("Category", ['All'] + cats_hm, key='waste_heatmap_category')
                    else:
                        sel_cat_hm = 'All'
                with hcolw3:
                    sel_month_hm = 'All months'
                    wa_heatmap_custom_range = None
                    if heatmap_granularity == 'Weekly':
                        _wa_month_opts = ['All months','January','February','March','April','May','June',
                                       'July','August','September','October','November','December']
                        sel_month_hm = st.selectbox(
                            "Month (Week 1-4)", _wa_month_opts, key='waste_heatmap_month'
                        )
                    elif heatmap_granularity == 'Custom Date':
                        wa_heatmap_custom_range = st.date_input(
                            "Date Range",
                            value=(display_waste['date'].min().date(), display_waste['date'].max().date()),
                            min_value=display_waste['date'].min().date(), max_value=display_waste['date'].max().date(),
                            key='waste_heatmap_customrange'
                        )
                with hcolw4:
                    years_hm = sorted(display_waste['date'].dt.year.dropna().unique(), reverse=True)
                    sel_year_hm = 'All'
                    if heatmap_granularity not in ('Yearly', 'Custom Date'):
                        sel_year_hm = st.selectbox("Year", ['All'] + [str(y) for y in years_hm], key='waste_heatmap_year')

                hmw = display_waste.copy()
                if sel_cat_hm != 'All':
                    hmw = hmw[hmw['category'] == sel_cat_hm]
                if sel_year_hm != 'All':
                    hmw = hmw[hmw['date'].dt.year == int(sel_year_hm)]

                hmw['day'] = hmw['date'].dt.day_name()
                if heatmap_granularity == 'Custom Date':
                    if isinstance(wa_heatmap_custom_range, tuple) and len(wa_heatmap_custom_range) == 2:
                        hmw = hmw[(hmw['date'].dt.date >= wa_heatmap_custom_range[0]) & (hmw['date'].dt.date <= wa_heatmap_custom_range[1])]
                    hmw['period'] = hmw['date'].dt.strftime('%b %d, %Y')
                    x_label = 'Date'
                elif heatmap_granularity == 'Weekly':
                    if sel_month_hm != 'All months':
                        hmw = hmw[hmw['date'].dt.month_name() == sel_month_hm]
                        hmw['period'] = 'Week ' + (((hmw['date'].dt.day - 1) // 7) + 1).clip(upper=4).astype(str)
                        x_label = f'Week of {sel_month_hm}'
                    else:
                        hmw['period'] = 'Wk ' + hmw['date'].dt.strftime('%V, %Y')
                        x_label = 'Week'
                elif heatmap_granularity == 'Yearly':
                    hmw['period'] = hmw['date'].dt.year.astype(str)
                    x_label = 'Year'
                else:
                    hmw['period'] = hmw['date'].dt.to_period('M').dt.strftime('%b %Y')
                    x_label = 'Month'

                if len(hmw) == 0:
                    st.info("No waste records for that month.")
                else:
                    pivot_hw = hmw.groupby(['day','period'])['total_waste_cost'].sum().unstack(fill_value=0)
                    pivot_hw = pivot_hw.reindex([d for d in day_order_w if d in pivot_hw.index])
                    pivot_hw = pivot_hw[sorted(pivot_hw.columns, key=lambda c: hmw[hmw['period']==c]['date'].min())]
                    fig = px.imshow(pivot_hw,
                                    color_continuous_scale=['#FAF6F1','#E8A33D','#B03A2E'],
                                    aspect='auto',
                                    labels=dict(x=x_label, y='Day', color='Waste Cost (₱)'))
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)')
                    # Force one tick per column — with only a few Yearly/Custom
                    # columns, Plotly's default tick spacing can otherwise
                    # land between categories and label it with a fractional
                    # index (e.g. "2022.5") instead of a real value.
                    fig.update_xaxes(type='category', dtick=1)
                    if heatmap_granularity == 'Weekly' and sel_month_hm == 'All months':
                        fig.update_xaxes(tickangle=-45)
                    elif heatmap_granularity == 'Custom Date':
                        fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    by_day_wa = pivot_hw.sum(axis=1)
                    if len(by_day_wa) > 0:
                        by_day_wa_sorted = by_day_wa.sort_values(ascending=False)
                        by_day_wa_html = full_breakdown_html(
                            list(by_day_wa_sorted.items()), prefix='₱', show_pct=False
                        )
                        chart_insight(
                            f"Waste cost by day — {by_day_wa_html}. "
                            f"<b>{by_day_wa_sorted.index[0]}</b> shows the darkest cells — the highest total "
                            f"waste cost in this view."
                        )

        st.markdown("")

        # ── Download — full, unfiltered data ────────────────────────────────
        csv_out = display_waste.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Waste Data as CSV",
            data=csv_out,
            file_name="waste_data_complete.csv",
            mime="text/csv"
        )

    else:
        st.info("No waste data yet. Go to the Database page to upload your waste log.")

elif page == 'Menu Performance':

    st.title("Menu Performance")
    st.caption("To add or update menu items or sales records, go to the **Database** page — all data uploads now happen there.")

    PERF_COLORS = {
        'Keep':       EARTH['success'],
        'Improve':    EARTH['warning'],
        'Reconsider': EARTH['danger'],
    }

    # ── Compute menu performance LIVE from the DB-first sales_df, instead of
    # depending on the static Feature_Engineering batch file. This uses the
    # same Keep/Improve/Reconsider rule as the original ML pipeline
    # (75th/25th percentile of quantity sold, median profit margin), so it
    # always reflects whatever is currently in the database.
    src = sales_df.copy() if sales_df is not None else pd.DataFrame()
    if len(src) > 0 and {'price', 'cost'}.issubset(src.columns):
        src['price'] = pd.to_numeric(src['price'], errors='coerce')
        src['cost']  = pd.to_numeric(src['cost'], errors='coerce')

        # Some rows carry a real per-item cost (from the POS upload);
        # others were backfilled with a flat 30%-of-price estimate at
        # data-prep time because no real COST value was available for
        # them. Using that same flat estimate for every such row is why
        # margins used to cluster around ~70% almost everywhere. Instead,
        # detect which rows still carry the flat estimate, and replace
        # just those with the average cost RATIO (cost ÷ price) of items
        # in the same category that DO have a real, non-flat cost — a
        # more representative, sales-data-driven estimate than one
        # constant applied across the whole menu. Falls back to the flat
        # ratio only when a category has no real-cost items to learn from.
        _is_flat_est = (src['price'] > 0) & ((src['cost'] - src['price'] * 0.3).abs() <= 0.01)
        if 'category' in src.columns and _is_flat_est.any() and (~_is_flat_est).any():
            _real = src[~_is_flat_est & (src['price'] > 0)]
            _real_ratio = _real['cost'] / _real['price']
            _cat_ratio = _real_ratio.groupby(_real['category']).mean()
            _fallback_ratio = _real_ratio.mean() if len(_real) else 0.3
            _ratio_for_row = src.loc[_is_flat_est, 'category'].map(_cat_ratio).fillna(_fallback_ratio)
            src.loc[_is_flat_est, 'cost'] = src.loc[_is_flat_est, 'price'] * _ratio_for_row

        src['profit_margin'] = np.where(src['price'] > 0, (src['price'] - src['cost']) / src['price'], np.nan)

    item_col = 'item' if 'item' in src.columns else \
               'item_name' if 'item_name' in src.columns else None

    def _mp_apply_filters(base_df, key_prefix):
        """Renders its own Year/Quarter/Month (Month cascades off Quarter)
        + Category filter row and returns the filtered dataframe — kept
        local to each chart so no two visualizations on this page share
        the same filter controls."""
        df_f = render_year_quarter_month_filter(base_df, key_prefix, date_col='date') \
            if 'date' in base_df.columns else base_df.copy()
        if 'category' in df_f.columns:
            cats = sorted(df_f['category'].dropna().unique().tolist())
            sel_cat = st.selectbox("Category", ['All'] + cats, key=f'{key_prefix}_cat')
            if sel_cat != 'All':
                df_f = df_f[df_f['category'] == sel_cat]
        return df_f

    def _mp_build_summary(df_f):
        """Aggregates a (already-filtered) sales slice into one row per item,
        with a live Keep/Improve/Reconsider classification."""
        if not item_col:
            return pd.DataFrame()

        # Normalize item/category spelling first — inconsistent casing or
        # whitespace in the uploaded data otherwise splits one item's volume
        # across multiple rows, and can pick an inconsistent category label.
        df_f, mp_item_disp, mp_cat_disp = add_normalized_keys(
            df_f, item_col=item_col, cat_col='category' if 'category' in df_f.columns else None
        )
        df_f[item_col] = df_f['_item_key'].map(mp_item_disp)
        if mp_cat_disp is not None:
            df_f['category'] = df_f['_cat_key'].map(mp_cat_disp)

        agg_dict = {'quantity': 'sum'} if 'quantity' in df_f.columns else {}
        if 'total' in df_f.columns:         agg_dict['total'] = 'sum'
        if 'profit_margin' in df_f.columns: agg_dict['profit_margin'] = 'mean'
        if 'category' in df_f.columns:      agg_dict['category'] = 'first'

        if agg_dict:
            s = df_f.groupby(item_col).agg(agg_dict).reset_index()
        else:
            s = df_f[item_col].value_counts().reset_index()
            s.columns = [item_col, 'count']

        if {'quantity', 'profit_margin'}.issubset(s.columns) and len(s) > 0:
            qty_75    = s['quantity'].quantile(0.75)
            qty_25    = s['quantity'].quantile(0.25)
            margin_50 = s['profit_margin'].quantile(0.50)

            def _classify_menu(row):
                if row['quantity'] >= qty_75 and row['profit_margin'] >= margin_50:
                    return 'Keep'
                elif row['quantity'] >= qty_25:
                    return 'Improve'
                else:
                    return 'Reconsider'

            s['menu_performance'] = s.apply(_classify_menu, axis=1)
        return s

    # ── Performance Distribution — its own filter ───────────────────────────
    with st.container(border=True):
        st.markdown("#### Performance Distribution")
        st.caption("Which items to Keep, Improve, or Reconsider, filtered by period and category.")
        src_dist = _mp_apply_filters(src, 'mp_dist')
        summary  = _mp_build_summary(src_dist)

    st.markdown("")

    if item_col and len(summary) > 0:

        # ── Sort controls (for the tables below) ────────────────────────
        sort_options = [c for c in ['quantity','total','profit_margin'] if c in summary.columns]
        scol1, scol2 = st.columns(2)
        with scol1:
            sort_by  = st.selectbox("Sort by", sort_options if sort_options else [item_col], key='mp_sort_by')
        with scol2:
            sort_dir = st.selectbox("Order", ['Descending','Ascending'], key='mp_sort_dir')

        ascending = sort_dir == 'Ascending'
        if sort_by in summary.columns:
            summary = summary.sort_values(sort_by, ascending=ascending)

        if 'menu_performance' in summary.columns:
            with st.container(border=True):
                # Maintain order: Keep → Improve → Reconsider
                perf_order = ['Keep', 'Improve', 'Reconsider']
                dist = summary['menu_performance'].value_counts().reindex(perf_order).fillna(0).reset_index()
                dist.columns = ['Performance', 'Count']
                fig = px.bar(dist, x='Performance', y='Count',
                             color='Performance',
                             color_discrete_map=PERF_COLORS,
                             text_auto=True,
                             category_orders={'Performance': perf_order})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, margin=dict(l=0,r=0,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                total_items_mp = int(dist['Count'].sum())
                reconsider_ct = int(dist.loc[dist['Performance']=='Reconsider','Count'].sum())
                reconsider_pct = (reconsider_ct / total_items_mp * 100) if total_items_mp else 0
                perf_breakdown = full_breakdown_html(
                    list(zip(dist['Performance'], dist['Count'])), decimals=0
                )
                chart_insight(
                    f"Out of <b>{total_items_mp}</b> menu items — {perf_breakdown}. "
                    f"<b>{reconsider_ct}</b> item(s) ({reconsider_pct:.1f}%) are flagged as "
                    f"<b>Reconsider</b> — these may need a menu review.",
                    'warn' if reconsider_pct > 20 else 'info'
                )

            st.markdown("")

            # ── Grouped tables by performance label ───────────────────────
            def _style_perf_col(val):
                m = {
                    'Keep':       'background-color:#E7F3E8;color:#2E7D32;font-weight:600',
                    'Improve':    'background-color:#FDF1DE;color:#B85C00;font-weight:600',
                    'Reconsider': 'background-color:#FBEAEA;color:#C62828;font-weight:600',
                }
                return m.get(val, '')

            for status in perf_order:
                group = summary[summary['menu_performance'] == status]
                if len(group) == 0:
                    continue
                color = PERF_COLORS[status]
                st.markdown(
                    f"<h4 style='color:{color}'>{status} — {len(group)} items</h4>",
                    unsafe_allow_html=True
                )
                # Show min 10 records per group
                n_show = max(10, len(group))
                group_display = group.head(n_show).copy()

                fmt_map_mp = {}
                if 'total' in group_display.columns:
                    fmt_map_mp['total'] = '₱{:,.2f}'
                if 'profit_margin' in group_display.columns:
                    fmt_map_mp['profit_margin'] = '{:.1%}'
                if 'quantity' in group_display.columns:
                    fmt_map_mp['quantity'] = '{:,.0f}'

                styled_group = group_display.style
                if 'menu_performance' in group_display.columns:
                    styled_group = style_map(styled_group, _style_perf_col, subset=['menu_performance'])
                if fmt_map_mp:
                    styled_group = styled_group.format(fmt_map_mp)

                st.dataframe(styled_group, use_container_width=True, hide_index=True)

        else:
            # Fallback table if no performance labels yet
            st.markdown("#### Item Summary")
            st.info("Run the ML pipeline to see performance classifications.")
            n_show = st.slider("Records to show", 10, max(11, len(summary)), min(10, len(summary)), 10, key='mp_fallback_n')
            st.dataframe(summary.head(n_show), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No items to display yet. This summary is calculated from **sales transactions** — "
            "go to the Database page to add your sales log."
        )

    st.markdown("")

    # ── Profitability by Item (Top 20) — its own, separate filter ──────────
    with st.container(border=True):
        st.markdown("#### Profitability by Item (Top 20)")
        st.caption("Top revenue-generating items, filtered by period and category independently of the chart above.")
        src_profit = _mp_apply_filters(src, 'mp_profit')
        summary_profit = _mp_build_summary(src_profit)

        if item_col and 'total' in summary_profit.columns and len(summary_profit) > 0:
            summary_profit['total'] = pd.to_numeric(summary_profit['total'], errors='coerce')
            top20 = summary_profit.dropna(subset=['total']).nlargest(20, 'total')
            fig = px.bar(top20, x='total', y=item_col, orientation='h',
                         color='total',
                         color_continuous_scale=['#C4A882','#6F4E37'],
                         text_auto='.2s',
                         labels={'total':'Revenue (₱)', item_col:'Item'})
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            top20_sorted = top20.sort_values('total', ascending=False)
            top5_profit = top20_sorted.head(5)
            top5_profit_html = full_breakdown_html(
                list(zip(top5_profit[item_col], top5_profit['total'])), prefix='₱', show_pct=False
            )
            chart_insight(
                f"Top 5 by revenue — {top5_profit_html}. Together, all {len(top20)} items shown here "
                f"total <b>₱{top20['total'].sum():,.2f}</b> in this filtered view."
            )
        else:
            st.info("No revenue data for this filter combination yet.")

    # ── Spoilage Alert Level by Category (moved here from Forecast & Predictions) ──
    st.markdown("")
    with st.container(border=True):
        st.markdown("### Spoilage Alert Level by Category")
        st.caption("Today's spoilage risk per category, from the trained spoilage model.")

        spoil_bundle = models.get('spoilage')
        waste_for_spoilage = waste_df.copy() if (waste_df is not None and len(waste_df) > 0) else None

        if spoil_bundle is None or waste_for_spoilage is None or 'category' not in waste_for_spoilage.columns or len(waste_for_spoilage) == 0:
            st.info("Run the prediction models (and upload waste data) to see spoilage alerts here.")
        else:
            spoil_cats = sorted(waste_for_spoilage['category'].dropna().unique().tolist())
            sel_spoil_cat = st.selectbox("Category", ['All'] + spoil_cats, key='mp_spoil_cat')
            if sel_spoil_cat != 'All':
                waste_for_spoilage = waste_for_spoilage[waste_for_spoilage['category'] == sel_spoil_cat]

            _today          = pd.Timestamp.now()
            _today_dow      = _today.dayofweek
            _today_month    = _today.month
            _today_quarter  = (_today.month - 1) // 3 + 1
            _today_week     = _today.isocalendar()[1]
            _today_is_wknd  = int(_today_dow >= 5)
            _today_dry      = int(_today.month in [12,1,2,3,4,5])

            sf = waste_for_spoilage
            sf['hour']          = 12
            sf['day_of_week']   = _today_dow
            sf['month']         = _today_month
            sf['quarter']       = _today_quarter
            sf['week_number']   = _today_week
            sf['is_weekend']    = _today_is_wknd
            sf['is_peak_hour']  = 0
            sf['is_dry_season'] = _today_dry
            if 'item_name' in sf.columns:
                sf['item_encoded'] = pd.factorize(sf['item_name'])[0]
            if 'cost_per_item' in sf.columns:
                sf['price']              = pd.to_numeric(sf['cost_per_item'], errors='coerce').fillna(0) / 0.3
                sf['cost']               = pd.to_numeric(sf['cost_per_item'], errors='coerce').fillna(0)
                sf['profit_margin']      = 0.7
                sf['gross_profit']       = sf['price'] * 0.7
                sf['total_cost']         = sf['cost']
                sf['revenue']            = sf['price']
                sf['transaction_profit'] = sf['price'] * 0.7
            if 'quantity_wasted' in sf.columns:
                sf['quantity']            = pd.to_numeric(sf['quantity_wasted'], errors='coerce').fillna(1)
                sf['estimated_waste_qty'] = sf['quantity']
                sf['is_high_waste_item']  = (sf['quantity'] > 2).astype(int)
            if 'total_waste_cost' in sf.columns:
                sf['estimated_waste_cost'] = pd.to_numeric(sf['total_waste_cost'], errors='coerce').fillna(0)
            sf['rolling_7d_avg']  = sf.get('quantity', pd.Series([1]*len(sf)))
            sf['rolling_30d_avg'] = sf.get('quantity', pd.Series([1]*len(sf)))
            sf['lag_1d']          = sf.get('quantity', pd.Series([1]*len(sf)))
            sf['demand_trend']    = 0.0
            _waste_rate_map = {
                'Coffee Based': 0.05, 'Signature Kôfē': 0.06,
                'Kôfē Frappé':  0.08, 'Sans Coffee':    0.07, 'Mini Bites': 0.15,
            }
            sf['waste_rate'] = sf['category'].map(_waste_rate_map).fillna(0.10)
            for cat in ['Coffee Based','Kôfē Frappé','Mini Bites','Sans Coffee','Signature Kôfē']:
                sf[f'cat_{cat}'] = (sf['category'] == cat).astype(int)

            spoil_model    = spoil_bundle['model']
            spoil_features = [f for f in spoil_bundle['features'] if f in sf.columns]
            if not spoil_features:
                st.info("Spoilage model features unavailable — retrain the model to see this chart.")
            else:
                X_spoil = sf[spoil_features].fillna(0)
                spoil_preds = spoil_model.predict(X_spoil)
                spoil_label_map = {0:'Low', 1:'Medium', 2:'High'}
                sf['Alert Level'] = [spoil_label_map.get(int(p), str(p)) for p in spoil_preds]
                ALERT_COLORS2 = {'Low': EARTH['success'], 'Medium': EARTH['warning'], 'High': EARTH['danger']}
                alert_levels_order = ['Low', 'Medium', 'High']

                # Reindex to every (category, Alert Level) combination —
                # even ones with zero records — so Low/Medium/High all
                # show in the legend with their color, instead of Plotly
                # dropping a level from the legend entirely just because
                # no row happens to have it right now.
                all_cats_sp = sorted(sf['category'].dropna().unique().tolist())
                full_idx = pd.MultiIndex.from_product(
                    [all_cats_sp, alert_levels_order], names=['category', 'Alert Level']
                )
                alert_cat = (
                    sf.groupby(['category', 'Alert Level']).size()
                    .reindex(full_idx, fill_value=0)
                    .reset_index(name='Count')
                )
                fig = px.bar(
                    alert_cat, x='category', y='Count',
                    color='Alert Level',
                    color_discrete_map=ALERT_COLORS2,
                    barmode='stack',
                    text_auto=True,
                    category_orders={'Alert Level': alert_levels_order}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title='Category', yaxis_title='Number of Records',
                    legend=dict(orientation='h', y=1.1),
                    margin=dict(l=0,r=0,t=30,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Per-category breakdown — one line per category showing
                # its own Low/Medium/High split, instead of one combined
                # "overall" figure that hides how each category differs.
                per_cat_lines = []
                for cat in all_cats_sp:
                    cat_counts = (
                        alert_cat[alert_cat['category'] == cat]
                        .set_index('Alert Level')['Count']
                        .reindex(alert_levels_order).fillna(0)
                    )
                    cat_html = full_breakdown_html(list(zip(cat_counts.index, cat_counts.values)), decimals=0)
                    per_cat_lines.append(f"<b>{cat}</b> — {cat_html}")
                per_cat_html = "<br/>".join(per_cat_lines)

                high_by_cat = sf[sf['Alert Level'] == 'High'].groupby('category').size()
                if len(high_by_cat) > 0:
                    top_alert_cat   = high_by_cat.idxmax()
                    top_alert_count = int(high_by_cat.max())
                    st.markdown(f"""
                    <div style='background:#FFF8F3;border-left:4px solid #6F4E37;
                                padding:16px 20px;border-radius:8px;margin-top:8px'>
                        <b>Alert Level by category:</b><br/>{per_cat_html}<br/><br/>
                        <b>{top_alert_cat}</b> has the most High Alert items (<b>{top_alert_count} records</b>).
                        Immediately review ingredient freshness and storage for this category.
                        Consider adjusting order frequency to reduce spoilage risk.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:#F0EDE2;border-left:4px solid #6B8E4E;
                                padding:16px 20px;border-radius:8px;margin-top:8px'>
                        <b>Alert Level by category:</b><br/>{per_cat_html}<br/><br/>
                        <b>Good news!</b> No High Alert items detected.
                        Current ingredient management is working well.
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 5: INVENTORY STATUS — with Alert Level + Out of Stock
# ============================================================================

elif page == 'Inventory Status':

    st.title("Inventory Status")
    st.caption("To add new inventory records, go to the **Database** page — all data uploads now happen there.")

    # Alert level colors (renamed from Risk)
    ALERT_COLORS = {
        'High Alert':   EARTH['danger'],
        'Medium Alert': EARTH['warning'],
        'Low Alert':    EARTH['success'],
        'Expired':      '#4E342E',
        'Out of Stock': '#8D6E63',   # muted mocha — distinct from other levels
    }

    if inventory_df is not None and len(inventory_df) > 0:
        inv_hist = inventory_df.copy()

        # ── Rename "Risk" → "Alert Level" ────────────────────────────────
        if 'spoilage_risk' in inv_hist.columns:
            inv_hist['alert_level'] = inv_hist['spoilage_risk'].replace({
                'High Risk':   'High Alert',
                'Medium Risk': 'Medium Alert',
                'Low Risk':    'Low Alert',
                'Expired':     'Expired',
            })
        elif 'alert_level' not in inv_hist.columns:
            inv_hist['alert_level'] = 'Low Alert'

        # ── Mark Out of Stock — quantity = 0 ─────────────────────────────
        if 'quantity' in inv_hist.columns:
            inv_hist.loc[inv_hist['quantity'] <= 0, 'alert_level'] = 'Out of Stock'

        if 'purchase_date' in inv_hist.columns:
            inv_hist['purchase_date'] = pd.to_datetime(inv_hist['purchase_date'], errors='coerce')

        # ── "Current" snapshot — most recent 30 days of purchases ─────────
        # (inventory now holds 3 years of history for trend purposes; KPIs /
        # alert levels below should reflect what's actually on hand today)
        if 'purchase_date' in inv_hist.columns:
            cutoff = inv_hist['purchase_date'].max() - pd.Timedelta(days=30)
            inv = inv_hist[inv_hist['purchase_date'] >= cutoff].copy()
        else:
            inv = inv_hist.copy()

        # ── Summary badges ────────────────────────────────────────────────
        st.caption("Snapshot of the most recent 30 days of purchases. Full 3-year history is available in the 'Inventory Value Over Time' chart below.")
        alert_order = ['High Alert','Medium Alert','Low Alert','Expired','Out of Stock']
        alert_counts = inv['alert_level'].value_counts()

        cols = st.columns(len(alert_order))
        for i, level in enumerate(alert_order):
            count = alert_counts.get(level, 0)
            color = ALERT_COLORS.get(level, '#333')
            with cols[i]:
                st.markdown(
                    f"<div style='background:{color};color:white;padding:14px;"
                    f"border-radius:10px;text-align:center;'>"
                    f"<b style='font-size:13px'>{level}</b><br>"
                    f"<span style='font-size:28px;font-weight:bold'>{count}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("")
        st.markdown("---")

        # ── Alert level filter — now actually applied below (previously the
        # filtered result was computed but never used by any table) ───────
        filter_level = st.selectbox(
            "Filter by Alert Level",
            ['All'] + alert_order,
            key='inv_filter_level'
        )
        inv_filtered = inv if filter_level == 'All' else inv[inv['alert_level'] == filter_level]

        # ── Inventory Records table — the main filterable, row-level view ──
        with st.container(border=True):
            st.markdown("#### Inventory Records")
            st.caption(
                "Purchase Date and Expiry Date are shown side by side. Expired items are marked "
                "**Discard** in the Action column; Out of Stock items are marked **Restock**."
            )
            if len(inv_filtered) == 0:
                st.info("No inventory records match this filter.")
            else:
                inv_table = inv_filtered.copy()

                # Discard/Restock action column, derived from Alert Level
                def _inv_action(level):
                    if level == 'Expired':
                        return 'Discard'
                    if level == 'Out of Stock':
                        return 'Restock'
                    return '—'
                inv_table['Action'] = inv_table['alert_level'].apply(_inv_action)

                # Clean date columns — strip the 00:00:00 time component that
                # a plain Timestamp/string otherwise renders in a dataframe.
                if 'purchase_date' in inv_table.columns:
                    inv_table['Purchase Date'] = pd.to_datetime(inv_table['purchase_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                if 'expiration_date' in inv_table.columns:
                    inv_table['Expiry Date'] = pd.to_datetime(
                        inv_table['expiration_date'].astype(str).str.split(' ').str[0], errors='coerce'
                    ).dt.strftime('%Y-%m-%d')

                rename_map = {
                    'ingredient': 'Ingredient', 'quantity': 'Quantity', 'unit': 'Unit',
                    'cost_per_unit': 'Cost per Unit', 'total_cost': 'Total Cost',
                    'alert_level': 'Alert Level',
                }
                inv_table = inv_table.rename(columns=rename_map)

                # Sort most urgent first by default: High Alert / Expired /
                # Out of Stock rows surface at the top even when the filter
                # is left on 'All', with soonest-to-expire first within a level.
                _urgency_rank = {'High Alert': 0, 'Expired': 1, 'Out of Stock': 2, 'Medium Alert': 3, 'Low Alert': 4}
                inv_table['_urgency'] = inv_table['Alert Level'].map(_urgency_rank).fillna(9)
                sort_cols = ['_urgency'] + (['expiration_date'] if 'expiration_date' in inv_table.columns else [])
                inv_table = inv_table.sort_values(sort_cols).drop(columns=['_urgency'])

                # Column order: Purchase Date + Expiry Date side by side;
                # inventory_key, shelf_life_days, days_until_expiration are
                # dropped entirely (internal/derived fields, not useful here).
                display_cols = [c for c in [
                    'Ingredient', 'Purchase Date', 'Expiry Date', 'Quantity', 'Unit',
                    'Cost per Unit', 'Total Cost', 'Alert Level', 'Action'
                ] if c in inv_table.columns]
                inv_table = inv_table[display_cols]

                # Color-code the Alert Level column the same way the Menu
                # Performance page colors its Keep/Improve/Reconsider column.
                def _style_alert_col(val):
                    m = {
                        'High Alert':   'background-color:#FBEAEA;color:#C62828;font-weight:600',
                        'Medium Alert': 'background-color:#FDF1DE;color:#B85C00;font-weight:600',
                        'Low Alert':    'background-color:#E7F3E8;color:#2E7D32;font-weight:600',
                        'Expired':      'background-color:#E8DCD8;color:#4E342E;font-weight:600',
                        'Out of Stock': 'background-color:#EFE6E0;color:#6D4C41;font-weight:600',
                    }
                    return m.get(val, '')

                def _style_action_col(val):
                    if val == 'Discard':
                        return 'background-color:#FBEAEA;color:#C62828;font-weight:600'
                    if val == 'Restock':
                        return 'background-color:#FDF1DE;color:#B85C00;font-weight:600'
                    return ''

                fmt_map_inv = {}
                if 'Cost per Unit' in inv_table.columns: fmt_map_inv['Cost per Unit'] = '₱{:,.2f}'
                if 'Total Cost' in inv_table.columns:    fmt_map_inv['Total Cost']    = '₱{:,.2f}'
                if 'Quantity' in inv_table.columns:      fmt_map_inv['Quantity']      = '{:,.1f}'

                styled_inv = inv_table.style
                if 'Alert Level' in inv_table.columns:
                    styled_inv = style_map(styled_inv, _style_alert_col, subset=['Alert Level'])
                if 'Action' in inv_table.columns:
                    styled_inv = style_map(styled_inv, _style_action_col, subset=['Action'])
                if fmt_map_inv:
                    styled_inv = styled_inv.format(fmt_map_inv)

                st.dataframe(styled_inv, use_container_width=True, hide_index=True)

        st.markdown("")

        # ── Inventory Guide — what to restock this week/day ────────────────
        with st.container(border=True):
            st.markdown("#### Inventory Guide")
            st.caption("What to restock now, what to restock this week, and what's well stocked — based on each ingredient's latest purchase record.")

            if 'ingredient' in inv_hist.columns and 'purchase_date' in inv_hist.columns:
                latest_per_ing = (
                    inv_hist.dropna(subset=['purchase_date'])
                    .sort_values('purchase_date')
                    .drop_duplicates(subset='ingredient', keep='last')
                    .copy()
                )

                restock_now  = latest_per_ing[latest_per_ing['alert_level'].isin(['Out of Stock', 'Expired'])]
                restock_soon = latest_per_ing[latest_per_ing['alert_level'].isin(['High Alert', 'Medium Alert'])]
                well_stocked = latest_per_ing[latest_per_ing['alert_level'] == 'Low Alert']

                def _guide_col(df_g, color, empty_msg, show_reason=False):
                    if len(df_g) == 0:
                        st.caption(empty_msg)
                        return
                    rows = df_g.sort_values('quantity') if 'quantity' in df_g.columns else df_g
                    for _, r in rows.iterrows():
                        qty_txt = ""
                        if 'quantity' in df_g.columns and pd.notna(r['quantity']):
                            unit_txt = str(r['unit']) if 'unit' in df_g.columns and pd.notna(r.get('unit')) else ''
                            qty_txt = f" — {r['quantity']:.0f} {unit_txt} left".rstrip()
                        reason_txt = ""
                        if show_reason and 'alert_level' in df_g.columns and pd.notna(r.get('alert_level')):
                            reason_txt = f" <span style='color:#8A7460;font-size:11.5px'>({r['alert_level']})</span>"
                        st.markdown(
                            f"<div style='padding:7px 12px;border-left:3px solid {color};"
                            f"margin-bottom:4px;font-size:13.5px;color:#3E2723'>"
                            f"{r['ingredient']}{qty_txt}{reason_txt}</div>",
                            unsafe_allow_html=True
                        )

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown(
                        "<div style='background:#C62828;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>RESTOCK NOW</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col(restock_now, '#C62828', "Nothing out of stock or expired right now.", show_reason=True)
                with g2:
                    st.markdown(
                        "<div style='background:#B85C00;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>RESTOCK THIS WEEK</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col(restock_soon, '#B85C00', "Nothing needs restocking this week.")
                with g3:
                    st.markdown(
                        "<div style='background:#2E7D32;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>WELL STOCKED</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col(well_stocked, '#2E7D32', "No ingredients marked as well stocked yet.")

                st.markdown("")
                chart_insight(
                    f"<b>{len(restock_now)}</b> ingredient(s) need restocking right now, "
                    f"<b>{len(restock_soon)}</b> should be reordered this week, and "
                    f"<b>{len(well_stocked)}</b> are currently well stocked. "
                    "To add a brand-new ingredient that isn't tracked yet, upload it on the Database page — "
                    "it will show up here once it has a purchase record."
                )
            else:
                st.info("Not enough inventory history to build a restock guide yet.")

        st.markdown("")

        # ── Bar chart — inventory by alert level ──────────────────────────
        with st.container(border=True):
            st.markdown("#### Inventory Alert Level Distribution")
            dist = inv['alert_level'].value_counts().reindex(alert_order).fillna(0).reset_index()
            dist.columns = ['Alert Level', 'Count']
            fig = px.bar(dist, x='Alert Level', y='Count',
                         color='Alert Level',
                         color_discrete_map=ALERT_COLORS,
                         text_auto=True,
                         category_orders={'Alert Level': alert_order})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False, margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            high_ct = int(dist.loc[dist['Alert Level']=='High Alert','Count'].sum())
            alert_dist_sorted = dist[dist['Count'] > 0].sort_values('Count', ascending=False)
            alert_dist_html = full_breakdown_html(
                list(zip(alert_dist_sorted['Alert Level'], alert_dist_sorted['Count'])), decimals=0
            )
            if high_ct > 0:
                chart_insight(f"Inventory by alert level — {alert_dist_html}. "
                              f"<b>{high_ct}</b> item(s) are at High Alert and need urgent attention.", 'warn')
            else:
                chart_insight(f"Inventory by alert level — {alert_dist_html}. "
                              f"No items are currently at High Alert. Current ingredient management is working well.", 'good')

        st.markdown("")

        # ── Additional graphs ────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("#### Inventory Value by Ingredient (Top 10)")
            if 'ingredient' in inv.columns and 'total_cost' in inv.columns:
                val_by_ing = (
                    inv.groupby('ingredient')['total_cost']
                    .sum().nlargest(10).reset_index()
                )
                val_by_ing.columns = ['Ingredient', 'Value']
                fig2 = px.bar(val_by_ing, x='Value', y='Ingredient', orientation='h',
                              color='Value', color_continuous_scale=['#C4A882', '#6F4E37'],
                              text_auto=',.0f')
                fig2.update_traces(texttemplate='₱%{x:,.0f}', textposition='outside')
                fig2.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False, xaxis_title='Inventory Value (₱)',
                    margin=dict(l=0, r=80, t=10, b=0)
                )
                st.plotly_chart(fig2, use_container_width=True)

                val_by_ing_3 = val_by_ing.sort_values('Value', ascending=False).head(3)
                val_by_ing_html = full_breakdown_html(
                    list(zip(val_by_ing_3['Ingredient'], val_by_ing_3['Value'])), prefix='₱', show_pct=False
                )
                chart_insight(
                    f"Top 3 by inventory value — {val_by_ing_html}. Together, all {len(val_by_ing)} "
                    f"ingredients shown tie up <b>₱{val_by_ing['Value'].sum():,.2f}</b> in inventory."
                )

        st.markdown("")

        with st.container(border=True):
            st.markdown("#### Quantity vs. Days Until Expiration")
            if 'days_until_expiration' in inv.columns and 'quantity' in inv.columns:
                fig3 = px.scatter(
                    inv, x='days_until_expiration', y='quantity',
                    color='alert_level',
                    color_discrete_map=ALERT_COLORS,
                    hover_data=[c for c in ['ingredient'] if c in inv.columns],
                    labels={'days_until_expiration': 'Days Until Expiration', 'quantity': 'Quantity'}
                )
                fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', y=1.15),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig3, use_container_width=True)

                near_exp = inv[inv['days_until_expiration'] <= 2]
                batch_by_alert = inv['alert_level'].value_counts()
                batch_by_alert_html = full_breakdown_html(
                    list(batch_by_alert.sort_values(ascending=False).items()), decimals=0
                )
                if len(near_exp) > 0:
                    chart_insight(
                        f"Batches by alert level — {batch_by_alert_html}. "
                        f"<b>{len(near_exp)}</b> batch(es) have 2 days or less until expiration — "
                        f"the points furthest to the left need immediate attention.",
                        'warn'
                    )
                else:
                    chart_insight(
                        f"Batches by alert level — {batch_by_alert_html}. "
                        f"No batches are within 2 days of expiring right now.", 'good'
                    )

        st.markdown("")

        # ── Inventory value trend over time — full history, filterable ────
        with st.container(border=True):
            st.markdown("#### Inventory Value Over Time")
            st.caption("Uses the full purchase history (up to 3 years), independent of the 30-day snapshot above.")
            if 'purchase_date' in inv_hist.columns and 'total_cost' in inv_hist.columns:
                if 'ingredient' in inv_hist.columns:
                    ings = sorted(inv_hist['ingredient'].dropna().unique().tolist())
                    sel_ing = st.selectbox("Ingredient", ['All'] + ings, key='inv_value_ingredient')
                else:
                    sel_ing = 'All'

                inv_time = inv_hist.copy()
                if sel_ing != 'All':
                    inv_time = inv_time[inv_time['ingredient'] == sel_ing]

                inv_time, inv_value_gran = render_daily_weekly_yearly_filter(
                    inv_time, 'inv_value', date_col='purchase_date'
                )

                if inv_value_gran == 'Daily':
                    inv_time['period'] = inv_time['purchase_date'].dt.date
                elif inv_value_gran == 'Yearly':
                    inv_time['period'] = inv_time['purchase_date'].dt.year
                else:
                    inv_time['period'] = inv_time['purchase_date'].dt.to_period('W').dt.start_time

                trend = inv_time.groupby('period')['total_cost'].sum().reset_index()
                trend.columns = ['Date', 'Value']
                trend = trend.sort_values('Date')
                fig4 = px.line(trend, x='Date', y='Value', markers=True,
                                color_discrete_sequence=[EARTH['accent']])
                fig4.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickprefix='₱', tickformat=',.0f'),
                    xaxis_title=inv_value_gran,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig4, use_container_width=True)

                if len(trend) == 0:
                    st.info("No inventory records for this selection.")
                elif len(trend) >= 2:
                    val_dir = "increasing" if trend['Value'].iloc[-1] > trend['Value'].iloc[0] else "decreasing"
                    highest_row = trend.loc[trend['Value'].idxmax()]
                    lowest_row  = trend.loc[trend['Value'].idxmin()]
                    chart_insight(
                        f"Inventory value is <b>{val_dir}</b> over this {inv_value_gran.lower()} view. "
                        f"It peaked at <b>₱{highest_row['Value']:,.2f}</b> "
                        f"({pd.to_datetime(highest_row['Date']).strftime('%Y-%m-%d')}) and was "
                        f"lowest at <b>₱{lowest_row['Value']:,.2f}</b> "
                        f"({pd.to_datetime(lowest_row['Date']).strftime('%Y-%m-%d')}). "
                        f"Latest value: <b>₱{trend['Value'].iloc[-1]:,.2f}</b> "
                        f"(average across all periods shown: ₱{trend['Value'].mean():,.2f})."
                    )
                else:
                    chart_insight(f"Only one {inv_value_gran.lower()} period in this selection: "
                                   f"<b>₱{trend['Value'].iloc[0]:,.2f}</b>.")

        st.markdown("")

        # ── Download inventory report — same cleaned columns as the table ──
        inv_export = inv.copy()
        inv_export['Action'] = inv_export['alert_level'].apply(
            lambda lvl: 'Discard' if lvl == 'Expired' else ('Restock' if lvl == 'Out of Stock' else '—')
        )
        if 'purchase_date' in inv_export.columns:
            inv_export['Purchase Date'] = pd.to_datetime(inv_export['purchase_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'expiration_date' in inv_export.columns:
            inv_export['Expiry Date'] = pd.to_datetime(
                inv_export['expiration_date'].astype(str).str.split(' ').str[0], errors='coerce'
            ).dt.strftime('%Y-%m-%d')
        inv_export = inv_export.rename(columns={
            'ingredient': 'Ingredient', 'quantity': 'Quantity', 'unit': 'Unit',
            'cost_per_unit': 'Cost per Unit', 'total_cost': 'Total Cost', 'alert_level': 'Alert Level',
        })
        export_cols = [c for c in [
            'Ingredient', 'Purchase Date', 'Expiry Date', 'Quantity', 'Unit',
            'Cost per Unit', 'Total Cost', 'Alert Level', 'Action'
        ] if c in inv_export.columns]
        csv_inv = inv_export[export_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Inventory Report",
            data=csv_inv,
            file_name="inventory_report.csv",
            mime="text/csv"
        )

    else:
        st.info("No inventory data yet. Go to the Database page to upload your inventory records.")

# ============================================================================
# PAGE 6: FORECAST & PREDICTIONS
# Waste/menu/spoilage predictions (RF models) + adjustable waste trend forecast
# ============================================================================

elif page == 'Forecast & Predictions':

    # ── Page header ───────────────────────────────────────────────────────
    hero_banner(
        "DineData Intelligence",
        "Forecast & Predictions",
        "Waste, menu, and spoilage predictions — plus an adjustable demand, sales, and waste trend forecast"
    )

    # ============================================================================
    # SECTION 1: WASTE, MENU & SPOILAGE PREDICTIONS — Random Forest models
    # ============================================================================
    st.markdown("## Waste, Menu & Spoilage Predictions")
    st.caption("Runs your waste data through 4 trained models — each one chosen as the best of 3 algorithms compared during training.")

    if not metrics:
        st.warning("Run `python ML_Models/ml_models.py` to train models first.")

    # ── Auto-load existing waste data ─────────────────────────────────────
    # No upload needed here — data comes from the Database page uploader
    # Predictions auto-refresh daily based on today's date
    WASTE_FILE = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_waste_data.csv')

    # Get today's date info for daily-aware predictions
    today           = pd.Timestamp.now()
    today_dow       = today.dayofweek        # 0=Mon, 6=Sun
    today_day_name  = today.day_name()       # e.g. "Thursday"
    today_month     = today.month
    today_quarter   = (today.month - 1) // 3 + 1
    today_week      = today.isocalendar()[1]
    today_is_wknd   = int(today_dow >= 5)
    today_dry       = int(today.month in [12,1,2,3,4,5])

    # Show today's context to the user
    st.markdown(
        f"<div style='background:#FAF6F1;border-left:4px solid #6F4E37;"
        f"padding:10px 16px;border-radius:6px;margin-bottom:16px'>"
        f"<b>Today is {today_day_name}, {today.strftime('%B %d, %Y')}</b> — "
        f"predictions below are calibrated for today's day-of-week pattern."
        f"</div>",
        unsafe_allow_html=True
    )

    try:
        _wi = waste_df if db_available() else load_csv(WASTE_FILE)
        waste_input = _wi.copy() if _wi is not None and len(_wi) > 0 else None
    except Exception:
        waste_input = None

    if waste_input is None or len(waste_input) == 0:
        st.info("No waste data found. Go to the Database page to upload your waste records first.")
        st.stop()

    # ── Run all 4 RF models on waste data ─────────────────────────────────
    def build_features_from_waste(df):
        """
        Convert waste CSV into RF model features.
        Key: today's actual date features override historical dates
        so predictions are always calibrated to the current day of week,
        month, and season — giving fresh daily recommendations even
        without a new upload.
        """
        results = df.copy()

        # Override time features with TODAY's values
        # This makes predictions day-aware without requiring a new upload
        results['hour']          = 12                    # assume midday for today
        results['day_of_week']   = today_dow             # today's actual day
        results['month']         = today_month           # today's actual month
        results['quarter']       = today_quarter          # today's actual quarter
        results['week_number']   = today_week             # today's actual week
        results['is_weekend']    = today_is_wknd          # is today a weekend?
        results['is_peak_hour']  = 0                      # default: not peak hour
        results['is_dry_season'] = today_dry              # today's season
        if 'item_name' in results.columns:
            results['item_encoded'] = pd.factorize(results['item_name'])[0]
        if 'cost_per_item' in results.columns:
            results['price']             = pd.to_numeric(results['cost_per_item'], errors='coerce').fillna(0) / 0.3
            results['cost']              = pd.to_numeric(results['cost_per_item'], errors='coerce').fillna(0)
            results['profit_margin']     = 0.7
            results['gross_profit']      = results['price'] * 0.7
            results['total_cost']        = results['cost']
            results['revenue']           = results['price']
            results['transaction_profit']= results['price'] * 0.7
        if 'quantity_wasted' in results.columns:
            results['quantity']             = pd.to_numeric(results['quantity_wasted'], errors='coerce').fillna(1)
            results['estimated_waste_qty']  = results['quantity']
            results['is_high_waste_item']   = (results['quantity'] > 2).astype(int)
        if 'total_waste_cost' in results.columns:
            results['estimated_waste_cost'] = pd.to_numeric(results['total_waste_cost'], errors='coerce').fillna(0)
        results['rolling_7d_avg']  = results.get('quantity', pd.Series([1]*len(results)))
        results['rolling_30d_avg'] = results.get('quantity', pd.Series([1]*len(results)))
        results['lag_1d']          = results.get('quantity', pd.Series([1]*len(results)))
        results['demand_trend']    = 0.0
        waste_rate_map = {
            'Coffee Based': 0.05, 'Signature Kôfē': 0.06,
            'Kôfē Frappé':  0.08, 'Sans Coffee':    0.07, 'Mini Bites': 0.15,
        }
        if 'category' in results.columns:
            results['waste_rate'] = results['category'].map(waste_rate_map).fillna(0.10)
        else:
            results['waste_rate'] = 0.10
        for cat in ['cat_Coffee Based','cat_Kôfē Frappé','cat_Mini Bites',
                    'cat_Sans Coffee','cat_Signature Kôfē']:
            results[cat] = 0
        if 'category' in results.columns:
            for cat in ['Coffee Based','Kôfē Frappé','Mini Bites','Sans Coffee','Signature Kôfē']:
                results[f'cat_{cat}'] = (results['category'] == cat).astype(int)
        return results

    with st.status(f"Preparing {today_day_name} predictions...", expanded=False) as coffee_status:
        coffee_status.update(label=f"Grinding today's data ({today_day_name})...")
        feat_df = build_features_from_waste(waste_input)
        PERF_COLORS   = {'Keep': EARTH['success'], 'Improve': EARTH['warning'], 'Reconsider': EARTH['danger']}
        ALERT_COLORS2 = {'Low':  EARTH['success'], 'Medium':  EARTH['warning'], 'High':       EARTH['danger']}

        stage_labels = {
            'demand':   "Brewing demand predictions...",
            'waste':    "Brewing waste predictions...",
            'menu':     "Steaming menu performance...",
            'spoilage': "Stirring in spoilage alerts...",
        }

        for model_key, col_name, label_map in [
            ('demand',   'Predicted Demand (units)', None),
            ('waste',    'Predicted Waste (units)',  None),
            ('menu',     'Menu Performance',  {0:'Reconsider', 1:'Improve', 2:'Keep'}),
            ('spoilage', 'Alert Level',       {0:'Low', 1:'Medium', 2:'High'}),
        ]:
            coffee_status.update(label=stage_labels.get(model_key, "Brewing..."))
            bundle = models.get(model_key)
            if bundle is None:
                waste_input[col_name] = 'Model not trained'
                continue
            model    = bundle['model']
            features = [f for f in bundle['features'] if f in feat_df.columns]
            if not features:
                waste_input[col_name] = 'Features missing'
                continue
            X     = feat_df[features].fillna(0)
            preds = model.predict(X)
            if label_map:
                waste_input[col_name] = [label_map.get(int(p), str(p)) for p in preds]
            else:
                waste_input[col_name] = np.round(preds, 2)

        coffee_status.update(label="Pouring your results...")
        coffee_status.update(label="Predictions ready!", state="complete", expanded=False)

    # ── KPI Cards — show today's day context ────────────────────────────
    st.markdown(f"### {today_day_name} Prediction Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        total_waste = int(waste_input['quantity_wasted'].sum()) if 'quantity_wasted' in waste_input.columns else 0
        stat_card("Total Units Wasted", f"{total_waste:,}")
    with k2:
        total_cost = waste_input['total_waste_cost'].sum() if 'total_waste_cost' in waste_input.columns else 0
        stat_card("Total Waste Cost", f"₱{total_cost:,.2f}")
    with k3:
        to_reconsider = (waste_input['Menu Performance'] == 'Reconsider').sum() if 'Menu Performance' in waste_input.columns else 0
        stat_card("Items to Reconsider", int(to_reconsider))
    with k4:
        high_alerts = (waste_input['Alert Level'] == 'High').sum() if 'Alert Level' in waste_input.columns else 0
        stat_card("High Alert Items", int(high_alerts))

    st.markdown("")

    # ── Production Guide ──────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(f"### {today_day_name} Production Guide")
        st.caption(
            f"Based on {today_day_name}'s historical demand patterns — "
            "guide your kitchen staff on what to prioritize today."
        )

        if 'item_name' in waste_input.columns and 'Predicted Demand (units)' in waste_input.columns and 'category' in waste_input.columns:

            # ── Most Expected Today — top item per category ───────────────
            # Based on actual sales history, not the waste-log-derived demand
            # prediction (which under-counts, since it's modeled from small
            # per-incident waste quantities).
            st.markdown("#### Most Expected Today")
            st.caption("The item with the highest typical daily sales volume per category — quantity is set to cover a busier-than-average day, so you don't run out.")

            item_col_pg = None
            if sales_df is not None and len(sales_df) > 0 and 'category' in sales_df.columns:
                item_col_pg = 'item' if 'item' in sales_df.columns else ('item_name' if 'item_name' in sales_df.columns else None)

            if item_col_pg and 'quantity' in sales_df.columns:
                sales_recent = sales_df.copy()
                sales_recent['date'] = pd.to_datetime(sales_recent['date'], errors='coerce')
                sales_recent = sales_recent.dropna(subset=['date'])

                # Normalize item names before grouping — inconsistent casing/
                # spacing in the uploaded data (e.g. "Caramel Macchiato (iced)"
                # vs "caramel macchiato (iced)") was silently splitting the
                # same item's volume across multiple rows, making every
                # variant look far less popular than it really is.
                sales_recent['_cat_key']  = sales_recent['category'].astype(str).str.strip().str.lower()
                sales_recent['_item_key'] = sales_recent[item_col_pg].astype(str).str.strip().str.lower()
                # Use the most frequent original-case spelling as the display name
                display_names = (
                    sales_recent.groupby('_item_key')[item_col_pg]
                    .agg(lambda s: s.value_counts().idxmax())
                )
                display_cats = (
                    sales_recent.groupby('_cat_key')['category']
                    .agg(lambda s: s.value_counts().idxmax())
                )

                # Prefer the same day-of-week over the last ~6 months (more
                # relevant to today), but fall back to ALL days if that slice
                # is too thin (e.g. new data, or a low-volume item/category)
                # — a small same-weekday sample was producing unrealistically
                # low averages for some items.
                cutoff = sales_recent['date'].max() - pd.Timedelta(days=180)
                sales_dow = sales_recent[
                    (sales_recent['date'] >= cutoff) &
                    (sales_recent['date'].dt.dayofweek == today_dow)
                ]

                def _avg_daily_qty(src):
                    if len(src) == 0:
                        return pd.DataFrame(columns=['_cat_key', '_item_key', 'avg_daily_qty', 'n_days'])
                    daily = (
                        src.groupby(['_cat_key', '_item_key', src['date'].dt.date])['quantity']
                        .sum().reset_index()
                    )
                    # Use the 75th percentile of daily totals rather than the
                    # plain mean — a production guide should cover a
                    # busier-than-average day, not just a typical one, so
                    # kitchen staff don't run out on their better days.
                    out = (
                        daily.groupby(['_cat_key', '_item_key'])['quantity']
                        .agg(avg_daily_qty=lambda s: s.quantile(0.75), n_days='count')
                        .reset_index()
                    )
                    return out

                MIN_DAYS = 4  # minimum matching days before we trust the day-of-week average
                dow_avg = _avg_daily_qty(sales_dow)
                all_avg = _avg_daily_qty(sales_recent)

                merged = all_avg.merge(
                    dow_avg, on=['_cat_key', '_item_key'], how='left', suffixes=('_all', '_dow')
                )
                # Use the day-of-week average where we have enough matching
                # days, otherwise fall back to the all-days average.
                merged['avg_daily_qty'] = np.where(
                    merged['n_days_dow'].fillna(0) >= MIN_DAYS,
                    merged['avg_daily_qty_dow'],
                    merged['avg_daily_qty_all']
                )

                if len(merged) > 0:
                    merged['item_name'] = merged['_item_key'].map(display_names)
                    merged['category']  = merged['_cat_key'].map(display_cats)
                    top_by_cat = (
                        merged
                        .sort_values('avg_daily_qty', ascending=False)
                        .drop_duplicates(subset='category')
                        .sort_values('category')
                        .reset_index(drop=True)
                    )
                else:
                    top_by_cat = pd.DataFrame(columns=['category', 'item_name', 'avg_daily_qty'])
            else:
                top_by_cat = pd.DataFrame(columns=['category', 'item_name', 'avg_daily_qty'])

            if len(top_by_cat) == 0:
                st.info("Not enough sales history yet — upload more sales records on the Database page to see this guide.")
            else:
                # Single-row flexbox HTML — guarantees all cards same height
                cards_html = "<div style='display:flex;gap:10px;margin-bottom:16px'>"
                for _, row in top_by_cat.iterrows():
                    approx_qty = int(round(row['avg_daily_qty']))
                    unit_word  = 'cup' if row['category'] != 'Mini Bites' else 'order'
                    unit_word  = unit_word + ('s' if approx_qty != 1 else '')
                    cards_html += (
                        f"<div style='flex:1;background:#FAF6F1;border:1px solid #C4A882;"
                        f"border-top:4px solid #6F4E37;border-radius:8px;"
                        f"padding:14px 10px;text-align:center;"
                        f"display:flex;flex-direction:column;align-items:center;"
                        f"justify-content:center;min-height:90px'>"
                        f"<div style='font-size:10px;color:#A0826D;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>"
                        f"{row['category']}</div>"
                        f"<div style='font-size:13px;font-weight:600;color:#3E2723;line-height:1.4'>"
                        f"{row['item_name']}</div>"
                        f"<div style='font-size:11.5px;color:#6F4E37;font-weight:600;margin-top:4px'>"
                        f"~{approx_qty} {unit_word}</div>"
                        f"</div>"
                    )
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)

            st.markdown("---")

            # ── Prepare More / As Usual / Prepare Less ────────────────────
            st.markdown("#### Production Recommendations")

         # Deduplicate and get top unique items per demand
            waste_input_norm, wi_item_disp, _ = add_normalized_keys(waste_input, item_col='item_name')
            demand_col_numeric = pd.to_numeric(waste_input_norm['Predicted Demand (units)'], errors='coerce')

            if demand_col_numeric.notna().any():
                waste_input_norm['Predicted Demand (units)'] = demand_col_numeric
                demand_by_item = (
                    waste_input_norm.groupby('_item_key')['Predicted Demand (units)']
                    .mean().sort_values(ascending=False).reset_index()
                )
                demand_by_item['item_name'] = demand_by_item['_item_key'].map(wi_item_disp)
                demand_by_item = demand_by_item[['item_name', 'Predicted Demand (units)']].drop_duplicates(subset='item_name')
                demand_by_item.columns = ['Item', 'Demand']
            else:
                st.info("Run the prediction models to see production recommendations here.")
                demand_by_item = pd.DataFrame(columns=['Item', 'Demand'])

            # 5 items per column — focused and actionable for kitchen staff
            N = 5
            prepare_more   = demand_by_item.head(N)['Item'].tolist()
            prepare_normal = demand_by_item.iloc[N:N*2]['Item'].tolist()
            prepare_less   = demand_by_item.iloc[-(N):][::-1]['Item'].tolist()

            PREVIEW = 5  # show all 5 — no See More needed
            if 'pg_more_exp'   not in st.session_state: st.session_state.pg_more_exp   = False
            if 'pg_normal_exp' not in st.session_state: st.session_state.pg_normal_exp = False
            if 'pg_less_exp'   not in st.session_state: st.session_state.pg_less_exp   = False

            # Category-based ingredient mapping
            CATEGORY_INGREDIENTS = {
                'Coffee Based':   ['Coffee Beans', 'Milk', 'Fresh Cream'],
                'Signature Kôfē': ['Coffee Beans', 'Milk', 'Chocolate Syrup', 'Caramel Syrup'],
                'Kôfē Frappé':    ['Coffee Beans', 'Milk', 'Oreo', 'Chocolate Syrup', 'Caramel Syrup'],
                'Sans Coffee':    ['Milk', 'Matcha Powder', 'Chocolate Syrup', 'Strawberry'],
                'Mini Bites':     ['Chicken', 'Potatoes', 'Bread', 'Cheese', 'Noodles'],
            }
            cat_map = dict(zip(waste_input['item_name'], waste_input['category'])) if 'category' in waste_input.columns else {}

            def render_items(items, color, key):
                expanded = st.session_state.get(key, False)
                visible  = items if expanded else items[:PREVIEW]
                for item in visible:
                    st.markdown(
                        f"<div style='padding:7px 12px;border-left:3px solid {color};"
                        f"margin-bottom:4px;font-size:13.5px;color:#3E2723'>{item}</div>",
                        unsafe_allow_html=True
                    )
                if len(items) > PREVIEW:
                    lbl = "See Less" if expanded else f"See More ({len(items) - PREVIEW} more)"
                    if st.button(lbl, key=f"btn_{key}"):
                        st.session_state[key] = not expanded
                        st.rerun()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<div style='background:#2E7D32;color:white;padding:8px 14px;"
                    "border-radius:8px;text-align:center;font-weight:bold;"
                    "font-size:13px;margin-bottom:10px'>PREPARE MORE</div>",
                    unsafe_allow_html=True
                )
                render_items(prepare_more, '#2E7D32', 'pg_more_exp')

            with col2:
                st.markdown(
                    "<div style='background:#6F4E37;color:white;padding:8px 14px;"
                    "border-radius:8px;text-align:center;font-weight:bold;"
                    "font-size:13px;margin-bottom:10px'>PREPARE AS USUAL</div>",
                    unsafe_allow_html=True
                )
                render_items(prepare_normal, '#6F4E37', 'pg_normal_exp')

            with col3:
                st.markdown(
                    "<div style='background:#C62828;color:white;padding:8px 14px;"
                    "border-radius:8px;text-align:center;font-weight:bold;"
                    "font-size:13px;margin-bottom:10px'>PREPARE LESS</div>",
                    unsafe_allow_html=True
                )
                render_items(prepare_less, '#C62828', 'pg_less_exp')

            st.markdown("")

            # Collapsible ingredients section — clean, not cluttering the columns
            with st.expander("Ingredients Needed Today"):
                st.caption("Key ingredients to check based on today's active menu categories.")
                all_cats = sorted(set(cat_map.values())) if cat_map else []
                if all_cats:
                    ing_cols = st.columns(len(all_cats))
                    for i, cat in enumerate(all_cats):
                        ings = CATEGORY_INGREDIENTS.get(cat, [])
                        with ing_cols[i]:
                            st.markdown(
                                f"<div style='font-size:11px;color:#A0826D;font-weight:700;"
                                f"text-transform:uppercase;letter-spacing:0.5px;"
                                f"margin-bottom:8px'>{cat}</div>",
                                unsafe_allow_html=True
                            )
                            for ing in ings:
                                st.markdown(
                                    f"<div style='padding:4px 10px;border-left:3px solid #C4A882;"
                                    f"margin-bottom:3px;font-size:13px;color:#3E2723'>{ing}</div>",
                                    unsafe_allow_html=True
                                )
                else:
                    st.info("No category data available.")

            st.markdown("")
            chart_insight(
                f"This guide is calibrated for <b>{today_day_name}</b>: "
                f"<b style='color:#2E7D32'>{len(prepare_more)} item(s) to Prepare More</b>, "
                f"<b style='color:#6F4E37'>{len(prepare_normal)} item(s) to Prepare As Usual</b>, and "
                f"<b style='color:#C62828'>{len(prepare_less)} item(s) to Prepare Less</b>. "
                f"Items under Prepare More have historically high demand on {today_day_name}s — "
                f"increase your batch size. Items under Prepare Less tend to have lower demand — "
                "prepare in smaller batches to avoid waste."
            )

        else:
            st.info("Run the prediction models to generate today's production guide.")

    st.markdown("")

    # ── Restock Guide — same logic as Inventory Status, shown here too ─────
    with st.container(border=True):
        st.markdown("### Restock Guide")
        st.caption("What to restock now, what to restock this week, and what's well stocked — based on each ingredient's latest purchase record.")

        if inventory_df is not None and len(inventory_df) > 0:
            inv_hist_fc = inventory_df.copy()

            if 'spoilage_risk' in inv_hist_fc.columns:
                inv_hist_fc['alert_level'] = inv_hist_fc['spoilage_risk'].replace({
                    'High Risk':   'High Alert',
                    'Medium Risk': 'Medium Alert',
                    'Low Risk':    'Low Alert',
                    'Expired':     'Expired',
                })
            elif 'alert_level' not in inv_hist_fc.columns:
                inv_hist_fc['alert_level'] = 'Low Alert'

            if 'quantity' in inv_hist_fc.columns:
                inv_hist_fc.loc[inv_hist_fc['quantity'] <= 0, 'alert_level'] = 'Out of Stock'

            if 'purchase_date' in inv_hist_fc.columns:
                inv_hist_fc['purchase_date'] = pd.to_datetime(inv_hist_fc['purchase_date'], errors='coerce')

            if 'ingredient' in inv_hist_fc.columns and 'purchase_date' in inv_hist_fc.columns:
                latest_per_ing_fc = (
                    inv_hist_fc.dropna(subset=['purchase_date'])
                    .sort_values('purchase_date')
                    .drop_duplicates(subset='ingredient', keep='last')
                    .copy()
                )

                restock_now_fc  = latest_per_ing_fc[latest_per_ing_fc['alert_level'].isin(['Out of Stock', 'Expired'])]
                restock_soon_fc = latest_per_ing_fc[latest_per_ing_fc['alert_level'].isin(['High Alert', 'Medium Alert'])]
                well_stocked_fc = latest_per_ing_fc[latest_per_ing_fc['alert_level'] == 'Low Alert']

                def _guide_col_fc(df_g, color, empty_msg, show_reason=False):
                    if len(df_g) == 0:
                        st.caption(empty_msg)
                        return
                    rows = df_g.sort_values('quantity') if 'quantity' in df_g.columns else df_g
                    for _, r in rows.iterrows():
                        qty_txt = ""
                        if 'quantity' in df_g.columns and pd.notna(r['quantity']):
                            unit_txt = str(r['unit']) if 'unit' in df_g.columns and pd.notna(r.get('unit')) else ''
                            qty_txt = f" — {r['quantity']:.0f} {unit_txt} left".rstrip()
                        reason_txt = ""
                        if show_reason and 'alert_level' in df_g.columns and pd.notna(r.get('alert_level')):
                            reason_txt = f" <span style='color:#8A7460;font-size:11.5px'>({r['alert_level']})</span>"
                        st.markdown(
                            f"<div style='padding:7px 12px;border-left:3px solid {color};"
                            f"margin-bottom:4px;font-size:13.5px;color:#3E2723'>"
                            f"{r['ingredient']}{qty_txt}{reason_txt}</div>",
                            unsafe_allow_html=True
                        )

                rg1, rg2, rg3 = st.columns(3)
                with rg1:
                    st.markdown(
                        "<div style='background:#C62828;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>RESTOCK NOW</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col_fc(restock_now_fc, '#C62828', "Nothing out of stock or expired right now.", show_reason=True)
                with rg2:
                    st.markdown(
                        "<div style='background:#B85C00;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>RESTOCK THIS WEEK</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col_fc(restock_soon_fc, '#B85C00', "Nothing needs restocking this week.")
                with rg3:
                    st.markdown(
                        "<div style='background:#2E7D32;color:white;padding:8px 14px;"
                        "border-radius:8px;text-align:center;font-weight:bold;"
                        "font-size:13px;margin-bottom:10px'>WELL STOCKED</div>",
                        unsafe_allow_html=True
                    )
                    _guide_col_fc(well_stocked_fc, '#2E7D32', "No ingredients marked as well stocked yet.")

                st.markdown("")
                chart_insight(
                    f"<b>{len(restock_now_fc)}</b> ingredient(s) need restocking right now, "
                    f"<b>{len(restock_soon_fc)}</b> should be reordered this week, and "
                    f"<b>{len(well_stocked_fc)}</b> are currently well stocked. See the "
                    "Inventory Status page for the full breakdown."
                )
            else:
                st.info("Not enough inventory history to build a restock guide yet.")
        else:
            st.info("No inventory data found. Go to the Database page to upload inventory records first.")

    st.markdown("")

    # ============================================================
    # GRAPH 4: Menu Recommendations
    # One glance: which items to keep, improve, or remove
    # ============================================================
    with st.container(border=True):
        st.markdown("### Menu Recommendations")
        if 'Menu Performance' in waste_input.columns and 'item_name' in waste_input.columns:
            perf_order = ['Reconsider','Improve','Keep']

            col1, col2 = st.columns([1, 2])

            with col1:
                # Simple count bar
                dist = waste_input['Menu Performance'].value_counts().reindex(perf_order).fillna(0).reset_index()
                dist.columns = ['Recommendation','Count']
                fig = px.bar(
                    dist, x='Recommendation', y='Count',
                    color='Recommendation',
                    color_discrete_map=PERF_COLORS,
                    text_auto=True,
                    category_orders={'Recommendation': perf_order}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis_title='', yaxis_title='Items',
                    margin=dict(l=0,r=0,t=10,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Color-coded item lists
                for status in perf_order:
                    items_in_status = waste_input[waste_input['Menu Performance'] == status]['item_name'].unique()
                    if len(items_in_status) == 0:
                        continue
                    color = PERF_COLORS[status]
                    st.markdown(
                        f"<p style='background:{color};color:white;padding:6px 14px;"
                        f"border-radius:6px;font-weight:bold;margin:4px 0'>"
                        f"{status} — {len(items_in_status)} items</p>",
                        unsafe_allow_html=True
                    )
                    st.caption(', '.join(items_in_status[:5]) +
                              (f' +{len(items_in_status)-5} more' if len(items_in_status) > 5 else ''))

            # ── Forecast text ─────────────────────────────────────────────────
            n_keep       = int((waste_input['Menu Performance'] == 'Keep').sum())
            n_improve    = int((waste_input['Menu Performance'] == 'Improve').sum())
            n_reconsider = int((waste_input['Menu Performance'] == 'Reconsider').sum())
            total_items  = n_keep + n_improve + n_reconsider

            st.markdown(f"""
            <div style='background:#FFF8F3;border-left:4px solid #6F4E37;
                        padding:16px 20px;border-radius:8px;margin-top:8px'>
                <b>Recommendation:</b> Out of <b>{total_items}</b> menu items analyzed —
                <b style='color:{EARTH["success"]}'>{n_keep} items</b> are performing well (Keep),
                <b style='color:{EARTH["warning"]}'>{n_improve} items</b> need improvement,
                and <b style='color:{EARTH["danger"]}'>{n_reconsider} items</b> should be reconsidered.
                {' Majority of items are performing well.' if n_keep > n_reconsider
                 else ' Several items need attention. Consider reviewing pricing, portions, or removing low performers.'}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("---")

    # ── Prediction Results — kept only for the CSV download below ──────────
    result_cols = [c for c in [
        'date','item_name','category','quantity_wasted',
        'waste_reason','total_waste_cost',
        'Predicted Demand (units)', 'Predicted Waste (units)',
        'Menu Performance','Alert Level'
    ] if c in waste_input.columns]

    st.markdown("")

    # ── Download ──────────────────────────────────────────────────────────
    st.markdown("### Download Predictions")
    csv_preds = waste_input[result_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Predictions as CSV",
        data=csv_preds,
        file_name="dinedata_waste_predictions.csv",
        mime="text/csv"
    )

    # ============================================================================
    # ============================================================================
    # SECTION 2: DEMAND & WASTE FORECAST — one chart per metric, each with its
    # own filters, instead of one chart shared across a metric picker.
    # ============================================================================
    st.markdown("## Demand & Waste Forecast")
    st.caption("Projects historical monthly figures forward using a linear trend model — each metric below has its own filters.")

    waste_fc_source = waste_df if (waste_df is not None and len(waste_df) > 0) else None
    sales_fc_source = sales_df if (sales_df is not None and len(sales_df) > 0) else None

    def _render_forecast_metric(metric_choice, source_type, key_prefix):
        raw_source = waste_fc_source if source_type == 'waste' else sales_fc_source
        with st.container(border=True):
            title_ph = st.empty()
            title_ph.markdown(f"#### {metric_choice} Forecast")

            if raw_source is None:
                st.info(
                    f"No {'waste' if source_type == 'waste' else 'sales'} data with dates "
                    "found. Upload records on the Database page first."
                )
                return

            fc_df = raw_source.copy()
            fc_df['date'] = pd.to_datetime(fc_df['date'], errors='coerce')
            fc_df = fc_df.dropna(subset=['date'])

            # ── Filters — specific to this metric only ─────────────────────
            fcol2, fcol3, fcol4 = st.columns(3)
            with fcol2:
                if 'category' in fc_df.columns:
                    cats = sorted(fc_df['category'].dropna().unique().tolist())
                    sel_cat = st.selectbox("Category (optional)", ['All'] + cats, key=f'{key_prefix}_cat')
                    if sel_cat != 'All':
                        fc_df = fc_df[fc_df['category'] == sel_cat]
            with fcol3:
                if source_type == 'waste' and 'waste_reason' in fc_df.columns:
                    reasons = sorted(fc_df['waste_reason'].dropna().unique().tolist())
                    sel_reason = st.selectbox("Type of Waste (optional)", ['All'] + reasons, key=f'{key_prefix}_reason')
                    if sel_reason != 'All':
                        fc_df = fc_df[fc_df['waste_reason'] == sel_reason]
                else:
                    item_col_fc = 'item' if 'item' in fc_df.columns else ('item_name' if 'item_name' in fc_df.columns else None)
                    if source_type == 'sales' and item_col_fc:
                        items_fc = sorted(fc_df[item_col_fc].dropna().unique().tolist())
                        sel_item_fc = st.selectbox("Item (optional)", ['All'] + items_fc, key=f'{key_prefix}_item')
                        if sel_item_fc != 'All':
                            fc_df = fc_df[fc_df[item_col_fc] == sel_item_fc]
            with fcol4:
                horizon_map = {
                    '1 Year ahead': 1, '2 Years ahead': 2, '3 Years ahead': 3,
                    '5 Years ahead': 5, '10 Years ahead': 10,
                }
                sel_horizon = st.selectbox(
                    "Forecast Horizon", list(horizon_map.keys()), index=2, key=f'{key_prefix}_horizon'
                )
                horizon_years = horizon_map[sel_horizon]

            fc_df['month_period'] = fc_df['date'].dt.to_period('M').dt.to_timestamp()

            agg_kwargs = {}
            if source_type == 'waste':
                if 'total_waste_cost' in fc_df.columns:  agg_kwargs['waste_cost'] = ('total_waste_cost', 'sum')
                if 'quantity_wasted' in fc_df.columns:    agg_kwargs['units_wasted'] = ('quantity_wasted', 'sum')
            else:
                if 'quantity' in fc_df.columns: agg_kwargs['product_demand'] = ('quantity', 'sum')
                if 'total' in fc_df.columns:    agg_kwargs['sales_revenue'] = ('total', 'sum')

            monthly = fc_df.groupby('month_period').agg(**agg_kwargs).reset_index().sort_values('month_period') if agg_kwargs \
                      else pd.DataFrame(columns=['month_period'])

            y_map = {
                'Waste Cost':             ('waste_cost', 'Waste Cost (₱)', '₱'),
                'Units Wasted':           ('units_wasted', 'Units Wasted', ''),
                'Product Demand (units)': ('product_demand', 'Product Demand (units)', ''),
                'Sales Revenue':          ('sales_revenue', 'Sales Revenue (₱)', '₱'),
            }
            y_col, y_label, prefix = y_map[metric_choice]

            if y_col not in monthly.columns or len(monthly) < 3:
                st.warning("Not enough historical monthly data to build a reliable forecast (need at least 3 months).")
                return

            # ── Fit a simple linear trend on month index ────────────────
            monthly = monthly.reset_index(drop=True)
            monthly['t'] = np.arange(len(monthly))
            slope, intercept = np.polyfit(monthly['t'], monthly[y_col], 1)

            last_period = monthly['month_period'].max()
            forecast_end = last_period + pd.DateOffset(years=horizon_years)
            title_ph.markdown(f"#### {metric_choice} Forecast to {forecast_end.strftime('%Y')}")
            future_periods = pd.date_range(
                start=last_period + pd.DateOffset(months=1),
                end=forecast_end, freq='MS'
            )

            if len(future_periods) == 0:
                st.info(f"Historical data already extends to or past {forecast_end.strftime('%B %Y')} — nothing to forecast.")
                return

            future_t = np.arange(len(monthly), len(monthly) + len(future_periods))
            future_vals = np.clip(slope * future_t + intercept, a_min=0, a_max=None)

            forecast_df = pd.DataFrame({
                'month_period': future_periods,
                y_col: future_vals
            })

            # ── Optional "with intervention" target line ──────────────
            # Waste metrics: goal is to *reduce* waste.
            # Demand/sales metrics: goal is to *grow* them.
            is_growth = source_type == 'sales'
            tgcol1, tgcol2 = st.columns([1, 2])
            with tgcol1:
                show_target = st.checkbox("Show target if recommendations are followed",
                                           value=True, key=f'{key_prefix}_show_target')
            with tgcol2:
                if show_target:
                    change_pct = st.slider(
                        f"Assumed {'increase' if is_growth else 'reduction'} by end of horizon (%)",
                        5, 50, 20,
                        key=f'{key_prefix}_change_pct'
                    )
                else:
                    change_pct = 0

            if show_target:
                n = len(future_vals)
                ramp = np.linspace(0, change_pct / 100, n)   # gradual adoption, not instant
                target_vals = future_vals * (1 + ramp) if is_growth else future_vals * (1 - ramp)
                target_df = pd.DataFrame({'month_period': future_periods, y_col: target_vals})

            # ── Chart: actual (solid) + baseline forecast (dashed) [+ target] ──
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['month_period'], y=monthly[y_col],
                mode='lines+markers', name='Historical',
                line=dict(color=EARTH['primary'], width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([monthly['month_period'].tail(1), forecast_df['month_period']]),
                y=pd.concat([monthly[y_col].tail(1), forecast_df[y_col]]),
                mode='lines+markers', name='Forecast — no action (baseline)',
                line=dict(color=EARTH['accent'], width=2.5, dash='dash')
            ))
            if show_target:
                fig.add_trace(go.Scatter(
                    x=pd.concat([monthly['month_period'].tail(1), target_df['month_period']]),
                    y=pd.concat([monthly[y_col].tail(1), target_df[y_col]]),
                    mode='lines+markers',
                    name=f"Target — with action ({'+' if is_growth else '-'}{change_pct}%)",
                    line=dict(color=EARTH['success'], width=2.5, dash='dot')
                ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Month', yaxis_title=y_label,
                yaxis=dict(tickprefix=prefix, tickformat=',.0f'),
                xaxis=dict(
                    tickformat='%b %Y',      # e.g. "Dec 2026" — stays readable when zoomed in
                    dtick='M1',               # force a tick for every month
                    hoverformat='%b %Y',
                    rangeslider=dict(visible=True, thickness=0.08),
                    rangeselector=dict(
                        buttons=[
                            dict(count=6,  label='6m',  step='month', stepmode='backward'),
                            dict(count=1,  label='1y',  step='year',  stepmode='backward'),
                            dict(count=2,  label='2y',  step='year',  stepmode='backward'),
                            dict(step='all', label='All'),
                        ]
                    )
                ),
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── KPIs ────────────────────────────────────────────────
            if show_target:
                k1, k2, k3, k4 = st.columns(4)
            else:
                k1, k2, k3 = st.columns(3)
            with k1:
                stat_card(f"Latest actual ({last_period.strftime('%b %Y')})",
                          f"{prefix}{monthly[y_col].iloc[-1]:,.0f}")
            with k2:
                stat_card(f"Baseline {forecast_end.strftime('%b %Y')}",
                          f"{prefix}{forecast_df[y_col].iloc[-1]:,.0f}")
            with k3:
                trend_word = "growing" if slope > 0 else ("declining" if slope < 0 else "flat")
                trend_dir_ui = "up" if slope > 0 else ("down" if slope < 0 else "flat")
                stat_card("Monthly trend", f"{prefix}{slope:,.0f} / mo", trend_word, trend_dir_ui)
            if show_target:
                with k4:
                    diff_amt = target_df[y_col].iloc[-1] - forecast_df[y_col].iloc[-1]
                    stat_card(f"Target {forecast_end.strftime('%b %Y')}",
                              f"{prefix}{target_df[y_col].iloc[-1]:,.0f}",
                              f"{prefix}{abs(diff_amt):,.0f} {'gained' if is_growth else 'saved'}", "up")

            target_line = (
                (
                    f"If the system's recommendations are followed (better prep quantities, storage, "
                    f"and FIFO practices), {metric_choice.lower()} could instead be reduced to roughly "
                    f"<b>{prefix}{target_df[y_col].iloc[-1]:,.0f}</b> by {forecast_end.strftime('%B %Y')} — "
                    f"a savings of about <b>{prefix}{abs(diff_amt):,.0f}</b>."
                ) if (show_target and not is_growth) else (
                    f"If the system's recommendations are followed (better stocking, promos, and "
                    f"demand-matched prep), {metric_choice.lower()} could instead grow to roughly "
                    f"<b>{prefix}{target_df[y_col].iloc[-1]:,.0f}</b> by {forecast_end.strftime('%B %Y')} — "
                    f"about <b>{prefix}{abs(diff_amt):,.0f}</b> more than the baseline trend."
                ) if show_target else ""
            )

            baseline_note = (
                "<b>This dashed \"baseline\" line is what happens if nothing changes</b> — it's the "
                "problem this system exists to solve, not a prediction that waste has to increase."
                if source_type == 'waste' else
                "<b>This dashed \"baseline\" line is a simple continuation of the current trend</b> — "
                "useful for planning staffing, inventory, and production levels ahead of time."
            )
            st.markdown(f"""
            <div style='background:#FFF8F3;border-left:4px solid #6F4E37;
                        padding:16px 20px;border-radius:8px;margin-top:8px'>
                <b>Forecast:</b> Based on the historical trend from
                <b>{monthly['month_period'].min().strftime('%b %Y')}</b> to
                <b>{last_period.strftime('%b %Y')}</b>, {metric_choice.lower()} is
                <b>{trend_word.split()[0]}</b> by roughly <b>{prefix}{abs(slope):,.0f} per month</b>.
                {baseline_note}
                {target_line}
                This is a simple linear projection — actual results will vary with seasonality
                and operational changes.
            </div>
            """, unsafe_allow_html=True)

            # ── Combined table ──────────────────────────────────────
            st.markdown("###### Historical + Forecasted Values")
            combined_table = pd.concat([
                monthly[['month_period', y_col]].assign(Type='Historical'),
                forecast_df[['month_period', y_col]].assign(Type='Forecast (baseline)')
            ]).rename(columns={'month_period': 'Month', y_col: metric_choice})
            if show_target:
                target_rows = target_df[['month_period', y_col]].assign(Type='Target (with action)') \
                    .rename(columns={'month_period': 'Month', y_col: metric_choice})
                combined_table = pd.concat([combined_table, target_rows])
            combined_table['Month'] = pd.to_datetime(combined_table['Month']).dt.strftime('%b %Y')
            combined_table = combined_table.reset_index(drop=True)

            def _style_fc_type(val):
                m = {
                    'Historical':           'background-color:#EFE6DA;color:#3E2723;font-weight:600',
                    'Forecast (baseline)':  'background-color:#FDF1DE;color:#B85C00;font-weight:600',
                    'Target (with action)': 'background-color:#E7F3E8;color:#2E7D32;font-weight:600',
                }
                return m.get(val, '')

            metric_fmt = (lambda x: f"{prefix}{x:,.0f}") if prefix else (lambda x: f"{x:,.0f}")
            styled_fc = style_map(combined_table.style, _style_fc_type, subset=['Type'])
            styled_fc = styled_fc.format({metric_choice: metric_fmt})
            st.dataframe(styled_fc, use_container_width=True, hide_index=True)

            csv_fc = combined_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Forecast as CSV", data=csv_fc,
                file_name=f"{metric_choice.lower().replace(' ', '_').replace('(', '').replace(')', '')}_forecast_{forecast_end.strftime('%Y')}.csv",
                mime="text/csv", key=f'{key_prefix}_download'
            )

    _render_forecast_metric('Waste Cost', 'waste', 'fc_wastecost')
    st.markdown("")
    _render_forecast_metric('Units Wasted', 'waste', 'fc_unitswasted')
    st.markdown("")
    _render_forecast_metric('Product Demand (units)', 'sales', 'fc_demand')
    st.markdown("")
    _render_forecast_metric('Sales Revenue', 'sales', 'fc_salesrev')

elif page == 'Database':

    hero_banner(
        "Business Records",
        "Database",
        "View, filter, and manage all data stored in the DineData system."
    )

    # ============================================================================
    # UPLOAD DATA — single entry point for all 4 data types.
    # Uploading is no longer scattered across Sales Analytics, Waste Analytics,
    # Menu Performance, and Inventory Status — everything comes through here.
    # ============================================================================
    with st.container(border=True):
        st.markdown("### Upload Data")
        st.caption("Add new records to any part of the system. Pick a data type, download its template if needed, then upload your filled CSV.")

        upload_type = st.selectbox(
            "Data type", ['Sales', 'Waste', 'Menu', 'Inventory'], key='db_upload_type'
        )

        # ── SALES ────────────────────────────────────────────────────────
        if upload_type == 'Sales':
            SALES_FILE_UP = DATA_PATHS['sales']

            def _load_sales_up():
                try:
                    df = pd.read_csv(SALES_FILE_UP)
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    return df
                except Exception:
                    return pd.DataFrame(columns=['date','item','category','quantity','price','total','cost'])

            sales_template = (
                "date,item,category,quantity,price,total,cost\n"
                "2026-04-01,Caramel Macchiato (iced),Coffee Based,25,140.00,3500.00,70.00\n"
                "2026-04-01,Overload Lomi (malapot),Mini Bites,10,100.00,1000.00,50.00\n"
                "2026-04-02,Chicken Wings (3pcs),Mini Bites,12,150.00,1800.00,80.00\n"
            )
            st.download_button(
                "Download Sales Log Template (CSV)", data=sales_template,
                file_name="sales_log_template.csv", mime="text/csv", key='db_sales_template_dl'
            )

            if 'sales_upload_msg' in st.session_state:
                st.success(st.session_state.pop('sales_upload_msg'))

            uploaded_sales = st.file_uploader(
                "Upload Sales CSV", type=['csv'], key='db_sales_upload'
            )
            if uploaded_sales is not None:
                try:
                    new_sales = pd.read_csv(uploaded_sales)
                    new_sales['date'] = pd.to_datetime(new_sales['date'], errors='coerce')
                    st.markdown(f"#### Preview — {len(new_sales)} new records")
                    st.dataframe(new_sales, use_container_width=True)

                    if st.button("Add to Existing Data", type="primary", key='db_sales_add_btn'):
                        existing_sales = _load_sales_up()
                        combined = pd.concat([existing_sales, new_sales], ignore_index=True)
                        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
                        # Normalize item/category casing across existing + new
                        # rows together, so upload-time spelling differences
                        # don't create a permanent duplicate item variant.
                        combined = normalize_text_columns(combined, ['item', 'category'])
                        new_sales_norm = combined.tail(len(new_sales)).copy()
                        before = len(combined)
                        combined = combined.drop_duplicates(subset=['date','item','quantity','total'], keep='first').reset_index(drop=True)
                        dupes_removed = before - len(combined)
                        combined = combined.sort_values('date').reset_index(drop=True)
                        combined.to_csv(SALES_FILE_UP, index=False)

                        db_inserted = 0
                        if db_available():
                            try:
                                db_inserted = insert_sales_records_to_db(new_sales_norm)
                            except Exception as db_err:
                                st.caption(f"Note: Could not sync to database — {db_err}")

                        dupes_msg = f" ({dupes_removed} duplicates removed)" if dupes_removed > 0 else " No duplicates found."
                        db_msg    = f" {db_inserted} records also saved to database." if db_inserted > 0 else ""
                        st.session_state['sales_upload_msg'] = f"Added {len(new_sales)} records!{dupes_msg}{db_msg}"
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # ── WASTE ────────────────────────────────────────────────────────
        elif upload_type == 'Waste':
            WASTE_FILE_UP = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_waste_data.csv')

            def _load_waste_up():
                try:
                    df = pd.read_csv(WASTE_FILE_UP)
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    return df
                except Exception:
                    return pd.DataFrame(columns=[
                        'date','item_name','category','quantity_wasted',
                        'waste_reason','cost_per_item','total_waste_cost'
                    ])

            waste_template = (
                "date,item_name,category,quantity_wasted,waste_reason,cost_per_item,total_waste_cost\n"
                "2026-04-01,Caramel Macchiato (iced),Coffee Based,2,Over-preparation,140.00,280.00\n"
                "2026-04-01,Overload Lomi (malapot),Mini Bites,1,Spoilage,100.00,100.00\n"
                "2026-04-02,Chicken Wings (3pcs),Mini Bites,3,Spoilage,150.00,450.00\n"
            )
            st.download_button(
                "Download Waste Log Template (CSV)", data=waste_template,
                file_name="waste_log_template.csv", mime="text/csv", key='db_waste_template_dl'
            )

            if 'waste_upload_msg' in st.session_state:
                st.success(st.session_state.pop('waste_upload_msg'))

            uploaded_waste = st.file_uploader(
                "Upload Waste CSV", type=['csv'], key='db_waste_upload'
            )
            if uploaded_waste is not None:
                try:
                    new_waste = pd.read_csv(uploaded_waste)
                    new_waste['date'] = pd.to_datetime(new_waste['date'], errors='coerce')
                    st.markdown(f"#### Preview — {len(new_waste)} new records")
                    st.dataframe(new_waste, use_container_width=True)

                    if st.button("Add to Existing Data", type="primary", key='db_waste_add_btn'):
                        existing_waste_up = _load_waste_up()
                        combined = pd.concat([existing_waste_up, new_waste], ignore_index=True)
                        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
                        combined = normalize_text_columns(combined, ['item_name', 'category', 'waste_reason'])
                        new_waste_norm = combined.tail(len(new_waste)).copy()
                        before = len(combined)
                        combined = combined.drop_duplicates(
                            subset=['date','item_name','quantity_wasted','waste_reason'], keep='first'
                        ).reset_index(drop=True)
                        dupes_removed = before - len(combined)
                        combined = combined.sort_values('date').reset_index(drop=True)
                        combined.to_csv(WASTE_FILE_UP, index=False)

                        db_inserted = 0
                        if db_available():
                            try:
                                db_inserted = insert_waste_records_to_db(new_waste_norm)
                            except Exception as db_err:
                                st.caption(f"Note: Could not sync to database — {db_err}")

                        dupes_msg = f" ({dupes_removed} duplicates removed)" if dupes_removed > 0 else " No duplicates found."
                        db_msg    = f" {db_inserted} records also saved to database." if db_inserted > 0 else ""
                        st.session_state['waste_upload_msg'] = f"Added {len(new_waste)} records!{dupes_msg}{db_msg}"
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # ── MENU ─────────────────────────────────────────────────────────
        elif upload_type == 'Menu':
            MENU_FILE_UP = DATA_PATHS['menu']

            def _load_menu_up():
                try:
                    return pd.read_csv(MENU_FILE_UP)
                except Exception:
                    return pd.DataFrame(columns=['item_name','category','price','cost'])

            menu_template = (
                "item_name,category,price,cost\n"
                "Caramel Macchiato (iced),Coffee Based,140.00,70.00\n"
                "Overload Lomi (malapot),Mini Bites,100.00,50.00\n"
                "Chicken Wings (3pcs),Mini Bites,150.00,80.00\n"
            )
            st.download_button(
                "Download Menu Template (CSV)", data=menu_template,
                file_name="menu_template.csv", mime="text/csv", key='db_menu_template_dl'
            )
            st.caption("Uploading an item that already exists (same name) will update its price/cost.")

            if 'menu_upload_msg' in st.session_state:
                st.success(st.session_state.pop('menu_upload_msg'))

            uploaded_menu = st.file_uploader(
                "Upload Menu CSV", type=['csv'], key='db_menu_upload'
            )
            if uploaded_menu is not None:
                try:
                    new_menu = pd.read_csv(uploaded_menu)
                    st.markdown(f"#### Preview — {len(new_menu)} items")
                    st.dataframe(new_menu, use_container_width=True)

                    if st.button("Add / Update Menu Items", type="primary", key='db_menu_add_btn'):
                        existing_menu = _load_menu_up()
                        combined = pd.concat([existing_menu, new_menu], ignore_index=True)
                        combined = normalize_text_columns(combined, ['item_name', 'category'])
                        new_menu_norm = combined.tail(len(new_menu)).copy()
                        combined = combined.drop_duplicates(subset=['item_name'], keep='last').reset_index(drop=True)
                        combined.to_csv(MENU_FILE_UP, index=False)

                        db_inserted = 0
                        if db_available():
                            try:
                                db_inserted = insert_menu_records_to_db(new_menu_norm)
                            except Exception as db_err:
                                st.caption(f"Note: Could not sync to database — {db_err}")

                        db_msg = f" {db_inserted} item(s) also saved to database." if db_inserted > 0 else ""
                        st.session_state['menu_upload_msg'] = f"Added/updated {len(new_menu)} menu item(s)!{db_msg}"
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # ── INVENTORY ────────────────────────────────────────────────────
        else:
            INVENTORY_FILE_UP = DATA_PATHS['inventory']

            st.caption(
                "Columns: purchase_date, ingredient, quantity, unit, cost_per_unit, "
                "total_cost, expiration_date, shelf_life_days. New records are **added** "
                "to existing inventory — restockable items and brand-new ingredients both go here."
            )
            template_inv = (
                "purchase_date,ingredient,quantity,unit,cost_per_unit,total_cost,expiration_date,shelf_life_days\n"
                "2026-06-01,Coffee Beans,10,kg,850,8500,2026-07-31,60\n"
                "2026-06-01,Milk,8,liter,90,720,2026-06-08,7\n"
            )
            st.download_button(
                "Download Inventory Template (CSV)", data=template_inv,
                file_name="inventory_template.csv", mime="text/csv", key='db_inv_template_dl'
            )

            if 'inv_upload_msg' in st.session_state:
                st.success(st.session_state.pop('inv_upload_msg'))

            uploaded_inv = st.file_uploader(
                "Upload Inventory CSV", type=['csv'], key='db_inventory_upload'
            )
            if uploaded_inv is not None:
                try:
                    new_inv = pd.read_csv(uploaded_inv)
                    st.markdown(f"#### Preview — {len(new_inv)} new records")
                    st.dataframe(new_inv, use_container_width=True)

                    if st.button("Add to Existing Inventory", type="primary", key='db_inv_add_btn'):
                        existing_raw = pd.read_csv(INVENTORY_FILE_UP) if os.path.exists(INVENTORY_FILE_UP) else pd.DataFrame()
                        combined = pd.concat([existing_raw, new_inv], ignore_index=True)
                        if 'ingredient' in combined.columns:
                            combined = normalize_text_columns(combined, ['ingredient'])
                        new_inv_norm = combined.tail(len(new_inv)).copy()
                        before_count = len(combined)
                        if {'purchase_date', 'ingredient'}.issubset(combined.columns):
                            combined = combined.drop_duplicates(
                                subset=['purchase_date', 'ingredient', 'quantity'], keep='first'
                            )
                        dupes_removed_inv = before_count - len(combined)
                        combined.to_csv(INVENTORY_FILE_UP, index=False)

                        n_inv_inserted = 0
                        if db_available():
                            new_inv_typed = new_inv_norm.copy()
                            new_inv_typed['purchase_date'] = pd.to_datetime(new_inv_typed['purchase_date'], errors='coerce')
                            if 'expiration_date' not in new_inv_typed.columns and 'shelf_life_days' in new_inv_typed.columns:
                                new_inv_typed['expiration_date'] = new_inv_typed['purchase_date'] + pd.to_timedelta(new_inv_typed['shelf_life_days'], unit='D')
                            else:
                                new_inv_typed['expiration_date'] = pd.to_datetime(new_inv_typed.get('expiration_date'), errors='coerce')
                            today_db = pd.Timestamp.now().normalize()
                            new_inv_typed['days_until_expiration'] = (new_inv_typed['expiration_date'] - today_db).dt.days
                            def _alert_lvl(days):
                                if pd.isna(days): return 'Low Alert'
                                if days < 0: return 'Expired'
                                elif days <= 2: return 'High Alert'
                                elif days <= 7: return 'Medium Alert'
                                else: return 'Low Alert'
                            new_inv_typed['alert_level'] = new_inv_typed['days_until_expiration'].apply(_alert_lvl)
                            n_inv_inserted = insert_inventory_records_to_db(new_inv_typed)

                        dupes_msg_inv = f" ({dupes_removed_inv} duplicates removed)" if dupes_removed_inv > 0 else " No duplicates found."
                        db_msg_inv    = f" {n_inv_inserted} record(s) also saved to database." if n_inv_inserted > 0 else ""
                        st.session_state['inv_upload_msg'] = f"Added {len(new_inv)} records to inventory!{dupes_msg_inv}{db_msg_inv}"
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    st.markdown("")

    if not db_available():
        st.warning(
            "No database found yet. Please contact your system administrator "
            "to set up the database."
        )
    else:
        conn = get_db_connection()

        # ── Record count summary cards ─────────────────────────────────────
        try:
            sales_count = conn.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0]
            waste_count = conn.execute("SELECT COUNT(*) FROM FACT_WASTE").fetchone()[0]
            inv_count   = conn.execute("SELECT COUNT(*) FROM FACT_INVENTORY").fetchone()[0]
            menu_count  = conn.execute("SELECT COUNT(*) FROM DIM_ITEM").fetchone()[0]
        except Exception:
            sales_count = waste_count = inv_count = menu_count = 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: stat_card("Sales Records",     f"{sales_count:,}")
        with c2: stat_card("Waste Records",     f"{waste_count:,}")
        with c3: stat_card("Inventory Records", f"{inv_count:,}")
        with c4: stat_card("Menu Items",        f"{menu_count:,}")

        st.markdown("")

        month_list = [
            'All','January','February','March','April','May','June',
            'July','August','September','October','November','December'
        ]
        mn_map = {m: i+1 for i, m in enumerate(month_list[1:])}
        q_map  = {'Q1':1,'Q2':2,'Q3':3,'Q4':4}

        # ── Tabs — one per data type ───────────────────────────────────────
        with st.container(border=True):
            st.markdown("### View Records")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Sales", "Waste Records", "Inventory", "Menu Items"
            ])

            # ─────────────────────────────────────────────────────────────
            # TAB 1: SALES (editable — now that Sales has an upload feature)
            # ─────────────────────────────────────────────────────────────
            with tab1:
                st.caption("All sales transactions. You can correct or remove wrong entries here.")

                try:
                    minmax_s = conn.execute(
                        "SELECT MIN(d.date), MAX(d.date) FROM FACT_SALES s "
                        "JOIN DIM_DATE d ON s.date_key=d.date_key"
                    ).fetchone()
                    min_date_s = pd.to_datetime(minmax_s[0]).date() if minmax_s and minmax_s[0] else date(2020, 1, 1)
                    max_date_s = pd.to_datetime(minmax_s[1]).date() if minmax_s and minmax_s[1] else date.today()
                except Exception:
                    min_date_s, max_date_s = date(2020, 1, 1), date.today()
                try:
                    cats = ['All'] + [r[0] for r in conn.execute(
                        "SELECT DISTINCT category FROM DIM_ITEM ORDER BY category"
                    ).fetchall() if r[0]]
                except Exception:
                    cats = ['All']

                f1, f2 = st.columns([2, 1])
                with f1:
                    s_date_range = st.date_input(
                        "Date Range", value=(min_date_s, max_date_s),
                        min_value=min_date_s, max_value=max_date_s, key='s_daterange'
                    )
                with f2:
                    s_cat = st.selectbox("Category", cats, key='s_cat')

                sw = []
                if isinstance(s_date_range, tuple) and len(s_date_range) == 2:
                    sw.append(f"d.date BETWEEN '{s_date_range[0]}' AND '{s_date_range[1]}'")
                elif isinstance(s_date_range, date):
                    sw.append(f"d.date = '{s_date_range}'")
                if s_cat != 'All': sw.append(f"i.category = '{s_cat}'")
                sw_sql = ("WHERE " + " AND ".join(sw)) if sw else ""

                sq = (
                    "SELECT d.date AS [Date], d.day_name AS [Day], "
                    "i.item_name AS [Item], i.category AS [Category], "
                    "s.quantity AS [Qty Sold], s.price AS [Price], "
                    "s.total AS [Revenue], s.cost AS [Cost] "
                    "FROM FACT_SALES s "
                    "JOIN DIM_DATE d ON s.date_key = d.date_key "
                    "JOIN DIM_ITEM i ON s.item_key = i.item_key "
                    f"{sw_sql} ORDER BY d.date DESC"
                )
                try:
                    total_s = conn.execute(f"SELECT COUNT(*) FROM ({sq})").fetchone()[0]
                    n_s = st.slider("Records to show", 10, 500, 50, 10, key='s_rows')
                    df_s = pd.read_sql(f"{sq} LIMIT {n_s}", conn)
                    st.caption(f"Showing {len(df_s):,} of {total_s:,} records")

                    edited_s = st.data_editor(
                        df_s, use_container_width=True, hide_index=True,
                        key='s_editor', num_rows="dynamic",
                        disabled=["Date", "Day", "Item", "Category"]
                    )

                    # ── Action buttons: Save, Delete by filter, Download ───────
                    sc1, sc2, sc3 = st.columns([1, 1, 2])

                    with sc1:
                        if st.button("Save Changes", type="primary", key='s_save'):
                            try:
                                for _, row in edited_s.iterrows():
                                    conn.execute(
                                        "UPDATE FACT_SALES SET quantity=?, price=?, total=?, cost=? "
                                        "WHERE sale_key IN ("
                                        "SELECT s.sale_key FROM FACT_SALES s "
                                        "JOIN DIM_DATE d ON s.date_key=d.date_key "
                                        "JOIN DIM_ITEM i ON s.item_key=i.item_key "
                                        "WHERE d.date=? AND i.item_name=? AND s.total=?)",
                                        (row.get('Qty Sold'), row.get('Price'), row.get('Revenue'),
                                         row.get('Cost'), row.get('Date'), row.get('Item'), row.get('Revenue'))
                                    )
                                conn.commit()
                                st.success("Changes saved successfully!")
                            except Exception as e:
                                st.error(f"Could not save: {e}")

                    with sc2:
                        # Same freeze-before-confirm safety pattern as Waste/Inventory
                        if sw_sql:
                            del_label_s = f"Delete filtered ({total_s:,} records)"
                        else:
                            del_label_s = "Delete All Records"

                        if st.button(del_label_s, key='s_del_filter'):
                            st.session_state['s_confirm_del']      = True
                            st.session_state['s_del_sql_frozen']   = sw_sql
                            st.session_state['s_del_count_frozen'] = total_s

                        if st.session_state.get('s_confirm_del', False):
                            frozen_sql_s   = st.session_state.get('s_del_sql_frozen', sw_sql)
                            frozen_count_s = st.session_state.get('s_del_count_frozen', total_s)

                            if not frozen_sql_s:
                                st.error(
                                    "⚠️ No filter is applied — this would delete "
                                    f"ALL {frozen_count_s:,} sales records. Type "
                                    "DELETE ALL below to confirm, or Cancel."
                                )
                                confirm_text_s = st.text_input(
                                    "Type DELETE ALL to confirm", key='s_del_all_text'
                                )
                                proceed_allowed_s = (confirm_text_s.strip() == "DELETE ALL")
                            else:
                                st.warning(
                                    f"This will permanently delete **{frozen_count_s:,} records** "
                                    f"matching the filter you selected. This cannot be undone."
                                )
                                proceed_allowed_s = True

                            scd1, scd2 = st.columns(2)
                            with scd1:
                                if st.button("Delete", type="primary", key='s_del_confirm',
                                             use_container_width=True, disabled=not proceed_allowed_s):
                                    try:
                                        del_sql_s = (
                                            "DELETE FROM FACT_SALES WHERE sale_key IN ("
                                            "SELECT s.sale_key FROM FACT_SALES s "
                                            "JOIN DIM_DATE d ON s.date_key=d.date_key "
                                            "JOIN DIM_ITEM i ON s.item_key=i.item_key "
                                            f"{frozen_sql_s})"
                                        )
                                        conn.execute(del_sql_s)
                                        conn.commit()
                                        st.session_state['s_confirm_del'] = False
                                        st.success(f"Deleted {frozen_count_s:,} records successfully!")
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not delete: {e}")
                            with scd2:
                                if st.button("Cancel", key='s_del_cancel', use_container_width=True):
                                    st.session_state['s_confirm_del'] = False
                                    st.rerun()

                    with sc3:
                        st.download_button(
                            "Download Sales as CSV",
                            data=df_s.to_csv(index=False).encode("utf-8"),
                            file_name="sales_records.csv", mime="text/csv", key="dl_s"
                    )

                    # ── Delete individual selected rows ────────────────────
                    st.markdown("---")
                    st.markdown("**Delete Individual Records**")
                    st.caption("Select rows to delete by checking the box on the left.")

                    df_s_select = df_s.copy()
                    df_s_select.insert(0, 'Select', False)
                    edited_s_sel = st.data_editor(
                        df_s_select,
                        use_container_width=True,
                        hide_index=True,
                        key='s_sel_editor',
                        column_config={
                            'Select': st.column_config.CheckboxColumn('Select', default=False)
                        },
                        disabled=[c for c in df_s_select.columns if c != 'Select']
                    )
                    selected_rows_s = edited_s_sel[edited_s_sel['Select'] == True]
                    if len(selected_rows_s) > 0:
                        st.caption(f"{len(selected_rows_s)} row(s) selected")
                        if st.button(f"Delete {len(selected_rows_s)} Selected Row(s)", type="primary", key='s_del_rows'):
                            try:
                                for _, row in selected_rows_s.iterrows():
                                    conn.execute(
                                        "DELETE FROM FACT_SALES WHERE sale_key IN ("
                                        "SELECT s.sale_key FROM FACT_SALES s "
                                        "JOIN DIM_DATE d ON s.date_key=d.date_key "
                                        "JOIN DIM_ITEM i ON s.item_key=i.item_key "
                                        "WHERE d.date=? AND i.item_name=? AND s.total=?)",
                                        (row.get('Date'), row.get('Item'), row.get('Revenue'))
                                    )
                                conn.commit()
                                st.success(f"Deleted {len(selected_rows_s)} record(s)!")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not delete: {e}")
                except Exception as e:
                    st.error(f"Could not load sales: {e}")

            # ─────────────────────────────────────────────────────────────
            # TAB 2: WASTE RECORDS (editable)
            # ─────────────────────────────────────────────────────────────
            with tab2:
                st.caption("Waste entries uploaded by staff. You can correct any wrong entries here.")

                try:
                    minmax_w = conn.execute(
                        "SELECT MIN(d.date), MAX(d.date) FROM FACT_WASTE w "
                        "JOIN DIM_DATE d ON w.date_key=d.date_key"
                    ).fetchone()
                    min_date_w = pd.to_datetime(minmax_w[0]).date() if minmax_w and minmax_w[0] else date(2020, 1, 1)
                    max_date_w = pd.to_datetime(minmax_w[1]).date() if minmax_w and minmax_w[1] else date.today()
                except Exception:
                    min_date_w, max_date_w = date(2020, 1, 1), date.today()
                try:
                    reasons = ['All'] + [r[0] for r in conn.execute(
                        "SELECT reason_name FROM DIM_WASTE_REASON ORDER BY reason_name"
                    ).fetchall()]
                except Exception:
                    reasons = ['All']

                f1, f2 = st.columns([2, 1])
                with f1:
                    w_date_range = st.date_input(
                        "Date Range", value=(min_date_w, max_date_w),
                        min_value=min_date_w, max_value=max_date_w, key='w_daterange'
                    )
                with f2:
                    w_reason = st.selectbox("Waste Reason", reasons, key='w_reason')

                ww = []
                if isinstance(w_date_range, tuple) and len(w_date_range) == 2:
                    ww.append(f"d.date BETWEEN '{w_date_range[0]}' AND '{w_date_range[1]}'")
                elif isinstance(w_date_range, date):
                    ww.append(f"d.date = '{w_date_range}'")
                if w_reason  != 'All': ww.append(f"r.reason_name = '{w_reason}'")
                ww_sql = ("WHERE " + " AND ".join(ww)) if ww else ""

                wq = (
                    "SELECT d.date AS [Date], d.day_name AS [Day], "
                    "i.item_name AS [Item], i.category AS [Category], "
                    "w.quantity_wasted AS [Units Wasted], "
                    "r.reason_name AS [Waste Reason], "
                    "w.cost_per_item AS [Cost per Item], "
                    "w.total_waste_cost AS [Total Waste Cost] "
                    "FROM FACT_WASTE w "
                    "JOIN DIM_DATE d ON w.date_key = d.date_key "
                    "JOIN DIM_ITEM i ON w.item_key = i.item_key "
                    "JOIN DIM_WASTE_REASON r ON w.reason_key = r.reason_key "
                    f"{ww_sql} ORDER BY d.date DESC"
                )
                try:
                    total_w = conn.execute(f"SELECT COUNT(*) FROM ({wq})").fetchone()[0]
                    n_w = st.slider("Records to show", 10, 500, 50, 10, key='w_rows')
                    df_w = pd.read_sql(f"{wq} LIMIT {n_w}", conn)
                    st.caption(f"Showing {len(df_w):,} of {total_w:,} records")

                    edited_w = st.data_editor(
                        df_w, use_container_width=True, hide_index=True,
                        key='w_editor', num_rows="dynamic",
                        disabled=["Date","Day","Item","Category","Waste Reason"]
                    )

                    # ── Action buttons: Save, Delete by filter, Download ───────
                    wc1, wc2, wc3 = st.columns([1, 1, 2])

                    with wc1:
                        if st.button("Save Changes", type="primary", key='w_save'):
                            try:
                                for _, row in edited_w.iterrows():
                                    conn.execute(
                                        "UPDATE FACT_WASTE SET "
                                        "quantity_wasted=?, cost_per_item=?, total_waste_cost=? "
                                        "WHERE waste_key IN ("
                                        "SELECT w.waste_key FROM FACT_WASTE w "
                                        "JOIN DIM_DATE d ON w.date_key=d.date_key "
                                        "JOIN DIM_ITEM i ON w.item_key=i.item_key "
                                        "JOIN DIM_WASTE_REASON r ON w.reason_key=r.reason_key "
                                        "WHERE d.date=? AND i.item_name=? AND r.reason_name=?)",
                                        (row.get('Units Wasted'), row.get('Cost per Item'),
                                         row.get('Total Waste Cost'), row.get('Date'),
                                         row.get('Item'), row.get('Waste Reason'))
                                    )
                                conn.commit()
                                st.success("Changes saved successfully!")
                            except Exception as e:
                                st.error(f"Could not save: {e}")

                    with wc2:
                        # Delete by current filter — confirms before deleting.
                        # CRITICAL: the filter (ww_sql) and count are FROZEN into
                        # session_state the instant "Delete filtered" is clicked,
                        # and the actual DELETE always uses that frozen value —
                        # never the live filter state. This prevents a dangerous
                        # bug where the widget's value could change between the
                        # first click and the confirm click (e.g. the date range
                        # briefly resolving to a single date mid-rerun), silently
                        # turning the filter into "no filter" and deleting the
                        # entire table instead of just the shown records.
                        if ww_sql:
                            del_label = f"Delete filtered ({total_w:,} records)"
                        else:
                            del_label = "Delete All Records"

                        if st.button(del_label, key='w_del_filter'):
                            st.session_state['w_confirm_del']      = True
                            st.session_state['w_del_sql_frozen']   = ww_sql
                            st.session_state['w_del_count_frozen'] = total_w

                        if st.session_state.get('w_confirm_del', False):
                            frozen_sql   = st.session_state.get('w_del_sql_frozen', ww_sql)
                            frozen_count = st.session_state.get('w_del_count_frozen', total_w)

                            # Hard safety net: never allow an unfiltered delete
                            # to proceed silently — require typed confirmation.
                            if not frozen_sql:
                                st.error(
                                    "⚠️ No filter is applied — this would delete "
                                    f"ALL {frozen_count:,} waste records. Type "
                                    "DELETE ALL below to confirm, or Cancel."
                                )
                                confirm_text = st.text_input(
                                    "Type DELETE ALL to confirm", key='w_del_all_text'
                                )
                                proceed_allowed = (confirm_text.strip() == "DELETE ALL")
                            else:
                                st.warning(
                                    f"This will permanently delete **{frozen_count:,} records** "
                                    f"matching the filter you selected. This cannot be undone."
                                )
                                proceed_allowed = True

                            cd1, cd2 = st.columns(2)
                            with cd1:
                                if st.button("Delete", type="primary", key='w_del_confirm',
                                             use_container_width=True, disabled=not proceed_allowed):
                                    try:
                                        # Use the FROZEN filter — guaranteed to match
                                        # what the user actually saw and agreed to.
                                        del_sql = (
                                            "DELETE FROM FACT_WASTE WHERE waste_key IN ("
                                            "SELECT w.waste_key FROM FACT_WASTE w "
                                            "JOIN DIM_DATE d ON w.date_key=d.date_key "
                                            "JOIN DIM_ITEM i ON w.item_key=i.item_key "
                                            "JOIN DIM_WASTE_REASON r ON w.reason_key=r.reason_key "
                                            f"{frozen_sql})"
                                        )
                                        conn.execute(del_sql)
                                        conn.commit()
                                        st.session_state['w_confirm_del'] = False
                                        st.success(f"Deleted {frozen_count:,} records successfully!")
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not delete: {e}")
                            with cd2:
                                if st.button("Cancel", key='w_del_cancel', use_container_width=True):
                                    st.session_state['w_confirm_del'] = False
                                    st.rerun()

                    with wc3:
                        st.download_button(
                            "Download Waste Records as CSV",
                            data=df_w.to_csv(index=False).encode("utf-8"),
                            file_name="waste_records.csv", mime="text/csv", key="dl_w"
                        )

                    # ── Delete individual selected rows ────────────────────
                    st.markdown("---")
                    st.markdown("**Delete Individual Records**")
                    st.caption("Select rows to delete by checking the box on the left.")

                    df_w_select = df_w.copy()
                    df_w_select.insert(0, 'Select', False)
                    edited_sel = st.data_editor(
                        df_w_select,
                        use_container_width=True,
                        hide_index=True,
                        key='w_sel_editor',
                        column_config={
                            'Select': st.column_config.CheckboxColumn('Select', default=False)
                        },
                        disabled=[c for c in df_w_select.columns if c != 'Select']
                    )
                    selected_rows = edited_sel[edited_sel['Select'] == True]
                    if len(selected_rows) > 0:
                        st.caption(f"{len(selected_rows)} row(s) selected")
                        if st.button(f"Delete {len(selected_rows)} Selected Row(s)", type="primary", key='w_del_rows'):
                            try:
                                for _, row in selected_rows.iterrows():
                                    conn.execute(
                                        "DELETE FROM FACT_WASTE WHERE waste_key IN ("
                                        "SELECT w.waste_key FROM FACT_WASTE w "
                                        "JOIN DIM_DATE d ON w.date_key=d.date_key "
                                        "JOIN DIM_ITEM i ON w.item_key=i.item_key "
                                        "JOIN DIM_WASTE_REASON r ON w.reason_key=r.reason_key "
                                        "WHERE d.date=? AND i.item_name=? AND r.reason_name=?)",
                                        (row.get('Date'), row.get('Item'), row.get('Waste Reason'))
                                    )
                                conn.commit()
                                st.success(f"Deleted {len(selected_rows)} record(s)!")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not delete: {e}")
                except Exception as e:
                    st.error(f"Could not load waste records: {e}")

            # ─────────────────────────────────────────────────────────────
            # TAB 3: INVENTORY (editable)
            # ─────────────────────────────────────────────────────────────
            with tab3:
                st.caption("Ingredient purchase records. You can update quantities and costs here.")

                try:
                    minmax_i = conn.execute(
                        "SELECT MIN(d.date), MAX(d.date) FROM FACT_INVENTORY inv "
                        "JOIN DIM_DATE d ON inv.date_key=d.date_key"
                    ).fetchone()
                    min_date_i = pd.to_datetime(minmax_i[0]).date() if minmax_i and minmax_i[0] else date(2020, 1, 1)
                    max_date_i = pd.to_datetime(minmax_i[1]).date() if minmax_i and minmax_i[1] else date.today()
                except Exception:
                    min_date_i, max_date_i = date(2020, 1, 1), date.today()
                try:
                    alerts = ['All'] + [r[0] for r in conn.execute(
                        "SELECT alert_name FROM DIM_ALERT_LEVEL ORDER BY alert_key"
                    ).fetchall()]
                except Exception:
                    alerts = ['All']

                f1, f2 = st.columns([2, 1])
                with f1:
                    i_date_range = st.date_input(
                        "Date Range", value=(min_date_i, max_date_i),
                        min_value=min_date_i, max_value=max_date_i, key='i_daterange'
                    )
                with f2:
                    i_alert = st.selectbox("Alert Level", alerts, key='i_alert')

                iw = []
                if isinstance(i_date_range, tuple) and len(i_date_range) == 2:
                    iw.append(f"d.date BETWEEN '{i_date_range[0]}' AND '{i_date_range[1]}'")
                elif isinstance(i_date_range, date):
                    iw.append(f"d.date = '{i_date_range}'")
                if i_alert   != 'All': iw.append(f"al.alert_name = '{i_alert}'")
                iw_sql = ("WHERE " + " AND ".join(iw)) if iw else ""

                iq = (
                    "SELECT substr(d.date,1,10) AS [Purchase Date], "
                    "substr(inv.expiration_date,1,10) AS [Expiry Date], "
                    "ing.ingredient_name AS [Ingredient], ing.unit AS [Unit], "
                    "inv.quantity AS [Quantity], "
                    "inv.cost_per_unit AS [Cost per Unit], "
                    "inv.total_cost AS [Total Cost], "
                    "al.alert_name AS [Alert Level], "
                    "CASE WHEN al.alert_name = 'Expired' THEN 'Discard' "
                    "     WHEN al.alert_name = 'Out of Stock' THEN 'Restock' "
                    "     ELSE '—' END AS [Action] "
                    "FROM FACT_INVENTORY inv "
                    "JOIN DIM_DATE d ON inv.date_key=d.date_key "
                    "JOIN DIM_INGREDIENT ing ON inv.ingredient_key=ing.ingredient_key "
                    "JOIN DIM_ALERT_LEVEL al ON inv.alert_key=al.alert_key "
                    f"{iw_sql} ORDER BY d.date DESC"
                )
                try:
                    total_i = conn.execute(f"SELECT COUNT(*) FROM ({iq})").fetchone()[0]
                    n_i = st.slider("Records to show", 10, 500, 50, 10, key='i_rows')
                    df_i = pd.read_sql(f"{iq} LIMIT {n_i}", conn)
                    st.caption(f"Showing {len(df_i):,} of {total_i:,} records")

                    edited_i = st.data_editor(
                        df_i, use_container_width=True, hide_index=True,
                        key='i_editor', num_rows="dynamic",
                        disabled=["Purchase Date","Expiry Date","Ingredient","Unit","Alert Level","Action"]
                    )

                    ic1, ic2, ic3 = st.columns([1, 1, 2])
                    with ic1:
                        if st.button("Save Changes", type="primary", key='i_save'):
                            try:
                                for _, row in edited_i.iterrows():
                                    # Days-until-expiration is no longer shown as an
                                    # editable column, so recompute it from Expiry
                                    # Date (still clean, e.g. '2026-06-08') instead
                                    # of trusting a value the user can't see/edit.
                                    try:
                                        _days_left = (pd.to_datetime(row.get('Expiry Date')) - pd.Timestamp.now().normalize()).days
                                    except Exception:
                                        _days_left = None
                                    conn.execute(
                                        "UPDATE FACT_INVENTORY SET "
                                        "quantity=?, cost_per_unit=?, total_cost=?, days_until_expiration=? "
                                        "WHERE inventory_key IN ("
                                        "SELECT inv.inventory_key FROM FACT_INVENTORY inv "
                                        "JOIN DIM_DATE d ON inv.date_key=d.date_key "
                                        "JOIN DIM_INGREDIENT ing ON inv.ingredient_key=ing.ingredient_key "
                                        "WHERE d.date=? AND ing.ingredient_name=?)",
                                        (row.get('Quantity'), row.get('Cost per Unit'),
                                         row.get('Total Cost'), _days_left,
                                         row.get('Purchase Date'), row.get('Ingredient'))
                                    )
                                conn.commit()
                                st.success("Changes saved successfully!")
                            except Exception as e:
                                st.error(f"Could not save: {e}")
                    with ic2:
                        # Delete by current filter — same freeze-before-confirm
                        # safety pattern as the Waste tab (see comment there).
                        if iw_sql:
                            inv_del_label = f"Delete filtered ({total_i:,} records)"
                        else:
                            inv_del_label = "Delete All Records"

                        if st.button(inv_del_label, key='i_del_filter'):
                            st.session_state['i_confirm_del']      = True
                            st.session_state['i_del_sql_frozen']   = iw_sql
                            st.session_state['i_del_count_frozen'] = total_i

                        if st.session_state.get('i_confirm_del', False):
                            frozen_sql_i   = st.session_state.get('i_del_sql_frozen', iw_sql)
                            frozen_count_i = st.session_state.get('i_del_count_frozen', total_i)

                            if not frozen_sql_i:
                                st.error(
                                    "⚠️ No filter is applied — this would delete "
                                    f"ALL {frozen_count_i:,} inventory records. Type "
                                    "DELETE ALL below to confirm, or Cancel."
                                )
                                confirm_text_i = st.text_input(
                                    "Type DELETE ALL to confirm", key='i_del_all_text'
                                )
                                proceed_allowed_i = (confirm_text_i.strip() == "DELETE ALL")
                            else:
                                st.warning(
                                    f"This will permanently delete **{frozen_count_i:,} records** "
                                    f"matching the filter you selected. This cannot be undone."
                                )
                                proceed_allowed_i = True

                            icd1, icd2 = st.columns(2)
                            with icd1:
                                if st.button("Delete", type="primary", key='i_del_confirm',
                                             use_container_width=True, disabled=not proceed_allowed_i):
                                    try:
                                        conn.execute(
                                            "DELETE FROM FACT_INVENTORY WHERE inventory_key IN ("
                                            "SELECT inv.inventory_key FROM FACT_INVENTORY inv "
                                            "JOIN DIM_DATE d ON inv.date_key=d.date_key "
                                            "JOIN DIM_INGREDIENT ing ON inv.ingredient_key=ing.ingredient_key "
                                            "JOIN DIM_ALERT_LEVEL al ON inv.alert_key=al.alert_key "
                                            f"{frozen_sql_i})"
                                        )
                                        conn.commit()
                                        st.session_state['i_confirm_del'] = False
                                        st.success(f"Deleted {frozen_count_i:,} records successfully!")
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not delete: {e}")
                            with icd2:
                                if st.button("Cancel", key='i_del_cancel', use_container_width=True):
                                    st.session_state['i_confirm_del'] = False
                                    st.rerun()

                    with ic3:
                        st.download_button(
                            "Download Inventory as CSV",
                            data=df_i.to_csv(index=False).encode("utf-8"),
                            file_name="inventory_records.csv", mime="text/csv", key="dl_i"
                        )

                    # ── Delete individual selected rows ────────────────────
                    st.markdown("---")
                    st.markdown("**Delete Individual Records**")
                    st.caption("Select rows to delete by checking the box on the left.")

                    df_i_select = df_i.copy()
                    df_i_select.insert(0, 'Select', False)
                    edited_i_sel = st.data_editor(
                        df_i_select,
                        use_container_width=True,
                        hide_index=True,
                        key='i_sel_editor',
                        column_config={
                            'Select': st.column_config.CheckboxColumn('Select', default=False)
                        },
                        disabled=[c for c in df_i_select.columns if c != 'Select']
                    )
                    sel_inv_rows = edited_i_sel[edited_i_sel['Select'] == True]
                    if len(sel_inv_rows) > 0:
                        st.caption(f"{len(sel_inv_rows)} row(s) selected")
                        if st.button(f"Delete {len(sel_inv_rows)} Selected Row(s)", type="primary", key='i_del_rows'):
                            try:
                                for _, row in sel_inv_rows.iterrows():
                                    conn.execute(
                                        "DELETE FROM FACT_INVENTORY WHERE inventory_key IN ("
                                        "SELECT inv.inventory_key FROM FACT_INVENTORY inv "
                                        "JOIN DIM_DATE d ON inv.date_key=d.date_key "
                                        "JOIN DIM_INGREDIENT ing ON inv.ingredient_key=ing.ingredient_key "
                                        "WHERE d.date=? AND ing.ingredient_name=?)",
                                        (row.get('Purchase Date'), row.get('Ingredient'))
                                    )
                                conn.commit()
                                st.success(f"Deleted {len(sel_inv_rows)} record(s)!")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not delete: {e}")
                except Exception as e:
                    st.error(f"Could not load inventory: {e}")

            # ─────────────────────────────────────────────────────────────
            # TAB 4: MENU ITEMS (editable — now that Menu has an upload feature)
            # ─────────────────────────────────────────────────────────────
            with tab4:
                st.caption("List of all menu items and their pricing information. You can correct or remove entries here.")
                try:
                    df_menu = pd.read_sql(
                        "SELECT item_name AS [Item], category AS [Category], "
                        "price AS [Price], cost AS [Cost], "
                        "profit_margin AS [Profit Margin] "
                        "FROM DIM_ITEM ORDER BY category, item_name",
                        conn
                    )
                    st.caption(f"{len(df_menu):,} menu items")

                    edited_menu = st.data_editor(
                        df_menu, use_container_width=True, hide_index=True,
                        key='menu_editor', num_rows="dynamic",
                        disabled=["Item", "Category", "Profit Margin"]
                    )

                    mc1, mc2, mc3 = st.columns([1, 1, 2])
                    with mc1:
                        if st.button("Save Changes", type="primary", key='menu_save'):
                            try:
                                for _, row in edited_menu.iterrows():
                                    price = row.get('Price')
                                    cost  = row.get('Cost')
                                    margin = (price - cost) if (pd.notna(price) and pd.notna(cost)) else None
                                    conn.execute(
                                        "UPDATE DIM_ITEM SET price=?, cost=?, profit_margin=? WHERE item_name=?",
                                        (price, cost, margin, row.get('Item'))
                                    )
                                conn.commit()
                                st.success("Changes saved successfully!")
                                st.cache_data.clear()
                            except Exception as e:
                                st.error(f"Could not save: {e}")
                    with mc2:
                        # Same freeze-before-confirm safety pattern as Sales/Waste/Inventory
                        if st.button("Delete All Records", key='menu_del_all'):
                            st.session_state['menu_confirm_del']      = True
                            st.session_state['menu_del_count_frozen'] = len(df_menu)

                        if st.session_state.get('menu_confirm_del', False):
                            frozen_count_menu = st.session_state.get('menu_del_count_frozen', len(df_menu))
                            st.error(
                                "⚠️ This would delete ALL "
                                f"{frozen_count_menu:,} menu items. Type "
                                "DELETE ALL below to confirm, or Cancel."
                            )
                            confirm_text_menu = st.text_input(
                                "Type DELETE ALL to confirm", key='menu_del_all_text'
                            )
                            proceed_allowed_menu = (confirm_text_menu.strip() == "DELETE ALL")

                            mcd1, mcd2 = st.columns(2)
                            with mcd1:
                                if st.button("Delete", type="primary", key='menu_del_confirm',
                                             use_container_width=True, disabled=not proceed_allowed_menu):
                                    try:
                                        conn.execute("DELETE FROM DIM_ITEM")
                                        conn.commit()
                                        st.session_state['menu_confirm_del'] = False
                                        st.success(f"Deleted {frozen_count_menu:,} menu items successfully!")
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not delete: {e}")
                            with mcd2:
                                if st.button("Cancel", key='menu_del_cancel', use_container_width=True):
                                    st.session_state['menu_confirm_del'] = False
                                    st.rerun()
                    with mc3:
                        st.download_button(
                            "Download Menu Items as CSV",
                            data=df_menu.to_csv(index=False).encode("utf-8"),
                            file_name="menu_items.csv", mime="text/csv", key="dl_menu"
                        )

                    # ── Delete individual selected rows ────────────────────
                    st.markdown("---")
                    st.markdown("**Delete Individual Records**")
                    st.caption(
                        "Select rows to delete by checking the box on the left. "
                        "⚠️ Deleting an item also hides its past sales/waste records "
                        "from the Sales and Waste Records tabs (they're linked by item name)."
                    )

                    df_menu_select = df_menu.copy()
                    df_menu_select.insert(0, 'Select', False)
                    edited_menu_sel = st.data_editor(
                        df_menu_select,
                        use_container_width=True,
                        hide_index=True,
                        key='menu_sel_editor',
                        column_config={
                            'Select': st.column_config.CheckboxColumn('Select', default=False)
                        },
                        disabled=[c for c in df_menu_select.columns if c != 'Select']
                    )
                    selected_menu_rows = edited_menu_sel[edited_menu_sel['Select'] == True]
                    if len(selected_menu_rows) > 0:
                        st.caption(f"{len(selected_menu_rows)} item(s) selected")
                        if st.button(f"Delete {len(selected_menu_rows)} Selected Item(s)", type="primary", key='menu_del_rows'):
                            try:
                                for _, row in selected_menu_rows.iterrows():
                                    conn.execute("DELETE FROM DIM_ITEM WHERE item_name=?", (row.get('Item'),))
                                conn.commit()
                                st.success(f"Deleted {len(selected_menu_rows)} item(s)!")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not delete: {e}")
                except Exception as e:
                    st.error(f"Could not load menu items: {e}")

        conn.close()
