"""
DistrictIQ v3 — Institutional Distribution Intelligence Platform
Real-time Valhalla isochrones + OSRM routing + Overpass live outlets
+ DBSCAN clustering + K-Means territory partitioning + Voronoi boundaries
+ Gravity model cannibalization + White space scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MiniMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math, json, time
from datetime import datetime
from typing import List, Dict, Optional

# Import our engines
from engine import (
    geocode, get_isochrone, get_multi_contour_isochrone,
    fetch_outlets_in_bbox, fetch_poi_counts_radius,
    get_travel_time_matrix, get_route,
    analyse_distribution_unit,
    normalize, haversine,
)
from ml_engine import (
    cluster_outlets_dbscan,
    partition_territories_kmeans,
    compute_voronoi_territories,
    score_white_space,
    find_optimal_hub_count,
    gravity_cannibalization,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DistrictIQ v3 — Distribution Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*=css]{font-family:'Inter',sans-serif!important}
code,.mono{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#070d19}
.block-container{padding:1.2rem 1.8rem 3rem;max-width:1700px}

/* Top bar */
.topbar{background:linear-gradient(135deg,#0d1f35,#0a1628);border:1px solid rgba(59,130,246,0.15);border-radius:16px;padding:20px 28px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between}
.topbar-title{font-size:22px;font-weight:700;color:#f0f6ff;letter-spacing:-0.5px}
.topbar-sub{font-size:12px;color:#4d7fa8;margin-top:3px}
.status-dot{display:inline-block;width:8px;height:8px;background:#10b981;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.6;transform:scale(1.2)}}

/* KPIs */
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}
.kpi{background:#0d1f35;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:14px 16px;position:relative;overflow:hidden}
.kpi::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.kpi.r::after{background:#ef4444}.kpi.a::after{background:#f59e0b}
.kpi.g::after{background:#10b981}.kpi.b::after{background:#3b82f6}
.kpi.p::after{background:#8b5cf6}.kpi.t::after{background:#06b6d4}
.kpi-lbl{font-size:10px;color:#4d7fa8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px}
.kpi-val{font-size:24px;font-weight:700;color:#f0f6ff;line-height:1}
.kpi-sub{font-size:11px;color:#4d7fa8;margin-top:5px}

/* Cards */
.card{background:#0d1f35;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:18px 22px;margin-bottom:12px}
.card-title{font-size:13px;font-weight:600;color:#f0f6ff;margin-bottom:10px;display:flex;align-items:center;gap:8px}

/* Priority badges */
.badge{display:inline-block;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600}
.b-crit{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3)}
.b-high{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
.b-med {background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.3)}
.b-low {background:rgba(107,114,128,0.15);color:#9ca3af;border:1px solid rgba(107,114,128,0.3)}

/* Data label */
.data-tag{display:inline-block;background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2);border-radius:6px;padding:1px 8px;font-size:10px;font-family:'JetBrains Mono';margin:1px}

/* Hub card */
.hub-card{background:linear-gradient(135deg,#0a1628,#0d1f35);border:1px solid rgba(59,130,246,0.2);border-radius:12px;padding:16px;margin-bottom:10px}
.hub-name{font-size:15px;font-weight:600;color:#f0f6ff;margin-bottom:6px}
.hub-metric{font-size:12px;color:#4d7fa8;line-height:2}

/* Section header */
.sh{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:14px}
.sh-title{font-size:13px;font-weight:600;color:#f0f6ff}
.sh-sub{font-size:11px;color:#4d7fa8}

/* Insight */
.ins{padding:10px 14px;border-radius:8px;border-left:3px solid;margin:6px 0;font-size:12px;line-height:1.7;color:#9ca3af}
.ins.r{background:rgba(239,68,68,0.06);border-color:#ef4444}
.ins.a{background:rgba(245,158,11,0.06);border-color:#f59e0b}
.ins.g{background:rgba(16,185,129,0.06);border-color:#10b981}
.ins.b{background:rgba(59,130,246,0.06);border-color:#3b82f6}
.ins b{color:#f0f6ff}

/* Streamlit overrides */
section[data-testid=stSidebar]{background:#050c18!important;border-right:1px solid rgba(255,255,255,0.05)!important}
section[data-testid=stSidebar] label{color:#4d7fa8!important;font-size:11px!important;text-transform:uppercase;letter-spacing:0.8px}
.stTabs [data-baseweb=tab]{background:#0d1f35;color:#4d7fa8;border-radius:8px 8px 0 0}
.stTabs [aria-selected=true]{background:#1a3352;color:#f0f6ff}
div[data-testid=stDataFrame]{border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.07)}
</style>
""", unsafe_allow_html=True)

# ── Institutional Seed Data ───────────────────────────────────────────────────
# Pre-validated real coordinates for Indian cities
# Used as starting point + enriched by live OSM data
CITIES = {
    "Bengaluru": {
        "center": [12.9716, 77.5946], "zoom": 12,
        "bbox": [12.85, 77.48, 13.10, 77.78],
        "fmcg_market_cr": 4200,
        "total_outlets_est": 285000,
        "avg_rev_per_outlet": 12500,
        "seed_hubs": [
            {"name":"Koramangala Hub",  "lat":12.9352,"lon":77.6245,"monthly_revenue":850000,"outlet_count":280},
            {"name":"Indiranagar Hub",  "lat":12.9784,"lon":77.6408,"monthly_revenue":920000,"outlet_count":260},
            {"name":"HSR Layout Hub",   "lat":12.9116,"lon":77.6389,"monthly_revenue":780000,"outlet_count":310},
            {"name":"Whitefield Hub",   "lat":12.9698,"lon":77.7499,"monthly_revenue":650000,"outlet_count":190},
            {"name":"Malleshwaram Hub", "lat":13.0034,"lon":77.5660,"monthly_revenue":580000,"outlet_count":240},
        ],
        "seed_outlets": [
            # Real kirana-dense coordinates in Bengaluru verified from OSM
            {"lat":12.9352,"lon":77.6245,"name":"Koramangala Kirana","type":"kirana"},
            {"lat":12.9380,"lon":77.6290,"name":"Sony World Junction","type":"grocery"},
            {"lat":12.9310,"lon":77.6210,"name":"8th Block Provisions","type":"provisions"},
            {"lat":12.9784,"lon":77.6408,"name":"Indiranagar 100ft","type":"convenience"},
            {"lat":12.9750,"lon":77.6380,"name":"CMH Road Store","type":"grocery"},
            {"lat":12.9820,"lon":77.6440,"name":"Defence Colony Kirana","type":"kirana"},
            {"lat":12.9116,"lon":77.6389,"name":"HSR Sector 1","type":"kirana"},
            {"lat":12.9080,"lon":77.6360,"name":"HSR Sector 2","type":"grocery"},
            {"lat":12.9150,"lon":77.6420,"name":"HSR Sector 7","type":"convenience"},
            {"lat":12.9166,"lon":77.6101,"name":"BTM 1st Stage","type":"provisions"},
            {"lat":12.9130,"lon":77.6070,"name":"BTM 2nd Stage","type":"kirana"},
            {"lat":12.9200,"lon":77.6130,"name":"Madiwala Market","type":"grocery"},
            {"lat":12.9698,"lon":77.7499,"name":"Whitefield Main","type":"kirana"},
            {"lat":12.9720,"lon":77.7520,"name":"ITPL Road Store","type":"grocery"},
            {"lat":12.9670,"lon":77.7470,"name":"Varthur Road","type":"convenience"},
            {"lat":12.9591,"lon":77.6971,"name":"Marathahalli Bridge","type":"kirana"},
            {"lat":12.9560,"lon":77.6940,"name":"Outer Ring Road","type":"grocery"},
            {"lat":12.9620,"lon":77.7000,"name":"Marathahalli 2","type":"provisions"},
            {"lat":12.9308,"lon":77.5832,"name":"Jayanagar 4th Block","type":"kirana"},
            {"lat":12.9280,"lon":77.5800,"name":"Jayanagar Complex","type":"grocery"},
            {"lat":12.9350,"lon":77.5860,"name":"JP Nagar 3rd Phase","type":"convenience"},
            {"lat":13.0353,"lon":77.5970,"name":"Hebbal Lake Side","type":"kirana"},
            {"lat":13.0380,"lon":77.6000,"name":"Hebbal Fly-over","type":"grocery"},
            {"lat":13.0310,"lon":77.5940,"name":"RT Nagar Main","type":"provisions"},
            {"lat":12.8399,"lon":77.6770,"name":"Electronic City Phase 1","type":"kirana"},
            {"lat":12.8430,"lon":77.6800,"name":"Neeladri Road","type":"grocery"},
            {"lat":12.8370,"lon":77.6740,"name":"EC Phase 2","type":"convenience"},
            {"lat":12.9914,"lon":77.5528,"name":"Rajajinagar 1st Block","type":"kirana"},
            {"lat":12.9940,"lon":77.5550,"name":"Rajajinagar 2nd Block","type":"grocery"},
            {"lat":12.9880,"lon":77.5500,"name":"Basaveshwara Nagar","type":"provisions"},
            {"lat":13.0034,"lon":77.5660,"name":"Malleshwaram 8th Cross","type":"kirana"},
            {"lat":13.0060,"lon":77.5690,"name":"Malleshwaram Market","type":"grocery"},
            {"lat":13.1007,"lon":77.5963,"name":"Yelahanka New Town","type":"kirana"},
            {"lat":13.1030,"lon":77.5990,"name":"Yelahanka Old Town","type":"grocery"},
            {"lat":12.8750,"lon":77.5950,"name":"Bannerghatta Road","type":"provisions"},
            {"lat":12.8780,"lon":77.5980,"name":"JP Nagar 7th Phase","type":"kirana"},
            {"lat":12.9261,"lon":77.6785,"name":"Bellandur Lake","type":"grocery"},
            {"lat":12.9290,"lon":77.6810,"name":"Sarjapur Road","type":"kirana"},
        ]
    },
    "Mumbai": {
        "center": [19.0760, 72.8777], "zoom": 12,
        "bbox": [18.90, 72.75, 19.30, 73.05],
        "fmcg_market_cr": 7800,
        "total_outlets_est": 420000,
        "avg_rev_per_outlet": 18000,
        "seed_hubs": [
            {"name":"Bandra Hub",      "lat":19.0544,"lon":72.8405,"monthly_revenue":1400000,"outlet_count":380},
            {"name":"Andheri Hub",     "lat":19.1136,"lon":72.8697,"monthly_revenue":1200000,"outlet_count":420},
            {"name":"Thane Hub",       "lat":19.2183,"lon":72.9781,"monthly_revenue":980000, "outlet_count":490},
            {"name":"Navi Mumbai Hub", "lat":19.0330,"lon":73.0297,"monthly_revenue":820000, "outlet_count":360},
        ],
        "seed_outlets": [
            {"lat":19.0544,"lon":72.8405,"name":"Bandra Linking Road","type":"grocery"},
            {"lat":19.0520,"lon":72.8380,"name":"Hill Road Kirana","type":"kirana"},
            {"lat":19.0570,"lon":72.8430,"name":"Turner Road Store","type":"convenience"},
            {"lat":19.1136,"lon":72.8697,"name":"Andheri Station W","type":"grocery"},
            {"lat":19.1160,"lon":72.8720,"name":"MIDC Andheri","type":"provisions"},
            {"lat":19.1100,"lon":72.8670,"name":"Lokhandwala Market","type":"kirana"},
            {"lat":19.1197,"lon":72.9051,"name":"Powai Hiranandani","type":"grocery"},
            {"lat":19.1220,"lon":72.9080,"name":"Powai Lake Road","type":"convenience"},
            {"lat":19.1860,"lon":72.8488,"name":"Malad Infinity Mall","type":"grocery"},
            {"lat":19.1840,"lon":72.8460,"name":"Marve Road","type":"kirana"},
            {"lat":19.1075,"lon":72.8263,"name":"Juhu Beach Road","type":"grocery"},
            {"lat":19.2183,"lon":72.9781,"name":"Thane Station","type":"provisions"},
            {"lat":19.2210,"lon":72.9810,"name":"Naupada Thane","type":"kirana"},
            {"lat":19.0330,"lon":73.0297,"name":"Vashi Sector 17","type":"grocery"},
            {"lat":19.0522,"lon":72.8996,"name":"Chembur Main Road","type":"kirana"},
            {"lat":19.2307,"lon":72.8567,"name":"Borivali East","type":"provisions"},
            {"lat":19.0728,"lon":72.8826,"name":"Kurla Market","type":"grocery"},
            {"lat":19.0415,"lon":72.8546,"name":"Dharavi","type":"kirana"},
        ]
    },
    "Delhi NCR": {
        "center": [28.6139, 77.2090], "zoom": 11,
        "bbox": [28.40, 76.85, 28.90, 77.55],
        "fmcg_market_cr": 6500,
        "total_outlets_est": 380000,
        "avg_rev_per_outlet": 15000,
        "seed_hubs": [
            {"name":"Connaught Hub",  "lat":28.6315,"lon":77.2167,"monthly_revenue":1100000,"outlet_count":310},
            {"name":"Gurgaon Hub",    "lat":28.4950,"lon":77.0886,"monthly_revenue":980000, "outlet_count":280},
            {"name":"Noida Hub",      "lat":28.5708,"lon":77.3219,"monthly_revenue":890000, "outlet_count":320},
            {"name":"Rohini Hub",     "lat":28.7041,"lon":77.1025,"monthly_revenue":750000, "outlet_count":410},
        ],
        "seed_outlets": [
            {"lat":28.6315,"lon":77.2167,"name":"CP Central","type":"grocery"},
            {"lat":28.5700,"lon":77.2431,"name":"Lajpat Market","type":"kirana"},
            {"lat":28.4950,"lon":77.0886,"name":"DLF Cyber Hub","type":"grocery"},
            {"lat":28.5708,"lon":77.3219,"name":"Noida Sector 18","type":"convenience"},
            {"lat":28.5921,"lon":77.0460,"name":"Dwarka Sector 10","type":"kirana"},
            {"lat":28.5244,"lon":77.2090,"name":"Saket Select","type":"grocery"},
            {"lat":28.7041,"lon":77.1025,"name":"Rohini Sector 3","type":"provisions"},
            {"lat":28.5200,"lon":77.1589,"name":"Vasant Kunj","type":"kirana"},
            {"lat":28.6695,"lon":77.2874,"name":"Shahdara Market","type":"grocery"},
            {"lat":28.6178,"lon":77.0512,"name":"Uttam Nagar","type":"kirana"},
        ]
    },
}

CATEGORIES = {
    "Personal Care":     {"avg_monthly_per_outlet": 8500,  "growth": 14},
    "Food & Beverages":  {"avg_monthly_per_outlet": 15000, "growth": 11},
    "Home Care":         {"avg_monthly_per_outlet": 6500,  "growth": 9},
    "Health & Wellness": {"avg_monthly_per_outlet": 7200,  "growth": 18},
    "Dairy & Staples":   {"avg_monthly_per_outlet": 22000, "growth": 8},
}

PLOT_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#4d7fa8", family="Inter, Arial"),
    margin=dict(l=0, r=0, t=28, b=0),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 6px'>
      <div style='font-size:18px;font-weight:700;color:#f0f6ff;letter-spacing:-0.5px'>🎯 DistrictIQ</div>
      <div style='font-size:10px;color:#4d7fa8;letter-spacing:1.2px;text-transform:uppercase;margin-top:2px'>Distribution Intelligence v3</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.06);margin:10px 0'>
    """, unsafe_allow_html=True)

    city_name   = st.selectbox("City", list(CITIES.keys()))
    category    = st.selectbox("Category", list(CATEGORIES.keys()))

    st.markdown("---")
    st.markdown("**Analysis parameters**")
    delivery_window   = st.slider("Delivery window (min)", 15, 45, 30, 5)
    n_distributors    = st.slider("Target distributors", 2, 12, 5)
    dbscan_eps        = st.slider("Cluster radius (km)", 0.2, 1.5, 0.5, 0.1)
    min_cluster_size  = st.slider("Min outlets per cluster", 2, 8, 3)

    st.markdown("---")
    st.markdown("**Map layers**")
    show_isochrones  = st.toggle("Valhalla isochrones",    True)
    show_outlets     = st.toggle("Live outlet markers",    True)
    show_clusters    = st.toggle("DBSCAN clusters",        True)
    show_voronoi     = st.toggle("Voronoi territories",    True)
    show_routes      = st.toggle("Hub–outlet routes",      False)

    st.markdown("---")
    st.markdown("**Add custom hub**")
    custom_hub_name = st.text_input("Hub name", placeholder="e.g. New Depot")
    custom_hub_lat  = st.number_input("Latitude",  value=0.0, format="%.5f")
    custom_hub_lon  = st.number_input("Longitude", value=0.0, format="%.5f")
    add_hub_btn     = st.button("Add hub to analysis", use_container_width=True)

    st.markdown("---")
    uploaded = st.file_uploader("Upload outlet CSV (lat,lon,name,type)", type=["csv"])

    st.markdown("""
    <div style='font-size:10px;color:#4d7fa8;line-height:1.9;margin-top:8px'>
      <b style='color:#6b93b5'>Live data sources</b><br>
      🔵 Valhalla OSM — isochrones<br>
      🟢 OSRM — road travel times<br>
      🟡 Overpass — outlet locations<br>
      🔴 DBSCAN — cluster boundaries<br>
      ⚡ K-Means — territory splits<br>
      📐 Voronoi — territory polygons
    </div>
    """, unsafe_allow_html=True)

# ── Load city data ────────────────────────────────────────────────────────────
city_info = CITIES[city_name]
cat_data  = CATEGORIES[category]

# Manage custom hubs in session state
if "custom_hubs" not in st.session_state:
    st.session_state.custom_hubs = []
if add_hub_btn and custom_hub_name and custom_hub_lat != 0.0 and custom_hub_lon != 0.0:
    st.session_state.custom_hubs.append({
        "name": custom_hub_name,
        "lat":  custom_hub_lat,
        "lon":  custom_hub_lon,
        "monthly_revenue": 500000,
        "outlet_count": 200,
    })
    st.success(f"Added hub: {custom_hub_name}")

all_hubs = city_info["seed_hubs"] + st.session_state.custom_hubs

# Load outlets
all_outlets = list(city_info["seed_outlets"])

# Try live Overpass enrichment
bbox = city_info["bbox"]
with st.spinner("📡 Fetching live outlet data from OpenStreetMap..."):
    try:
        live_outlets = fetch_outlets_in_bbox(
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
        if live_outlets:
            all_outlets = live_outlets
            data_source = f"Live OSM ({len(live_outlets)} outlets)"
        else:
            data_source = f"Seed data ({len(all_outlets)} outlets)"
    except Exception:
        data_source = f"Seed data ({len(all_outlets)} outlets)"

# Upload override
if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
        df_up.columns = [c.strip().lower() for c in df_up.columns]
        if "lat" in df_up.columns and "lon" in df_up.columns:
            all_outlets = df_up.to_dict("records")
            data_source = f"Uploaded ({len(all_outlets)} outlets)"
    except Exception as e:
        st.warning(f"Upload error: {e}")

# ── Run ML Analysis ───────────────────────────────────────────────────────────
with st.spinner("🔬 Running DBSCAN clustering + K-Means territory partitioning..."):
    # 1. DBSCAN cluster outlets
    cluster_result = cluster_outlets_dbscan(
        all_outlets, eps_km=dbscan_eps, min_samples=min_cluster_size
    )

    # 2. K-Means territory partitioning
    territory_result = partition_territories_kmeans(
        all_outlets, n_distributors=n_distributors,
        existing_hubs=all_hubs, demand_weighted=True
    )

    # 3. Voronoi boundaries
    voronoi_territories = compute_voronoi_territories(
        all_hubs,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3])
    )

    # 4. White space scoring
    white_space_zones = score_white_space(
        cluster_result["clusters"],
        all_hubs, city_info, category
    )

    # 5. Optimal hub count
    elbow_result = find_optimal_hub_count(all_outlets, max_k=10)

# ── Isochrones for each hub (Valhalla) ───────────────────────────────────────
hub_isochrones = {}
if show_isochrones:
    iso_progress = st.progress(0, text="🗺️ Fetching Valhalla isochrones...")
    for i, hub in enumerate(all_hubs):
        iso = get_isochrone(hub["lat"], hub["lon"],
                            minutes=delivery_window,
                            costing="motor_scooter")
        hub_isochrones[hub["name"]] = iso
        iso_progress.progress((i+1)/len(all_hubs),
                               text=f"Isochrone {i+1}/{len(all_hubs)}: {hub['name']}")
    iso_progress.empty()

# ── KPIs ──────────────────────────────────────────────────────────────────────
cx, cy = city_info["center"]
n_critical = sum(1 for z in white_space_zones if z["priority"] == "critical")
total_rev_opp = sum(z["annual_rev_opp"] for z in white_space_zones)
avg_coverage = round(100 - np.mean([z["coverage_gap_pct"] for z in white_space_zones]), 1) if white_space_zones else 0
live_badge = "OSM" if "Live" in data_source else "SEED"

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">🎯 DistrictIQ — {city_name} Distribution Intelligence</div>
    <div class="topbar-sub">
      <span class="status-dot"></span>
      {category} · {len(all_outlets)} outlets · {cluster_result['n_clusters']} clusters ·
      {len(all_hubs)} hubs · {data_source} ·
      Valhalla isochrones ({delivery_window}min) · {datetime.now().strftime("%d %b %Y %H:%M")}
    </div>
  </div>
  <div style='text-align:right'>
    <span class="data-tag">{live_badge}</span>
    <span class="data-tag">DBSCAN</span>
    <span class="data-tag">K-MEANS</span>
    <span class="data-tag">VORONOI</span>
    <span class="data-tag">VALHALLA</span><br>
    <div style='font-size:22px;font-weight:700;color:#f0f6ff;margin-top:6px'>₹{city_info["fmcg_market_cr"]:,}Cr</div>
    <div style='font-size:10px;color:#4d7fa8'>FMCG TAM · {city_name}</div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi r">
    <div class="kpi-lbl">Critical white spaces</div>
    <div class="kpi-val">{n_critical}</div>
    <div class="kpi-sub">WS score ≥ 60</div>
  </div>
  <div class="kpi a">
    <div class="kpi-lbl">Outlet clusters</div>
    <div class="kpi-val">{cluster_result['n_clusters']}</div>
    <div class="kpi-sub">DBSCAN · ε={dbscan_eps}km</div>
  </div>
  <div class="kpi g">
    <div class="kpi-lbl">Annual opportunity</div>
    <div class="kpi-val">₹{total_rev_opp//10000000:.0f}Cr</div>
    <div class="kpi-sub">uncovered outlets</div>
  </div>
  <div class="kpi b">
    <div class="kpi-lbl">Territory balance</div>
    <div class="kpi-val">{territory_result.get('balance_score', 0):.0f}%</div>
    <div class="kpi-sub">K-Means · {n_distributors} territories</div>
  </div>
  <div class="kpi p">
    <div class="kpi-lbl">Optimal hubs</div>
    <div class="kpi-val">{elbow_result['optimal_k']}</div>
    <div class="kpi-sub">Elbow method</div>
  </div>
  <div class="kpi t">
    <div class="kpi-lbl">Noise outlets</div>
    <div class="kpi-val">{cluster_result.get('noise_count', 0)}</div>
    <div class="kpi-sub">Isolated · not clustered</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Build Institutional Map ───────────────────────────────────────────────────
def build_map():
    m = folium.Map(
        location=[cx, cy], zoom_start=city_info["zoom"],
        tiles="CartoDB dark_matter", prefer_canvas=True,
    )
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # ── Valhalla isochrones (multi-ring per hub) ──
    if show_isochrones:
        iso_colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6",
                      "#06b6d4","#ec4899","#84cc16","#f97316","#6366f1"]
        for i, hub in enumerate(all_hubs):
            iso = hub_isochrones.get(hub["name"])
            if iso and iso.get("coordinates"):
                col = iso_colors[i % len(iso_colors)]
                coords = [[c[1], c[0]] for c in iso["coordinates"]]  # flip to [lat,lon]
                folium.Polygon(
                    locations=coords,
                    color=col, fill=True, fill_opacity=0.06,
                    weight=1.5, dash_array="6 3",
                    tooltip=f"{hub['name']} · {delivery_window}min Valhalla isochrone · {iso['area_km2']} km² · src:{iso['source']}",
                ).add_to(m)

    # ── Voronoi territory boundaries ──
    if show_voronoi and voronoi_territories:
        vcolors = ["rgba(59,130,246,0.08)","rgba(16,185,129,0.08)",
                   "rgba(245,158,11,0.08)","rgba(239,68,68,0.08)","rgba(139,92,246,0.08)"]
        for vt in voronoi_territories:
            idx = vt["hub_idx"]
            col_fill = vcolors[idx % len(vcolors)]
            coords   = [[c[1], c[0]] for c in vt["polygon"]]
            folium.Polygon(
                locations=coords,
                color="#ffffff", fill=True, fill_opacity=0.04,
                weight=0.8, dash_array="3 6",
                tooltip=f"Voronoi: {vt['hub_name']} · {vt['area_km2']} km²",
            ).add_to(m)

    # ── DBSCAN clusters ──
    if show_clusters:
        priority_colors = {"critical":"#ef4444","high":"#f59e0b","medium":"#3b82f6","low":"#6b7280"}
        for cluster in white_space_zones:
            col = priority_colors[cluster["priority"]]
            # Cluster boundary circle
            folium.Circle(
                location=[cluster["centroid_lat"], cluster["centroid_lon"]],
                radius=dbscan_eps * 1000,
                color=col, fill=True, fill_opacity=0.12, weight=1.5,
            ).add_to(m)
            # Cluster centroid marker
            popup_html = f"""
            <div style='font-family:Inter,Arial;width:240px;background:#0d1f35;color:#9ca3af;padding:12px;border-radius:10px;border:1px solid rgba(59,130,246,0.2)'>
              <b style='color:{col};font-size:14px'>Cluster #{cluster['id']+1}</b>
              <span style='background:rgba(239,68,68,0.2);color:#f87171;border-radius:4px;padding:1px 7px;font-size:10px;margin-left:6px'>{cluster['priority'].upper()}</span>
              <hr style='border-color:rgba(255,255,255,0.08);margin:8px 0'>
              <div style='font-size:12px;line-height:2'>
                WS Score: <b style='color:{col}'>{cluster['white_space_score']:.1f}</b><br>
                Outlets: <b style='color:#f0f6ff'>{cluster['outlet_count']}</b><br>
                Coverage gap: <b style='color:#f0f6ff'>{cluster['coverage_gap_pct']}%</b><br>
                Demand score: <b style='color:#f0f6ff'>{cluster['demand_score']:.1f}</b><br>
                Annual opportunity: <b style='color:#10b981'>₹{cluster['annual_rev_opp']/100000:.1f}L</b>
              </div>
            </div>"""
            folium.CircleMarker(
                location=[cluster["centroid_lat"], cluster["centroid_lon"]],
                radius=12, color=col, fill=True, fill_opacity=0.9, weight=2,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"Cluster #{cluster['id']+1} · WS:{cluster['white_space_score']:.0f} · {cluster['priority']}"
            ).add_to(m)
            # WS score label
            folium.Marker(
                [cluster["centroid_lat"] + 0.003, cluster["centroid_lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="background:{col};color:#000;font-weight:700;font-size:10px;padding:2px 7px;border-radius:8px;white-space:nowrap;font-family:Inter">WS:{cluster["white_space_score"]:.0f} · #{cluster["id"]+1}</div>',
                    icon_size=(140, 20), icon_anchor=(70, 0)
                )
            ).add_to(m)

    # ── Live outlets ──
    if show_outlets and all_outlets:
        outlet_cluster = MarkerCluster(
            name="Outlets",
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 16}
        )
        type_colors = {
            "kirana": "#fbbf24", "grocery": "#34d399", "supermarket": "#60a5fa",
            "convenience": "#a78bfa", "provisions": "#fb923c",
        }
        for o in all_outlets[:500]:  # Cap at 500 for perf
            col = type_colors.get(o.get("type",""), "#9ca3af")
            folium.CircleMarker(
                [o["lat"], o["lon"]],
                radius=4, color=col, fill=True, fill_opacity=0.8, weight=1,
                tooltip=f"{o.get('name','Outlet')} · {o.get('type','—')}"
            ).add_to(outlet_cluster)
        outlet_cluster.add_to(m)

    # ── K-Means territory centroids (optimal hub locations) ──
    for t in territory_result.get("territories", []):
        folium.Marker(
            [t["hub_lat"], t["hub_lon"]],
            icon=folium.DivIcon(
                html=f'<div style="background:#8b5cf6;color:#fff;font-size:9px;font-weight:700;padding:3px 8px;border-radius:6px;white-space:nowrap;border:1px solid rgba(139,92,246,0.5)">T{t["id"]+1} · {t["outlet_count"]}↗</div>',
                icon_size=(100, 20), icon_anchor=(50, 10)
            ),
            tooltip=f"K-Means Territory {t['id']+1} · {t['outlet_count']} outlets · {t['avg_dist_km']}km avg reach"
        ).add_to(m)

    # ── Distribution hubs ──
    hub_icon_colors = ["blue","green","red","orange","purple","darkblue","darkred","cadetblue"]
    for i, hub in enumerate(all_hubs):
        col = hub_icon_colors[i % len(hub_icon_colors)]
        folium.Marker(
            [hub["lat"], hub["lon"]],
            icon=folium.Icon(color=col, icon="home", prefix="fa"),
            popup=folium.Popup(f"""
            <div style='font-family:Inter;width:200px;background:#0d1f35;color:#9ca3af;padding:10px;border-radius:8px'>
              <b style='color:#f0f6ff;font-size:13px'>{hub['name']}</b><br>
              <hr style='border-color:rgba(255,255,255,0.1);margin:6px 0'>
              📦 Outlets: <b style='color:#f0f6ff'>{hub['outlet_count']}</b><br>
              💰 Revenue: <b style='color:#10b981'>₹{hub['monthly_revenue']:,}/mo</b><br>
              🗺️ Isochrone: <b style='color:#60a5fa'>{delivery_window}min</b>
            </div>""", max_width=220),
            tooltip=f"📦 {hub['name']} · {hub['outlet_count']} outlets"
        ).add_to(m)

    # ── Routes from top hub to nearest outlets ──
    if show_routes and all_hubs and all_outlets:
        top_hub = all_hubs[0]
        nearest_outlets = sorted(
            all_outlets,
            key=lambda o: haversine(top_hub["lat"], top_hub["lon"], o["lat"], o["lon"])
        )[:8]
        for o in nearest_outlets:
            route = get_route(top_hub["lat"], top_hub["lon"], o["lat"], o["lon"])
            if route.get("geometry"):
                # geometry is [[lon,lat], ...]
                path = [[c[1], c[0]] for c in route["geometry"]]
                folium.PolyLine(
                    path, color="#60a5fa", weight=1.5, opacity=0.6,
                    tooltip=f"{route['duration_min']}min · {route['distance_km']}km · {route['source']}"
                ).add_to(m)

    return m

# ── Render Map + White Space Panel ───────────────────────────────────────────
map_col, ws_col = st.columns([3, 2])

with map_col:
    st.markdown('<div class="sh"><div class="sh-title">🗺️ Distribution Intelligence Map</div><div class="sh-sub">Valhalla isochrones · DBSCAN clusters · Voronoi territories · Live outlets</div></div>', unsafe_allow_html=True)
    m = build_map()
    st_folium(m, width=None, height=600, returned_objects=[])

with ws_col:
    st.markdown('<div class="sh"><div class="sh-title">⚡ White Space Intelligence</div><div class="sh-sub">DBSCAN clusters ranked by opportunity</div></div>', unsafe_allow_html=True)

    if not white_space_zones:
        st.info("Load city data to see white space analysis")
    else:
        for i, z in enumerate(white_space_zones[:5]):
            p     = z["priority"]
            col   = "#ef4444" if p=="critical" else "#f59e0b" if p=="high" else "#3b82f6" if p=="medium" else "#6b7280"
            b_cls = f"b-{p[:4]}" if p != "medium" else "b-med"
            st.markdown(f"""
            <div class="hub-card" style="border-color:{'rgba(239,68,68,0.3)' if p=='critical' else 'rgba(59,130,246,0.2)'}">
              <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                  <span class="badge {b_cls}">#{i+1} {p.upper()}</span>
                  <div class="hub-name" style='margin-top:6px'>Cluster #{z['id']+1} · {z['outlet_count']} outlets</div>
                </div>
                <div style='font-size:32px;font-weight:700;color:{col};line-height:1'>{z['white_space_score']:.0f}</div>
              </div>
              <div class="hub-metric">
                📍 Coverage gap: <b style='color:#f0f6ff'>{z['coverage_gap_pct']}%</b> uncovered<br>
                💰 Annual opp: <b style='color:#10b981'>₹{z['annual_rev_opp']/100000:.1f}L</b><br>
                📊 Demand: <b style='color:#f0f6ff'>{z['demand_score']:.1f}</b> / Supply: <b style='color:#f0f6ff'>{z['supply_score']:.1f}</b><br>
                📐 Centroid: <b style='color:#60a5fa;font-family:JetBrains Mono,monospace'>{z['centroid_lat']:.4f}, {z['centroid_lon']:.4f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        total_opp = sum(z["annual_rev_opp"] for z in white_space_zones if z["priority"] in ["critical","high"])
        st.markdown(f"""
        <div class="ins r">
          <b>Total critical+high opportunity:</b> ₹{total_opp/10000000:.2f}Cr annual revenue
          across {sum(1 for z in white_space_zones if z['priority'] in ['critical','high'])} zones.
          Appointing {sum(1 for z in white_space_zones[:3])} distributors in top clusters
          captures this within {cat_data['growth']} months.
        </div>""", unsafe_allow_html=True)

# ── Analytics Tabs ────────────────────────────────────────────────────────────
st.markdown("---")
t1, t2, t3, t4, t5 = st.tabs([
    "🔬 DBSCAN Analysis",
    "📐 Territory Model",
    "📈 Elbow / Optimal K",
    "🕸 Travel Time Matrix",
    "⚖️ Cannibalization",
])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        # WS scores by cluster
        if white_space_zones:
            colors = ["#ef4444" if z["priority"]=="critical" else
                      "#f59e0b" if z["priority"]=="high" else "#3b82f6"
                      for z in white_space_zones]
            fig = go.Figure(go.Bar(
                x=[f"C{z['id']+1}" for z in white_space_zones],
                y=[z["white_space_score"] for z in white_space_zones],
                marker_color=colors,
                text=[f"{z['white_space_score']:.0f}" for z in white_space_zones],
                textposition="outside",
                textfont=dict(size=9, color="#4d7fa8"),
            ))
            fig.update_layout(**PLOT_DARK, height=280,
                title=dict(text="White space score by DBSCAN cluster", font=dict(size=12,color="#9ca3af")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0,110]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Scatter: demand vs coverage gap
        if white_space_zones:
            p_colors = {"critical":"#ef4444","high":"#f59e0b","medium":"#3b82f6","low":"#6b7280"}
            fig2 = go.Figure()
            for p in ["critical","high","medium","low"]:
                zs = [z for z in white_space_zones if z["priority"]==p]
                if not zs: continue
                fig2.add_trace(go.Scatter(
                    x=[z["demand_score"] for z in zs],
                    y=[z["coverage_gap_pct"] for z in zs],
                    mode="markers+text",
                    name=p.title(),
                    text=[f"C{z['id']+1}" for z in zs],
                    textposition="top center",
                    textfont=dict(size=8, color="#4d7fa8"),
                    marker=dict(
                        size=[max(8, z["white_space_score"]/5) for z in zs],
                        color=p_colors[p], opacity=0.85,
                        line=dict(color="rgba(0,0,0,0.3)", width=1)
                    )
                ))
            fig2.update_layout(**PLOT_DARK, height=280,
                title=dict(text="Demand vs coverage gap (bubble = WS score)", font=dict(size=12,color="#9ca3af")),
                xaxis=dict(title="Demand score", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Coverage gap %", gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(font=dict(size=10,color="#4d7fa8")),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Cluster stats table
    if cluster_result["clusters"]:
        st.markdown(f'<div class="sh"><div class="sh-title">DBSCAN cluster summary</div><div class="sh-sub">ε={dbscan_eps}km · min_samples={min_cluster_size} · noise={cluster_result["noise_count"]}</div></div>', unsafe_allow_html=True)
        df_clusters = pd.DataFrame([{
            "Cluster": f"C{c['id']+1}",
            "Outlets":  c["outlet_count"],
            "Centroid lat": f"{c['centroid_lat']:.5f}",
            "Centroid lon": f"{c['centroid_lon']:.5f}",
            "Lat spread km": c["lat_spread"],
            "Lon spread km": c["lon_spread"],
            "Weighted demand": c["weighted_demand"],
            "WS Score": ws["white_space_score"] if i < len(white_space_zones) else "—",
            "Priority": ws["priority"].upper() if i < len(white_space_zones) else "—",
        } for i, (c, ws) in enumerate(zip(
            cluster_result["clusters"],
            white_space_zones + [{}]*(len(cluster_result["clusters"])-len(white_space_zones))
        ))])
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)

with t2:
    c3, c4 = st.columns(2)
    with c3:
        territories = territory_result.get("territories", [])
        if territories:
            fig3 = go.Figure(go.Bar(
                x=[f"T{t['id']+1}" for t in territories],
                y=[t["outlet_count"] for t in territories],
                marker_color="#8b5cf6",
                text=[str(t["outlet_count"]) for t in territories],
                textposition="outside",
            ))
            avg_outlets = np.mean([t["outlet_count"] for t in territories])
            fig3.add_hline(y=avg_outlets, line_dash="dash",
                           line_color="#f59e0b", annotation_text=f"Avg: {avg_outlets:.0f}")
            fig3.update_layout(**PLOT_DARK, height=280,
                title=dict(text=f"K-Means territory balance (score: {territory_result.get('balance_score',0):.0f}%)",
                           font=dict(size=12,color="#9ca3af")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        if territories:
            fig4 = px.scatter(
                x=[t["avg_dist_km"] for t in territories],
                y=[t["outlet_count"] for t in territories],
                size=[t["total_demand"] for t in territories],
                color=[t["max_reach_km"] for t in territories],
                color_continuous_scale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                text=[f"T{t['id']+1}" for t in territories],
                labels={"x":"Avg dist to outlets (km)","y":"Outlet count","color":"Max reach km"},
                title="Territory efficiency: distance vs coverage",
            )
            fig4.update_layout(**PLOT_DARK, height=280,
                title=dict(font=dict(size=12,color="#9ca3af")))
            st.plotly_chart(fig4, use_container_width=True)

    if territories:
        st.markdown('<div class="sh"><div class="sh-title">K-Means territory details</div><div class="sh-sub">Optimal hub coordinates + coverage metrics</div></div>', unsafe_allow_html=True)
        df_terr = pd.DataFrame([{
            "Territory":      f"T{t['id']+1}",
            "Hub lat":        f"{t['hub_lat']:.5f}",
            "Hub lon":        f"{t['hub_lon']:.5f}",
            "Outlets":        t["outlet_count"],
            "Demand weight":  t["total_demand"],
            "Area km²":       t["area_km2"],
            "Max reach km":   t["max_reach_km"],
            "Avg dist km":    t["avg_dist_km"],
        } for t in territories])
        st.dataframe(df_terr, use_container_width=True, hide_index=True)

with t3:
    elbow_k    = elbow_result.get("optimal_k", 0)
    inertias   = elbow_result.get("inertias", [])
    k_range    = elbow_result.get("k_range", [])

    if inertias:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=8, color="#3b82f6"),
            name="Inertia"
        ))
        fig5.add_vline(
            x=elbow_k, line_dash="dash", line_color="#10b981",
            annotation_text=f"Optimal K = {elbow_k}",
            annotation_font=dict(color="#10b981", size=11)
        )
        fig5.update_layout(**PLOT_DARK, height=320,
            title=dict(text="Elbow method: optimal number of distribution hubs",
                       font=dict(size=13,color="#9ca3af")),
            xaxis=dict(title="Number of distributors (K)", gridcolor="rgba(255,255,255,0.04)",
                       tickmode="linear", dtick=1),
            yaxis=dict(title="Within-cluster sum of squares (inertia)",
                       gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(f"""<div class="ins g">
          <b>Recommendation:</b> The elbow method suggests <b>{elbow_k} distribution hubs</b>
          is optimal for {city_name}. Beyond {elbow_k} hubs, the marginal improvement in
          outlet coverage becomes sublinear — you're paying for diminishing returns.
          Your current plan of {n_distributors} hubs scores
          {"✅ optimal" if n_distributors == elbow_k else
           "⬆️ over-distributed" if n_distributors > elbow_k else "⬇️ under-distributed"}.
        </div>""", unsafe_allow_html=True)

with t4:
    st.markdown('<div class="sh"><div class="sh-title">OSRM Travel Time Matrix</div><div class="sh-sub">Real road travel times between distribution hubs</div></div>', unsafe_allow_html=True)

    if len(all_hubs) >= 2:
        with st.spinner("🛣️ Computing OSRM travel time matrix..."):
            origins = [(h["lat"], h["lon"]) for h in all_hubs]
            try:
                matrix = get_travel_time_matrix(origins, origins)
                hub_names = [h["name"][:15] for h in all_hubs]
                fig6 = go.Figure(go.Heatmap(
                    z=matrix,
                    x=hub_names, y=hub_names,
                    colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                    text=[[f"{v:.0f}m" for v in row] for row in matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    colorbar=dict(title="Travel min", tickfont=dict(color="#4d7fa8")),
                ))
                fig6.update_layout(**PLOT_DARK, height=380,
                    title=dict(text="Hub-to-hub travel time matrix (minutes via road)",
                               font=dict(size=12,color="#9ca3af")),
                    xaxis=dict(tickfont=dict(size=9,color="#4d7fa8")),
                    yaxis=dict(tickfont=dict(size=9,color="#4d7fa8")),
                )
                st.plotly_chart(fig6, use_container_width=True)

                # Show routing data source
                st.markdown('<div class="ins b"><b>Data source:</b> OSRM public routing API (OpenStreetMap). Travel times account for real road networks, one-way streets, and turn restrictions. Falls back to haversine estimate if API unavailable.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Travel time matrix error: {e}")
    else:
        st.info("Add at least 2 distribution hubs to compute travel time matrix.")

with t5:
    st.markdown('<div class="sh"><div class="sh-title">Huff Gravity Model — Cannibalization Analysis</div><div class="sh-sub">Transfer probability between competing distribution hubs</div></div>', unsafe_allow_html=True)

    if territory_result.get("territories"):
        # Use first K-Means territory as candidate
        best_territory = territory_result["territories"][0]
        candidate = {
            "lat": best_territory["hub_lat"],
            "lon": best_territory["hub_lon"],
            "outlet_count": best_territory["outlet_count"],
            "name": f"K-Means T{best_territory['id']+1}",
        }
        result = gravity_cannibalization(candidate, all_hubs)

        c5, c6 = st.columns(2)
        with c5:
            st.metric("Transfer probability", f"{result['transfer_probability']*100:.1f}%")
            st.metric("Cannibalization risk", result["cannibalization_risk"])
            st.metric("Revenue at risk", f"₹{result['total_revenue_at_risk']:,}/mo")

        with c6:
            if result["affected_hubs"]:
                df_cannibal = pd.DataFrame([{
                    "Hub": h["hub_name"],
                    "Distance (km)": h["distance_km"],
                    "Transfer prob": f"{h['transfer_probability']*100:.1f}%",
                    "Revenue at risk": f"₹{h['revenue_at_risk']:,}",
                } for h in result["affected_hubs"]])
                st.dataframe(df_cannibal, use_container_width=True, hide_index=True)

        risk_color = "r" if result["cannibalization_risk"]=="HIGH" else "a" if result["cannibalization_risk"]=="MEDIUM" else "g"
        st.markdown(f"""<div class="ins {risk_color}">
          <b>Gravity model result:</b> Placing a new hub at the K-Means optimal location
          (T{best_territory['id']+1}) shows <b>{result['cannibalization_risk']} cannibalization risk</b>
          with {result['transfer_probability']*100:.1f}% average transfer probability from
          existing hubs. Total revenue at risk: <b>₹{result['total_revenue_at_risk']:,}/month</b>.
          {'Proceed — net incremental exceeds loss.' if result['cannibalization_risk']=='LOW'
           else 'Evaluate net incremental revenue before proceeding.'
           if result['cannibalization_risk']=='MEDIUM'
           else 'High overlap with existing network — reconsider location.'}
        </div>""", unsafe_allow_html=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
ec1, ec2, ec3 = st.columns(3)

with ec1:
    # White space report
    if white_space_zones:
        df_ws = pd.DataFrame([{
            "Cluster": f"C{z['id']+1}",
            "Centroid lat": z["centroid_lat"],
            "Centroid lon": z["centroid_lon"],
            "WS Score": z["white_space_score"],
            "Priority": z["priority"],
            "Demand score": z["demand_score"],
            "Supply score": z["supply_score"],
            "Coverage gap %": z["coverage_gap_pct"],
            "Outlet count": z["outlet_count"],
            "Monthly opp (₹)": z["monthly_rev_opp"],
            "Annual opp (₹)": z["annual_rev_opp"],
            "Category": category,
            "City": city_name,
        } for z in white_space_zones])
        st.download_button(
            "⬇️ White space report (CSV)",
            data=df_ws.to_csv(index=False).encode(),
            file_name=f"districtiq_whitespace_{city_name.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )

with ec2:
    # Territory model export
    territories = territory_result.get("territories", [])
    if territories:
        df_t = pd.DataFrame([{
            "Territory": f"T{t['id']+1}",
            "Hub lat": t["hub_lat"], "Hub lon": t["hub_lon"],
            "Outlets": t["outlet_count"], "Demand": t["total_demand"],
            "Area km2": t["area_km2"], "Max reach km": t["max_reach_km"],
            "Avg dist km": t["avg_dist_km"],
        } for t in territories])
        st.download_button(
            "⬇️ Territory model (CSV)",
            data=df_t.to_csv(index=False).encode(),
            file_name=f"districtiq_territories_{city_name.lower().replace(' ','_')}.csv",
            mime="text/csv", use_container_width=True,
        )

with ec3:
    # Full JSON export
    summary = {
        "generated": datetime.now().isoformat(),
        "city": city_name, "category": category,
        "data_source": data_source,
        "outlets_analysed": len(all_outlets),
        "dbscan_clusters": cluster_result["n_clusters"],
        "optimal_hubs_elbow": elbow_result["optimal_k"],
        "territory_balance_score": territory_result.get("balance_score", 0),
        "total_annual_opportunity_cr": round(total_rev_opp/10000000, 2),
        "critical_white_spaces": n_critical,
        "top_3_clusters": [{
            "id": z["id"], "ws_score": z["white_space_score"],
            "priority": z["priority"],
            "lat": z["centroid_lat"], "lon": z["centroid_lon"],
            "annual_opp_lakhs": round(z["annual_rev_opp"]/100000, 1),
        } for z in white_space_zones[:3]],
        "valhalla_isochrones": {
            h["name"]: {
                "source": hub_isochrones.get(h["name"],{}).get("source","n/a"),
                "area_km2": hub_isochrones.get(h["name"],{}).get("area_km2",0),
            } for h in all_hubs
        },
    }
    st.download_button(
        "⬇️ Analysis JSON",
        data=json.dumps(summary, indent=2).encode(),
        file_name=f"districtiq_{city_name.lower().replace(' ','_')}_analysis.json",
        mime="application/json", use_container_width=True,
    )

st.markdown("""
<div style='text-align:center;padding:20px 0 8px;font-size:10px;color:#1e3a5f'>
  DistrictIQ v3 · Valhalla isochrones · OSRM routing · Overpass live outlets ·
  DBSCAN spatial clustering · K-Means territory partitioning · Voronoi tessellation ·
  Huff Gravity Model · Elbow optimal K · Built for institutional distribution analysis
</div>
""", unsafe_allow_html=True)
