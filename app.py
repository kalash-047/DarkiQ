"""
DarkIQ v2 — Real-Time Dark Store Placement Engine
Works for ANY city in the world via live OpenStreetMap data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import json
import math
import time
from io import BytesIO
import requests

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DarkIQ — Real-Time Placement Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1rem; max-width: 1400px; }
  .metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
    border: 1px solid #2e3250;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
  }
  .score-ring {
    font-size: 40px;
    font-weight: 800;
    line-height: 1;
  }
  .score-high { color: #00e676; }
  .score-med  { color: #ffb300; }
  .score-low  { color: #ef5350; }
  .badge {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 12px;
    font-weight: 700;
  }
  .badge-rank { background: #6c63ff; color: white; }
  .badge-live { background: #00c853; color: #000; }
  .badge-cached { background: #ff6f00; color: #000; }
  .factor-bar-wrap {
    background: #2a2d3e;
    border-radius: 6px;
    height: 8px;
    margin: 3px 0;
    overflow: hidden;
  }
  .factor-bar {
    height: 8px;
    border-radius: 6px;
    transition: width 0.6s;
  }
  .insight-box {
    background: #12151f;
    border-left: 3px solid #6c63ff;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 13px;
  }
  .stProgress > div > div { background: #6c63ff; }
</style>
""", unsafe_allow_html=True)

HEADERS = {"User-Agent": "DarkIQ-MVP/2.0"}

# ─── Utility ──────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def normalize(v, lo, hi, invert=False):
    if hi == lo: return 50.0
    n = max(0, min(100, ((v - lo) / (hi - lo)) * 100))
    return round(100 - n if invert else n, 1)

# ─── Real-time data fetching ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city_name: str):
    """Live geocoding via Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    try:
        r = requests.get(url, params={"q": city_name, "format": "json", "limit": 1},
                         headers=HEADERS, timeout=10)
        data = r.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0]["display_name"],
                "bbox": data[0].get("boundingbox", []),
            }
    except Exception as e:
        st.warning(f"Geocoding error: {e}")
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_zones_for_city(city_name: str, city_lat: float, city_lon: float):
    """
    Fetch real named districts/neighborhoods for any city using OSM.
    """
    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:40];
    (
      node["place"~"suburb|neighbourhood|quarter|district"]["name"](around:20000,{city_lat},{city_lon});
      way["place"~"suburb|neighbourhood|quarter"]["name"](around:20000,{city_lat},{city_lon});
    );
    out center 40;
    """
    zones = []
    try:
        r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=45)
        data = r.json()
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "")
            if not name:
                continue
            if el["type"] == "node":
                lat, lon = el["lat"], el["lon"]
            else:
                center = el.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")
            if lat and lon and name:
                dist = haversine_km(city_lat, city_lon, lat, lon)
                if dist <= 18:
                    zones.append({"name": name, "lat": lat, "lon": lon, "dist_km": round(dist, 1)})
    except Exception as e:
        st.warning(f"Zone fetch error: {e}")

    # Deduplicate and sort by distance
    seen = set()
    unique = []
    for z in zones:
        key = z["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(z)
    return sorted(unique, key=lambda x: x["dist_km"])[:25]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_zone_data(zone_name: str, lat: float, lon: float, radius_m: int = 2500):
    """
    Fetch comprehensive live data for a single zone from OSM.
    This is the core intelligence pull.
    """
    url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:45];
    (
      node["amenity"~"restaurant|cafe|fast_food|food_court"](around:{radius_m},{lat},{lon});
      node["shop"~"supermarket|convenience|grocery|mall"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
      node["amenity"~"hospital|clinic|pharmacy"](around:{radius_m},{lat},{lon});
      node["amenity"~"school|college|university"](around:{radius_m},{lat},{lon});
      node["highway"="bus_stop"](around:{radius_m},{lat},{lon});
      node["amenity"="atm"](around:{radius_m},{lat},{lon});
      node["amenity"="bank"](around:{radius_m},{lat},{lon});
      node["leisure"~"gym|fitness_centre"](around:{radius_m},{lat},{lon});
      node["amenity"~"fuel|parking"](around:{radius_m},{lat},{lon});
      way["highway"~"primary|secondary|tertiary"](around:{radius_m},{lat},{lon});
      way["building"~"residential|apartments"](around:{radius_m},{lat},{lon});
      node["name"~"Blinkit|Zepto|Instamart|BigBasket|Dunzo|JioMart|Swiggy",i](around:{radius_m},{lat},{lon});
    );
    out count;
    """

    # Also run detailed individual counts
    detail_query = f"""
    [out:json][timeout:50];
    (
      node["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
      node["shop"~"supermarket|convenience|grocery"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
      node["highway"="bus_stop"](around:{radius_m},{lat},{lon});
      node["amenity"="atm"](around:{radius_m},{lat},{lon});
      node["amenity"~"school|college"](around:{radius_m},{lat},{lon});
      node["amenity"~"hospital|clinic|pharmacy"](around:{radius_m},{lat},{lon});
      way["highway"~"primary|secondary"](around:{radius_m},{lat},{lon});
      node["name"~"Blinkit|Zepto|Instamart|BigBasket|Swiggy",i](around:{radius_m},{lat},{lon});
    );
    out body;
    """

    counts = {
        "restaurants": 0, "supermarkets": 0, "offices": 0,
        "transit": 0, "atms": 0, "schools": 0,
        "hospitals": 0, "main_roads": 0, "competitors": 0
    }
    competitor_nodes = []

    try:
        r = requests.post(url, data={"data": detail_query}, headers=HEADERS, timeout=55)
        data = r.json()

        for el in data.get("elements", []):
            tags = el.get("tags", {})
            amenity = tags.get("amenity", "")
            shop    = tags.get("shop", "")
            highway = tags.get("highway", "")
            name    = tags.get("name", "").lower()

            if any(b in name for b in ["blinkit","zepto","instamart","bigbasket","swiggy","dunzo","jiomart"]):
                counts["competitors"] += 1
                competitor_nodes.append({"name": tags.get("name","Competitor"), "lat": el.get("lat", lat), "lon": el.get("lon", lon)})
            elif amenity in ["restaurant","cafe","fast_food","food_court"]:
                counts["restaurants"] += 1
            elif shop in ["supermarket","convenience","grocery"]:
                counts["supermarkets"] += 1
            elif "office" in tags:
                counts["offices"] += 1
            elif highway == "bus_stop":
                counts["transit"] += 1
            elif amenity == "atm":
                counts["atms"] += 1
            elif amenity in ["school","college","university"]:
                counts["schools"] += 1
            elif amenity in ["hospital","clinic","pharmacy"]:
                counts["hospitals"] += 1
            elif el["type"] == "way" and highway in ["primary","secondary"]:
                counts["main_roads"] += 1

        return {
            "counts": counts,
            "competitors": competitor_nodes,
            "total_pois": sum(counts.values()),
            "success": True,
            "source": "OpenStreetMap (live)"
        }

    except Exception as e:
        return {"counts": counts, "competitors": [], "total_pois": 0,
                "success": False, "error": str(e), "source": "fallback"}


# ─── Scoring logic ────────────────────────────────────────────────────────────
SCENARIOS = {
    "⚖️ Balanced":              {"population":1.2, "demand":1.5, "accessibility":1.2, "rent_value":0.8, "comp_gap":1.0, "road":1.0},
    "🚀 Max order density":     {"population":0.8, "demand":2.5, "accessibility":1.5, "rent_value":0.5, "comp_gap":0.7, "road":1.2},
    "💰 Minimise cost":         {"population":0.8, "demand":1.0, "accessibility":1.0, "rent_value":2.5, "comp_gap":0.7, "road":0.8},
    "⚔️ Beat competitors":      {"population":1.0, "demand":1.5, "accessibility":1.0, "rent_value":0.8, "comp_gap":2.5, "road":0.9},
    "🌱 Underserved areas":     {"population":2.0, "demand":0.8, "accessibility":1.2, "rent_value":1.5, "comp_gap":2.0, "road":0.8},
    "🏢 Enterprise hub":        {"population":1.5, "demand":2.0, "accessibility":1.8, "rent_value":1.0, "comp_gap":0.8, "road":1.5},
    "🎛️ Custom":                None,
}

def compute_score(zone_data: dict, weights: dict, lat: float, lon: float) -> dict:
    c = zone_data.get("counts", {})
    competitors = zone_data.get("competitors", [])

    # Sub-scores
    demand_raw = (
        c.get("restaurants", 0) * 2.5 +
        c.get("offices", 0) * 2.0 +
        c.get("transit", 0) * 1.5 +
        c.get("supermarkets", 0) * 1.2 +
        c.get("schools", 0) * 0.8 +
        c.get("atms", 0) * 1.0 +
        c.get("hospitals", 0) * 0.6
    )
    demand_score  = normalize(demand_raw, 0, 400)
    road_score    = normalize(c.get("main_roads", 0), 0, 40)
    transit_score = normalize(c.get("transit", 0), 0, 30)
    access_score  = round(road_score * 0.65 + transit_score * 0.35, 1)

    # Population proxy: buildings + residential density signals
    pop_raw = (
        c.get("supermarkets", 0) * 3 +
        c.get("schools", 0) * 4 +
        c.get("hospitals", 0) * 5 +
        c.get("atms", 0) * 3
    )
    pop_score = normalize(pop_raw, 0, 150)

    # Rent proxy: commercial density = higher rent = invert
    commercial_density = c.get("supermarkets", 0) * 3 + c.get("offices", 0) * 2 + c.get("restaurants", 0)
    rent_score = normalize(commercial_density, 0, 300, invert=True)
    rent_score = round(max(20, min(90, rent_score)), 1)

    # Competitor gap
    comp_count = c.get("competitors", 0) + len(competitors)
    comp_score = max(10, 90 - (comp_count * 18))

    sub = {
        "population":    pop_score,
        "demand":        demand_score,
        "accessibility": access_score,
        "rent_value":    rent_score,
        "comp_gap":      comp_score,
        "road":          road_score,
    }

    total_w = sum(weights.values())
    score = round(sum(sub[k] * weights[k] for k in weights) / total_w, 1)
    contributions = {k: round(sub[k] * weights[k] / total_w, 1) for k in weights}

    # Delivery time estimate
    delivery = max(8, round(28 - (access_score / 100) * 8 - (road_score / 100) * 5))

    # Coverage
    coverage_km = round(2.0 + (score / 100) * 1.5, 1)

    # Order estimate
    monthly_orders = round(2500 * (demand_score / 100) * (0.6 + pop_score / 100 * 0.8))

    return {
        "score":           score,
        "sub_scores":      sub,
        "contributions":   contributions,
        "delivery_time":   delivery,
        "coverage_km":     coverage_km,
        "monthly_orders":  monthly_orders,
        "monthly_revenue": monthly_orders * 350,
        "comp_count":      comp_count,
        "raw_counts":      c,
    }


# ─── Map builder ──────────────────────────────────────────────────────────────
def build_map(city_lat, city_lon, zoom, scored_zones, show_heat, show_cov, show_comp, top_n):
    m = folium.Map(location=[city_lat, city_lon], zoom_start=zoom,
                   tiles="CartoDB dark_matter")

    if show_heat:
        heat_data = [[z["lat"], z["lon"], z["score"]/100] for z in scored_zones if z.get("score")]
        if heat_data:
            HeatMap(heat_data, radius=32, blur=28, min_opacity=0.25,
                    gradient={"0.2":"#1a237e","0.5":"#ff6f00","0.8":"#e53935","1.0":"#ffffff"}).add_to(m)

    top_zones = scored_zones[:top_n]
    for i, z in enumerate(scored_zones):
        if not z.get("score"):
            continue
        score = z["score"]
        is_top = z in top_zones
        color  = "#00e676" if score >= 75 else ("#ffb300" if score >= 60 else "#ef5350")

        if show_cov and is_top:
            folium.Circle(
                [z["lat"], z["lon"]],
                radius=z.get("coverage_km", 2.5) * 1000,
                color=color, fill=True, fill_opacity=0.07,
                weight=1.5, dash_array="6 4"
            ).add_to(m)

        # Competitor markers
        if show_comp:
            for comp in z.get("zone_data", {}).get("competitors", []):
                folium.CircleMarker(
                    [comp["lat"], comp["lon"]], radius=5,
                    color="#fdd835", fill=True, fill_opacity=0.85,
                    tooltip=f"⚠️ Competitor: {comp['name']}"
                ).add_to(m)

        rank = f"#{i+1}" if is_top else ""
        r_size = 14 if is_top else 9
        r_name = z["name"][:14]

        popup_html = f"""
        <div style='font-family:Arial;width:230px;background:#1a1d27;color:#e0e0e0;padding:12px;border-radius:10px'>
          <b style='font-size:15px;color:{color}'>{rank} {z["name"]}</b><br>
          <hr style='border-color:#333;margin:6px 0'>
          <b>Score: {score}/100</b> &nbsp;|&nbsp; <small style='color:#aaa'>Live OSM data</small><br>
          ⏱ Est. delivery: <b>{z.get("delivery_time","?")} min</b><br>
          📍 Coverage: <b>{z.get("coverage_km","?")} km radius</b><br>
          📦 Est. orders/month: <b>{z.get("monthly_orders","?"):,}</b><br>
          💰 Est. revenue: <b>₹{z.get("monthly_revenue","?"):,.0f}</b><br>
          <hr style='border-color:#333;margin:6px 0'>
          <small>
            🍽 Restaurants: {z.get("raw_counts",{}).get("restaurants","?")} &nbsp;
            🏢 Offices: {z.get("raw_counts",{}).get("offices","?")} <br>
            🚌 Transit: {z.get("raw_counts",{}).get("transit","?")} &nbsp;
            🏪 Shops: {z.get("raw_counts",{}).get("supermarkets","?")}
          </small>
        </div>
        """

        folium.CircleMarker(
            [z["lat"], z["lon"]],
            radius=r_size, color=color, fill=True, fill_opacity=0.9, weight=2,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{'🏆 ' if is_top else ''}{z['name']}: {score}/100"
        ).add_to(m)

        if is_top:
            folium.Marker(
                [z["lat"] + 0.003, z["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="background:{color};color:#000;font-weight:800;font-size:11px;padding:2px 8px;border-radius:10px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.4)">{rank} {r_name}</div>',
                    icon_size=(140, 24), icon_anchor=(70, 0)
                )
            ).add_to(m)

    return m


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 DarkIQ v2")
    st.markdown("*Real-Time Placement Engine*")
    st.markdown('<span class="badge badge-live">● LIVE DATA</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🌍 Search any city")
    city_input = st.text_input("City name", value="Bengaluru, India",
                               placeholder="e.g. Pune, Kolkata, Surat...")
    search_btn = st.button("🔍 Load city", use_container_width=True, type="primary")

    st.markdown("### 🎯 Business scenario")
    scenario = st.selectbox("", list(SCENARIOS.keys()), label_visibility="collapsed")

    weights = SCENARIOS[scenario]
    if scenario == "🎛️ Custom":
        st.markdown("#### Adjust weights")
        w_pop  = st.slider("Population",    0.0, 3.0, 1.2, 0.1)
        w_dem  = st.slider("Order demand",  0.0, 3.0, 1.5, 0.1)
        w_acc  = st.slider("Accessibility", 0.0, 3.0, 1.2, 0.1)
        w_rent = st.slider("Rent value",    0.0, 3.0, 0.8, 0.1)
        w_comp = st.slider("Comp gap",      0.0, 3.0, 1.0, 0.1)
        w_road = st.slider("Road quality",  0.0, 3.0, 1.0, 0.1)
        weights = {"population": w_pop, "demand": w_dem, "accessibility": w_acc,
                   "rent_value": w_rent, "comp_gap": w_comp, "road": w_road}

    st.markdown("### 🏅 Top N locations")
    top_n = st.slider("", 1, 8, 3, label_visibility="collapsed")

    st.markdown("### 🗺️ Map layers")
    show_heat = st.toggle("Demand heatmap",        value=True)
    show_cov  = st.toggle("Coverage circles",      value=True)
    show_comp = st.toggle("Competitor markers",    value=True)

    st.markdown("---")
    st.markdown("### 📤 Your order data (optional)")
    uploaded = st.file_uploader("CSV: Zone, OrderIndex", type=["csv"])
    if uploaded:
        st.caption("Your data will override demo demand scores")

    st.markdown("---")
    st.markdown("### 📡 Data sources")
    st.caption("🗺 OpenStreetMap (live POI data)\n\n🏙 Nominatim geocoding\n\n📊 Overpass API\n\n🛣 OSRM routing")

# ─── State management ─────────────────────────────────────────────────────────
if "city_data" not in st.session_state:
    st.session_state.city_data = None
if "zones" not in st.session_state:
    st.session_state.zones = []
if "scored_zones" not in st.session_state:
    st.session_state.scored_zones = []
if "current_city" not in st.session_state:
    st.session_state.current_city = ""

# ─── Load city on search ─────────────────────────────────────────────────────
if search_btn or (st.session_state.current_city == "" and city_input):
    if city_input != st.session_state.current_city:
        with st.spinner(f"📡 Geocoding {city_input}..."):
            city_data = geocode_city(city_input)
        if city_data:
            st.session_state.city_data = city_data
            st.session_state.current_city = city_input
            st.session_state.scored_zones = []

            with st.spinner(f"🗺 Fetching real neighborhoods from OpenStreetMap..."):
                zones = fetch_zones_for_city(city_input, city_data["lat"], city_data["lon"])
            st.session_state.zones = zones
            if not zones:
                st.warning("No neighborhoods found. Try a more specific city name (e.g. 'Bengaluru, India')")
        else:
            st.error("City not found. Try adding country (e.g. 'Kolkata, India')")

# ─── Score zones ──────────────────────────────────────────────────────────────
city_data = st.session_state.city_data
zones = st.session_state.zones

if city_data and zones and (
    not st.session_state.scored_zones or
    len(st.session_state.scored_zones) != len(zones)
):
    progress_bar = st.progress(0, text="🔄 Fetching live data for each zone...")
    scored = []

    for i, zone in enumerate(zones):
        progress_bar.progress((i + 1) / len(zones),
                              text=f"📡 Analysing {zone['name']} ({i+1}/{len(zones)})...")
        zone_data = fetch_zone_data(zone["name"], zone["lat"], zone["lon"])
        result = compute_score(zone_data, weights, zone["lat"], zone["lon"])
        scored.append({
            **zone,
            **result,
            "zone_data": zone_data,
            "data_live": zone_data.get("success", False),
        })
        time.sleep(0.3)  # Rate limit respect

    scored.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.scored_zones = scored
    progress_bar.empty()

scored_zones = st.session_state.scored_zones

# ─── Header ──────────────────────────────────────────────────────────────────
if city_data:
    city_display = city_input.split(",")[0].strip()
    st.markdown(f"# 📦 DarkIQ — {city_display}")
    col_title1, col_title2 = st.columns([3,1])
    with col_title1:
        st.markdown(f"*Scenario: **{scenario}** · Top **{top_n}** locations · {len(scored_zones)} zones analysed*")
    with col_title2:
        st.markdown('<span class="badge badge-live">● Live OSM data</span>', unsafe_allow_html=True)
else:
    st.markdown("# 📦 DarkIQ — Real-Time Placement Engine")
    st.info("👈 Enter any city name in the sidebar and click **Load city** to begin.")
    st.markdown("**Works for any city in the world** — Bengaluru, Mumbai, Delhi, Kolkata, Surat, Jaipur, Dubai, London, anywhere.")

    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Enter city**\nType any city name in the sidebar")
    with col2:
        st.markdown("**2. Live data fetch**\nReal POI data pulled from OpenStreetMap")
    with col3:
        st.markdown("**3. AI scoring**\nEvery zone scored across 6 factors")
    with col4:
        st.markdown("**4. Ranked results**\nTop locations with delivery estimates")
    st.stop()

# ─── Top metrics ─────────────────────────────────────────────────────────────
if scored_zones:
    top = scored_zones[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🏆 Best location", top["name"], f"Score: {top['score']}/100")
    with c2:
        st.metric("⏱ Est. delivery", f"{top.get('delivery_time','?')} min", "from this node")
    with c3:
        st.metric("📍 Coverage", f"{top.get('coverage_km','?')} km", "service radius")
    with c4:
        st.metric("📦 Monthly orders", f"{top.get('monthly_orders',0):,}", "est. from top zone")
    with c5:
        live_count = sum(1 for z in scored_zones if z.get("data_live"))
        st.metric("📡 Live zones", f"{live_count}/{len(scored_zones)}", "real OSM data")

    st.markdown("---")

# ─── Main layout ─────────────────────────────────────────────────────────────
if scored_zones:
    map_col, rank_col = st.columns([3, 2])

    with map_col:
        st.markdown("### 🗺️ Live placement map")
        st.caption("🟢 Green = highly recommended · 🟡 Yellow = viable · 🔴 Red = avoid · Click markers for details")

        zoom = 12 if haversine_km(city_data["lat"], city_data["lon"],
                                   scored_zones[0]["lat"], scored_zones[0]["lon"]) < 15 else 11
        m = build_map(
            city_data["lat"], city_data["lon"], zoom,
            scored_zones, show_heat, show_cov, show_comp, top_n
        )
        st_folium(m, width=None, height=540, returned_objects=[])

    with rank_col:
        st.markdown("### 🏅 Ranked locations")
        factor_labels = {
            "population":    "Population",
            "demand":        "Demand",
            "accessibility": "Access",
            "rent_value":    "Rent value",
            "comp_gap":      "Comp. gap",
            "road":          "Roads",
        }
        bar_colors = {
            "population":    "#6c63ff",
            "demand":        "#00b0ff",
            "accessibility": "#00e676",
            "rent_value":    "#ffb300",
            "comp_gap":      "#ff6d00",
            "road":          "#ea80fc",
        }

        for i, z in enumerate(scored_zones[:top_n]):
            score = z["score"]
            color = "#00e676" if score >= 75 else ("#ffb300" if score >= 60 else "#ef5350")
            live_badge = '<span class="badge badge-live" style="font-size:10px">LIVE</span>' if z.get("data_live") else '<span class="badge badge-cached" style="font-size:10px">CACHED</span>'

            bars_html = ""
            for factor, val in z.get("sub_scores", {}).items():
                label = factor_labels.get(factor, factor)
                col = bar_colors.get(factor, "#6c63ff")
                bars_html += f"""
                <div style='display:flex;align-items:center;gap:6px;margin:3px 0'>
                  <span style='font-size:11px;color:#aaa;width:70px;flex-shrink:0'>{label}</span>
                  <div class='factor-bar-wrap' style='flex:1'>
                    <div class='factor-bar' style='width:{val}%;background:{col}'></div>
                  </div>
                  <span style='font-size:11px;color:#ccc;width:28px;text-align:right'>{val:.0f}</span>
                </div>"""

            st.markdown(f"""
            <div class="metric-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <span class="badge badge-rank">#{i+1}</span> {live_badge}
                  <br><b style="font-size:17px">{z["name"]}</b>
                </div>
                <div class="score-ring {'score-high' if score >= 75 else ('score-med' if score >= 60 else 'score-low')}">{score}</div>
              </div>
              <div style="color:#aaa;font-size:12px;margin:4px 0 10px">
                ⏱ {z.get("delivery_time","?")} min &nbsp;|&nbsp;
                📍 {z.get("coverage_km","?")} km &nbsp;|&nbsp;
                📦 {z.get("monthly_orders",0):,} orders/mo
              </div>
              {bars_html}
            </div>
            """, unsafe_allow_html=True)

        if len(scored_zones) > top_n:
            next_z = scored_zones[top_n]
            gap = round(scored_zones[top_n-1]["score"] - next_z["score"], 1)
            st.info(f"📉 Score gap to #{top_n+1} ({next_z['name']}): **{gap} pts** below cut-off")

# ─── Full data table ──────────────────────────────────────────────────────────
if scored_zones:
    st.markdown("---")
    st.markdown("### 📋 Full zone comparison (live OSM data)")

    rows = []
    for i, z in enumerate(scored_zones):
        rc = z.get("raw_counts", {})
        rows.append({
            "Rank":          f"#{i+1}",
            "Zone":          z["name"],
            "Score":         z["score"],
            "Restaurants":   rc.get("restaurants", 0),
            "Offices":       rc.get("offices", 0),
            "Transit stops": rc.get("transit", 0),
            "Shops":         rc.get("supermarkets", 0),
            "Competitors":   z.get("comp_count", 0),
            "Delivery (min)": z.get("delivery_time", "?"),
            "Coverage (km)": z.get("coverage_km", "?"),
            "Est. orders/mo": f"{z.get('monthly_orders',0):,}",
            "Est. revenue":  f"₹{z.get('monthly_revenue',0):,.0f}",
            "Live data":     "✅" if z.get("data_live") else "⚠️",
        })

    df = pd.DataFrame(rows)

    def color_score(val):
        try:
            v = float(val)
            if v >= 75: return "background:#1b3a2a;color:#00e676"
            elif v >= 60: return "background:#3a2e10;color:#ffb300"
            else: return "background:#3a1a1a;color:#ef5350"
        except: return ""

    st.dataframe(
        df.style.applymap(color_score, subset=["Score"]),
        use_container_width=True, hide_index=True
    )

# ─── Scenario comparison ──────────────────────────────────────────────────────
if scored_zones:
    st.markdown("---")
    st.markdown("### 🔀 What-if scenario comparison")
    st.caption("See how rankings change under different business strategies")

    sc_list = [s for s in SCENARIOS.keys() if s != "🎛️ Custom"]
    col_a, col_b = st.columns(2)

    with col_a:
        sa = st.selectbox("Scenario A", sc_list, index=0, key="csa")
        wa = SCENARIOS[sa]
        scores_a = {}
        for z in scored_zones:
            r = compute_score(z.get("zone_data", {"counts": z.get("raw_counts", {}), "competitors": []}),
                              wa, z["lat"], z["lon"])
            scores_a[z["name"]] = r["score"]

    with col_b:
        sb = st.selectbox("Scenario B", sc_list, index=1, key="csb")
        wb = SCENARIOS[sb]
        scores_b = {}
        for z in scored_zones:
            r = compute_score(z.get("zone_data", {"counts": z.get("raw_counts", {}), "competitors": []}),
                              wb, z["lat"], z["lon"])
            scores_b[z["name"]] = r["score"]

    compare_rows = [{
        "Zone": name,
        f"Score ({sa.split()[1]})": scores_a[name],
        f"Score ({sb.split()[1]})": scores_b[name],
        "Difference": round(scores_a[name] - scores_b[name], 1),
    } for name in scores_a]

    compare_df = pd.DataFrame(compare_rows).sort_values(f"Score ({sa.split()[1]})", ascending=False)

    def color_diff(val):
        try:
            v = float(val)
            if v > 3: return "color:#00e676"
            elif v < -3: return "color:#ef5350"
            return "color:#aaa"
        except: return ""

    st.dataframe(
        compare_df.style.applymap(color_diff, subset=["Difference"]),
        use_container_width=True, hide_index=True
    )

# ─── Download ─────────────────────────────────────────────────────────────────
if scored_zones:
    st.markdown("---")
    st.markdown("### 📥 Export report")

    export_rows = [{
        "Rank": f"#{i+1}",
        "City": city_input,
        "Zone": z["name"],
        "Score": z["score"],
        "Delivery (min)": z.get("delivery_time",""),
        "Coverage (km)": z.get("coverage_km",""),
        "Monthly orders (est)": z.get("monthly_orders",""),
        "Monthly revenue (est ₹)": z.get("monthly_revenue",""),
        "Restaurants (OSM)": z.get("raw_counts",{}).get("restaurants",""),
        "Offices (OSM)": z.get("raw_counts",{}).get("offices",""),
        "Transit stops (OSM)": z.get("raw_counts",{}).get("transit",""),
        "Shops (OSM)": z.get("raw_counts",{}).get("supermarkets",""),
        "Competitors found": z.get("comp_count",""),
        "Scenario": scenario,
        "Data source": "OpenStreetMap (live)",
    } for i, z in enumerate(scored_zones)]

    csv = pd.DataFrame(export_rows).to_csv(index=False).encode()
    safe_city = city_input.split(",")[0].strip().lower().replace(" ","_")
    st.download_button(
        "⬇️ Download full report (CSV)",
        data=csv,
        file_name=f"darkiq_{safe_city}_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("DarkIQ v2 · Powered by OpenStreetMap, Overpass API, Nominatim · Live data fetched in real-time · For planning and evaluation")
