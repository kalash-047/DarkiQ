"""
DarkIQ v2 — Real-Time Dark Store Placement Engine
Fixed for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import math
import requests

st.set_page_config(
    page_title="DarkIQ — Placement Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; max-width: 1400px; }
  .metric-card { background:#1a1d27; border:1px solid #2e3250; border-radius:14px; padding:18px 22px; margin-bottom:12px; }
  .score-high { color:#00e676; font-size:40px; font-weight:800; line-height:1; }
  .score-med  { color:#ffb300; font-size:40px; font-weight:800; line-height:1; }
  .score-low  { color:#ef5350; font-size:40px; font-weight:800; line-height:1; }
  .badge-rank { background:#6c63ff; color:white; border-radius:20px; padding:3px 14px; font-size:12px; font-weight:700; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ─── Real zone data for 9 Indian cities ──────────────────────────────────────
CITY_ZONES = {
    "Bengaluru": {
        "center": [12.9716, 77.5946], "zoom": 12,
        "zones": [
            {"name":"Koramangala",     "lat":12.9352,"lon":77.6245,"restaurants":312,"offices":145,"transit":28,"supermarkets":42,"schools":18,"atms":35,"main_roads":22},
            {"name":"Indiranagar",     "lat":12.9784,"lon":77.6408,"restaurants":285,"offices":128,"transit":24,"supermarkets":38,"schools":15,"atms":30,"main_roads":18},
            {"name":"HSR Layout",      "lat":12.9116,"lon":77.6389,"restaurants":198,"offices":110,"transit":20,"supermarkets":32,"schools":22,"atms":28,"main_roads":16},
            {"name":"BTM Layout",      "lat":12.9166,"lon":77.6101,"restaurants":210,"offices":95, "transit":22,"supermarkets":35,"schools":20,"atms":26,"main_roads":15},
            {"name":"Whitefield",      "lat":12.9698,"lon":77.7499,"restaurants":175,"offices":180,"transit":18,"supermarkets":28,"schools":25,"atms":32,"main_roads":20},
            {"name":"Marathahalli",    "lat":12.9591,"lon":77.6971,"restaurants":195,"offices":135,"transit":19,"supermarkets":30,"schools":19,"atms":27,"main_roads":17},
            {"name":"Jayanagar",       "lat":12.9308,"lon":77.5832,"restaurants":165,"offices":72, "transit":25,"supermarkets":45,"schools":28,"atms":38,"main_roads":14},
            {"name":"Hebbal",          "lat":13.0353,"lon":77.5970,"restaurants":140,"offices":90, "transit":16,"supermarkets":22,"schools":14,"atms":20,"main_roads":19},
            {"name":"Electronic City", "lat":12.8399,"lon":77.6770,"restaurants":155,"offices":210,"transit":14,"supermarkets":20,"schools":12,"atms":18,"main_roads":15},
            {"name":"Rajajinagar",     "lat":12.9914,"lon":77.5528,"restaurants":170,"offices":65, "transit":22,"supermarkets":40,"schools":24,"atms":32,"main_roads":13},
            {"name":"Malleshwaram",    "lat":13.0034,"lon":77.5660,"restaurants":158,"offices":58, "transit":26,"supermarkets":48,"schools":30,"atms":35,"main_roads":12},
            {"name":"Yelahanka",       "lat":13.1007,"lon":77.5963,"restaurants":98, "offices":45, "transit":12,"supermarkets":18,"schools":16,"atms":15,"main_roads":14},
        ],
        "competitors":[{"name":"Blinkit","lat":12.9360,"lon":77.6200},{"name":"Zepto","lat":12.9800,"lon":77.6380},{"name":"Swiggy","lat":12.9120,"lon":77.6370},{"name":"Blinkit","lat":12.9600,"lon":77.6950},{"name":"Zepto","lat":12.9300,"lon":77.5850}]
    },
    "Mumbai": {
        "center": [19.0760, 72.8777], "zoom": 12,
        "zones": [
            {"name":"Bandra West",   "lat":19.0544,"lon":72.8405,"restaurants":380,"offices":85, "transit":35,"supermarkets":55,"schools":22,"atms":48,"main_roads":18},
            {"name":"Andheri East",  "lat":19.1136,"lon":72.8697,"restaurants":295,"offices":175,"transit":38,"supermarkets":42,"schools":18,"atms":40,"main_roads":22},
            {"name":"Powai",         "lat":19.1197,"lon":72.9051,"restaurants":210,"offices":220,"transit":22,"supermarkets":30,"schools":20,"atms":28,"main_roads":16},
            {"name":"Malad West",    "lat":19.1860,"lon":72.8488,"restaurants":245,"offices":95, "transit":30,"supermarkets":38,"schools":25,"atms":35,"main_roads":17},
            {"name":"Goregaon East", "lat":19.1663,"lon":72.8526,"restaurants":220,"offices":115,"transit":28,"supermarkets":32,"schools":20,"atms":30,"main_roads":16},
            {"name":"Juhu",          "lat":19.1075,"lon":72.8263,"restaurants":265,"offices":55, "transit":25,"supermarkets":35,"schools":18,"atms":38,"main_roads":14},
            {"name":"Thane",         "lat":19.2183,"lon":72.9781,"restaurants":235,"offices":110,"transit":32,"supermarkets":45,"schools":28,"atms":40,"main_roads":20},
            {"name":"Chembur",       "lat":19.0522,"lon":72.8996,"restaurants":198,"offices":88, "transit":28,"supermarkets":30,"schools":22,"atms":28,"main_roads":15},
            {"name":"Borivali",      "lat":19.2307,"lon":72.8567,"restaurants":215,"offices":72, "transit":30,"supermarkets":40,"schools":26,"atms":35,"main_roads":16},
            {"name":"Navi Mumbai",   "lat":19.0330,"lon":73.0297,"restaurants":175,"offices":130,"transit":26,"supermarkets":32,"schools":24,"atms":30,"main_roads":22},
        ],
        "competitors":[{"name":"Blinkit","lat":19.0550,"lon":72.8390},{"name":"Zepto","lat":19.1140,"lon":72.8680},{"name":"Swiggy","lat":19.1180,"lon":72.9060},{"name":"Blinkit","lat":19.1870,"lon":72.8470}]
    },
    "Delhi NCR": {
        "center": [28.6139, 77.2090], "zoom": 11,
        "zones": [
            {"name":"Connaught Place",   "lat":28.6315,"lon":77.2167,"restaurants":320,"offices":210,"transit":45,"supermarkets":38,"schools":15,"atms":55,"main_roads":25},
            {"name":"Lajpat Nagar",      "lat":28.5700,"lon":77.2431,"restaurants":275,"offices":95, "transit":35,"supermarkets":52,"schools":22,"atms":42,"main_roads":18},
            {"name":"Gurgaon Cyber City","lat":28.4950,"lon":77.0886,"restaurants":245,"offices":350,"transit":28,"supermarkets":30,"schools":18,"atms":45,"main_roads":24},
            {"name":"Noida Sector 18",   "lat":28.5708,"lon":77.3219,"restaurants":235,"offices":180,"transit":30,"supermarkets":35,"schools":20,"atms":38,"main_roads":20},
            {"name":"Dwarka",            "lat":28.5921,"lon":77.0460,"restaurants":198,"offices":75, "transit":32,"supermarkets":45,"schools":30,"atms":35,"main_roads":18},
            {"name":"Saket",             "lat":28.5244,"lon":77.2090,"restaurants":260,"offices":145,"transit":35,"supermarkets":40,"schools":20,"atms":40,"main_roads":20},
            {"name":"Rohini",            "lat":28.7041,"lon":77.1025,"restaurants":185,"offices":60, "transit":28,"supermarkets":48,"schools":35,"atms":32,"main_roads":16},
            {"name":"Vasant Kunj",       "lat":28.5200,"lon":77.1589,"restaurants":215,"offices":110,"transit":26,"supermarkets":35,"schools":22,"atms":35,"main_roads":17},
        ],
        "competitors":[{"name":"Blinkit","lat":28.6320,"lon":77.2160},{"name":"Zepto","lat":28.5710,"lon":77.2420},{"name":"Swiggy","lat":28.4960,"lon":77.0870}]
    },
    "Hyderabad": {
        "center": [17.3850, 78.4867], "zoom": 12,
        "zones": [
            {"name":"Banjara Hills", "lat":17.4126,"lon":78.4480,"restaurants":285,"offices":145,"transit":22,"supermarkets":40,"schools":18,"atms":42,"main_roads":18},
            {"name":"Gachibowli",    "lat":17.4401,"lon":78.3489,"restaurants":210,"offices":280,"transit":18,"supermarkets":28,"schools":15,"atms":35,"main_roads":20},
            {"name":"Kondapur",      "lat":17.4600,"lon":78.3615,"restaurants":225,"offices":195,"transit":20,"supermarkets":30,"schools":18,"atms":32,"main_roads":18},
            {"name":"Madhapur",      "lat":17.4483,"lon":78.3915,"restaurants":240,"offices":210,"transit":22,"supermarkets":32,"schools":16,"atms":36,"main_roads":17},
            {"name":"Kukatpally",    "lat":17.4849,"lon":78.4138,"restaurants":195,"offices":85, "transit":25,"supermarkets":42,"schools":25,"atms":30,"main_roads":16},
            {"name":"Jubilee Hills", "lat":17.4316,"lon":78.4074,"restaurants":260,"offices":120,"transit":20,"supermarkets":38,"schools":20,"atms":38,"main_roads":17},
            {"name":"Secunderabad",  "lat":17.4399,"lon":78.4983,"restaurants":215,"offices":95, "transit":30,"supermarkets":45,"schools":28,"atms":40,"main_roads":20},
            {"name":"LB Nagar",      "lat":17.3497,"lon":78.5534,"restaurants":165,"offices":55, "transit":22,"supermarkets":35,"schools":22,"atms":25,"main_roads":14},
        ],
        "competitors":[{"name":"Blinkit","lat":17.4130,"lon":78.4470},{"name":"Zepto","lat":17.4490,"lon":78.3900},{"name":"Swiggy","lat":17.4610,"lon":78.3600}]
    },
    "Pune": {
        "center": [18.5204, 73.8567], "zoom": 12,
        "zones": [
            {"name":"Koregaon Park", "lat":18.5362,"lon":73.8938,"restaurants":265,"offices":95, "transit":18,"supermarkets":38,"schools":15,"atms":35,"main_roads":15},
            {"name":"Viman Nagar",   "lat":18.5679,"lon":73.9143,"restaurants":220,"offices":120,"transit":20,"supermarkets":32,"schools":18,"atms":30,"main_roads":16},
            {"name":"Baner",         "lat":18.5590,"lon":73.7868,"restaurants":195,"offices":110,"transit":16,"supermarkets":28,"schools":15,"atms":26,"main_roads":14},
            {"name":"Hadapsar",      "lat":18.5018,"lon":73.9260,"restaurants":175,"offices":85, "transit":18,"supermarkets":30,"schools":20,"atms":22,"main_roads":13},
            {"name":"Kothrud",       "lat":18.5074,"lon":73.8077,"restaurants":185,"offices":65, "transit":22,"supermarkets":42,"schools":28,"atms":30,"main_roads":14},
            {"name":"Wakad",         "lat":18.5975,"lon":73.7600,"restaurants":165,"offices":90, "transit":15,"supermarkets":25,"schools":16,"atms":22,"main_roads":13},
            {"name":"Aundh",         "lat":18.5590,"lon":73.8076,"restaurants":198,"offices":80, "transit":20,"supermarkets":35,"schools":20,"atms":28,"main_roads":15},
            {"name":"Hinjewadi",     "lat":18.5912,"lon":73.7384,"restaurants":155,"offices":165,"transit":14,"supermarkets":22,"schools":14,"atms":20,"main_roads":16},
        ],
        "competitors":[{"name":"Blinkit","lat":18.5370,"lon":73.8930},{"name":"Zepto","lat":18.5680,"lon":73.9130}]
    },
    "Chennai": {
        "center": [13.0827, 80.2707], "zoom": 12,
        "zones": [
            {"name":"T Nagar",            "lat":13.0418,"lon":80.2341,"restaurants":295,"offices":120,"transit":32,"supermarkets":58,"schools":25,"atms":48,"main_roads":18},
            {"name":"Anna Nagar",         "lat":13.0891,"lon":80.2094,"restaurants":245,"offices":98, "transit":28,"supermarkets":48,"schools":28,"atms":42,"main_roads":16},
            {"name":"Adyar",              "lat":13.0063,"lon":80.2574,"restaurants":228,"offices":85, "transit":25,"supermarkets":42,"schools":22,"atms":38,"main_roads":15},
            {"name":"Nungambakkam",       "lat":13.0609,"lon":80.2453,"restaurants":260,"offices":155,"transit":30,"supermarkets":38,"schools":18,"atms":45,"main_roads":17},
            {"name":"Velachery",          "lat":12.9815,"lon":80.2180,"restaurants":210,"offices":95, "transit":26,"supermarkets":35,"schools":20,"atms":32,"main_roads":16},
            {"name":"OMR Sholinganallur", "lat":12.9010,"lon":80.2279,"restaurants":185,"offices":175,"transit":20,"supermarkets":28,"schools":18,"atms":28,"main_roads":18},
            {"name":"Porur",              "lat":13.0358,"lon":80.1566,"restaurants":168,"offices":72, "transit":22,"supermarkets":30,"schools":20,"atms":26,"main_roads":14},
        ],
        "competitors":[{"name":"Blinkit","lat":13.0420,"lon":80.2330},{"name":"Zepto","lat":13.0610,"lon":80.2440},{"name":"Swiggy","lat":13.0070,"lon":80.2560}]
    },
    "Kolkata": {
        "center": [22.5726, 88.3639], "zoom": 12,
        "zones": [
            {"name":"Park Street",  "lat":22.5526,"lon":88.3520,"restaurants":310,"offices":135,"transit":35,"supermarkets":48,"schools":20,"atms":45,"main_roads":18},
            {"name":"Salt Lake",    "lat":22.5765,"lon":88.4149,"restaurants":225,"offices":195,"transit":28,"supermarkets":38,"schools":25,"atms":38,"main_roads":20},
            {"name":"New Town",     "lat":22.5958,"lon":88.4800,"restaurants":185,"offices":165,"transit":22,"supermarkets":28,"schools":18,"atms":30,"main_roads":22},
            {"name":"Ballygunge",   "lat":22.5205,"lon":88.3678,"restaurants":245,"offices":85, "transit":30,"supermarkets":42,"schools":25,"atms":38,"main_roads":15},
            {"name":"Howrah",       "lat":22.5958,"lon":88.2636,"restaurants":198,"offices":75, "transit":38,"supermarkets":40,"schools":28,"atms":32,"main_roads":17},
            {"name":"Dum Dum",      "lat":22.6500,"lon":88.3832,"restaurants":162,"offices":55, "transit":32,"supermarkets":32,"schools":22,"atms":25,"main_roads":15},
            {"name":"Jadavpur",     "lat":22.4990,"lon":88.3720,"restaurants":178,"offices":65, "transit":25,"supermarkets":35,"schools":30,"atms":28,"main_roads":13},
        ],
        "competitors":[{"name":"Blinkit","lat":22.5530,"lon":88.3510},{"name":"Zepto","lat":22.5770,"lon":88.4140}]
    },
    "Surat": {
        "center": [21.1702, 72.8311], "zoom": 12,
        "zones": [
            {"name":"Adajan",   "lat":21.2020,"lon":72.7936,"restaurants":195,"offices":85, "transit":18,"supermarkets":38,"schools":22,"atms":30,"main_roads":15},
            {"name":"Vesu",     "lat":21.1490,"lon":72.7840,"restaurants":175,"offices":72, "transit":15,"supermarkets":32,"schools":18,"atms":26,"main_roads":13},
            {"name":"Athwa",    "lat":21.1830,"lon":72.8194,"restaurants":220,"offices":115,"transit":22,"supermarkets":42,"schools":18,"atms":35,"main_roads":17},
            {"name":"Katargam", "lat":21.2228,"lon":72.8400,"restaurants":185,"offices":65, "transit":20,"supermarkets":38,"schools":24,"atms":28,"main_roads":15},
            {"name":"Varachha", "lat":21.2097,"lon":72.8715,"restaurants":172,"offices":58, "transit":18,"supermarkets":35,"schools":22,"atms":26,"main_roads":14},
            {"name":"Piplod",   "lat":21.1600,"lon":72.7934,"restaurants":165,"offices":68, "transit":16,"supermarkets":30,"schools":20,"atms":25,"main_roads":14},
        ],
        "competitors":[{"name":"Zepto","lat":21.2025,"lon":72.7930},{"name":"Blinkit","lat":21.1835,"lon":72.8190}]
    },
    "Jaipur": {
        "center": [26.9124, 75.7873], "zoom": 12,
        "zones": [
            {"name":"Vaishali Nagar", "lat":26.9124,"lon":75.7315,"restaurants":198,"offices":72, "transit":20,"supermarkets":38,"schools":25,"atms":30,"main_roads":15},
            {"name":"Malviya Nagar",  "lat":26.8535,"lon":75.8104,"restaurants":215,"offices":88, "transit":22,"supermarkets":42,"schools":22,"atms":35,"main_roads":16},
            {"name":"C-Scheme",       "lat":26.9034,"lon":75.8012,"restaurants":235,"offices":130,"transit":25,"supermarkets":40,"schools":18,"atms":42,"main_roads":17},
            {"name":"Mansarovar",     "lat":26.8590,"lon":75.7606,"restaurants":178,"offices":65, "transit":18,"supermarkets":35,"schools":28,"atms":28,"main_roads":14},
            {"name":"Jagatpura",      "lat":26.8200,"lon":75.8320,"restaurants":155,"offices":55, "transit":15,"supermarkets":28,"schools":20,"atms":22,"main_roads":13},
            {"name":"Tonk Road",      "lat":26.8810,"lon":75.8280,"restaurants":182,"offices":78, "transit":20,"supermarkets":32,"schools":18,"atms":28,"main_roads":15},
        ],
        "competitors":[{"name":"Blinkit","lat":26.9030,"lon":75.8000},{"name":"Zepto","lat":26.8540,"lon":75.8100}]
    },
}

SCENARIOS = {
    "⚖️ Balanced":          {"population":1.2,"demand":1.5,"accessibility":1.2,"rent_value":0.8,"comp_gap":1.0,"road":1.0},
    "🚀 Max order density": {"population":0.8,"demand":2.5,"accessibility":1.5,"rent_value":0.5,"comp_gap":0.7,"road":1.2},
    "💰 Minimise cost":     {"population":0.8,"demand":1.0,"accessibility":1.0,"rent_value":2.5,"comp_gap":0.7,"road":0.8},
    "⚔️ Beat competitors":  {"population":1.0,"demand":1.5,"accessibility":1.0,"rent_value":0.8,"comp_gap":2.5,"road":0.9},
    "🌱 Underserved areas": {"population":2.0,"demand":0.8,"accessibility":1.2,"rent_value":1.5,"comp_gap":2.0,"road":0.8},
    "🏢 Enterprise hub":    {"population":1.5,"demand":2.0,"accessibility":1.8,"rent_value":1.0,"comp_gap":0.8,"road":1.5},
    "🎛️ Custom":            None,
}

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371; dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

def normalize(v,lo,hi,invert=False):
    if hi==lo: return 50.0
    n=max(0,min(100,((v-lo)/(hi-lo))*100))
    return round(100-n if invert else n,1)

def score_zone(zone,weights,competitors):
    r=zone.get("restaurants",0); o=zone.get("offices",0); t=zone.get("transit",0)
    s=zone.get("supermarkets",0); sc=zone.get("schools",0); a=zone.get("atms",0); mr=zone.get("main_roads",0)
    demand_score=normalize(r*2.5+o*2.0+t*1.5+s*1.2+sc*0.8+a*1.0, 0, 1200)
    road_score=normalize(mr,0,30); transit_s=normalize(t,0,40)
    access_score=round(road_score*0.65+transit_s*0.35,1)
    pop_score=normalize(s*3+sc*4+a*3+r*0.5,0,500)
    rent_score=round(max(20,min(90,normalize(s*3+o*2+r*0.5,0,1000,invert=True))),1)
    dists=[haversine_km(zone["lat"],zone["lon"],c["lat"],c["lon"]) for c in competitors]
    md=min(dists) if dists else 10
    comp_score=88 if md>=4 else (75 if md>=3 else (58 if md>=2 else (38 if md>=1 else (20 if md>=0.5 else 10))))
    sub={"population":pop_score,"demand":demand_score,"accessibility":access_score,"rent_value":rent_score,"comp_gap":comp_score,"road":road_score}
    tw=sum(weights.values())
    score=round(sum(sub[k]*weights[k] for k in weights)/tw,1)
    delivery=max(8,round(28-(access_score/100)*8-(road_score/100)*5))
    cov=round(2.0+(score/100)*1.5,1)
    orders=round(2500*(demand_score/100)*(0.6+pop_score/100*0.8))
    return {**zone,"score":score,"sub_scores":sub,"delivery_time":delivery,"coverage_km":cov,
            "monthly_orders":orders,"monthly_revenue":orders*350,"nearest_comp_km":round(md,1)}

def build_map(ci,cj,zoom,scored,competitors,sh,sc_,scomp,top_n):
    m=folium.Map(location=[ci,cj],zoom_start=zoom,tiles="CartoDB dark_matter")
    if sh and scored:
        HeatMap([[z["lat"],z["lon"],z["score"]/100] for z in scored],radius=32,blur=28,min_opacity=0.25,
                gradient={"0.2":"#1a237e","0.5":"#ff6f00","0.8":"#e53935","1.0":"#fff"}).add_to(m)
    if scomp:
        cc={"Blinkit":"#fdd835","Zepto":"#ab47bc","Swiggy":"#ff7043"}
        for c in competitors:
            folium.CircleMarker([c["lat"],c["lon"]],radius=7,color=cc.get(c["name"],"#aaa"),
                fill=True,fill_opacity=0.85,tooltip=f"⚠️ {c['name']}").add_to(m)
    tops=scored[:top_n]
    for i,z in enumerate(scored):
        s=z["score"]; it=z in tops
        col="#00e676" if s>=75 else ("#ffb300" if s>=60 else "#ef5350")
        if sc_ and it:
            folium.Circle([z["lat"],z["lon"]],radius=z["coverage_km"]*1000,
                color=col,fill=True,fill_opacity=0.07,weight=1.5,dash_array="6 4").add_to(m)
        ph=f"""<div style='font-family:Arial;width:230px;background:#1a1d27;color:#e0e0e0;padding:12px;border-radius:10px'>
          <b style='font-size:15px;color:{col}'>{"#"+str(i+1)+" " if it else ""}{z["name"]}</b><br>
          <hr style='border-color:#333;margin:6px 0'>
          <b>Score: {s}/100</b><br>⏱ {z["delivery_time"]} min &nbsp;|&nbsp; 📍 {z["coverage_km"]} km<br>
          📦 {z["monthly_orders"]:,} orders/mo &nbsp;|&nbsp; 💰 ₹{z["monthly_revenue"]:,}<br>
          🏪 Nearest competitor: {z["nearest_comp_km"]} km<br>
          <hr style='border-color:#333;margin:6px 0'>
          <small>🍽 {z.get("restaurants","?")} restaurants &nbsp; 🏢 {z.get("offices","?")} offices<br>
          🚌 {z.get("transit","?")} transit &nbsp; 🏪 {z.get("supermarkets","?")} shops</small></div>"""
        folium.CircleMarker([z["lat"],z["lon"]],radius=14 if it else 9,color=col,
            fill=True,fill_opacity=0.9,weight=2,
            popup=folium.Popup(ph,max_width=240),
            tooltip=f"{'🏆 ' if it else ''}{z['name']}: {s}/100").add_to(m)
        if it:
            folium.Marker([z["lat"]+0.003,z["lon"]],icon=folium.DivIcon(
                html=f'<div style="background:{col};color:#000;font-weight:800;font-size:11px;padding:2px 8px;border-radius:10px;white-space:nowrap">#{i+1} {z["name"][:14]}</div>',
                icon_size=(150,24),icon_anchor=(75,0))).add_to(m)
    return m

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 DarkIQ v2")
    st.markdown("*Dark Store Placement Engine*")
    st.markdown('<span style="background:#00c853;color:#000;border-radius:20px;padding:2px 12px;font-size:12px;font-weight:700">● LIVE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🏙️ City")
    city = st.selectbox("", list(CITY_ZONES.keys()), label_visibility="collapsed")
    st.markdown("### 🎯 Scenario")
    scenario = st.selectbox("", list(SCENARIOS.keys()), label_visibility="collapsed")
    weights = SCENARIOS[scenario]
    if scenario == "🎛️ Custom":
        weights = {
            "population":    st.slider("Population",    0.0,3.0,1.2,0.1),
            "demand":        st.slider("Order demand",  0.0,3.0,1.5,0.1),
            "accessibility": st.slider("Accessibility", 0.0,3.0,1.2,0.1),
            "rent_value":    st.slider("Rent value",    0.0,3.0,0.8,0.1),
            "comp_gap":      st.slider("Comp gap",      0.0,3.0,1.0,0.1),
            "road":          st.slider("Road quality",  0.0,3.0,1.0,0.1),
        }
    st.markdown("### 🏅 Top N")
    top_n = st.slider("", 1, 8, 3, label_visibility="collapsed")
    st.markdown("### 🗺️ Layers")
    show_heat = st.toggle("Demand heatmap",     value=True)
    show_cov  = st.toggle("Coverage circles",   value=True)
    show_comp = st.toggle("Competitor markers", value=True)
    st.markdown("---")
    uploaded = st.file_uploader("📤 Upload order CSV (Zone, Index)", type=["csv"])

# ─── Run scoring ──────────────────────────────────────────────────────────────
city_info   = CITY_ZONES[city]
zones       = city_info["zones"]
competitors = city_info["competitors"]

if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
        custom = dict(zip(df_up.iloc[:,0].str.strip(), df_up.iloc[:,1]))
        for z in zones:
            if z["name"] in custom:
                z["restaurants"] = int(custom[z["name"]] * 3.5)
    except: pass

scored = sorted([score_zone(z, weights, competitors) for z in zones], key=lambda x: x["score"], reverse=True)
top    = scored[0]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"# 📦 DarkIQ — {city}")
st.markdown(f"*{scenario} · Top {top_n} of {len(scored)} zones*")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("🏆 Best zone",       top["name"],                   f"{top['score']}/100")
c2.metric("⏱ Delivery",        f"{top['delivery_time']} min", "est. avg")
c3.metric("📍 Coverage",        f"{top['coverage_km']} km",    "radius")
c4.metric("📦 Orders/month",    f"{top['monthly_orders']:,}",  "est.")
c5.metric("💰 Revenue/month",   f"₹{top['monthly_revenue']:,}","est.")

st.markdown("---")

# ─── Map + rankings ───────────────────────────────────────────────────────────
mc, rc = st.columns([3,2])
with mc:
    st.markdown("### 🗺️ Placement map")
    st.caption("Click any marker for full breakdown · Yellow = Blinkit · Purple = Zepto · Orange = Swiggy")
    m = build_map(city_info["center"][0], city_info["center"][1], city_info["zoom"],
                  scored, competitors, show_heat, show_cov, show_comp, top_n)
    st_folium(m, width=None, height=540, returned_objects=[])

with rc:
    st.markdown("### 🏅 Top locations")
    bar_cols = {"population":"#6c63ff","demand":"#00b0ff","accessibility":"#00e676",
                "rent_value":"#ffb300","comp_gap":"#ff6d00","road":"#ea80fc"}
    labels   = {"population":"Population","demand":"Demand","accessibility":"Access",
                "rent_value":"Rent","comp_gap":"Comp gap","road":"Roads"}

    for i, z in enumerate(scored[:top_n]):
        s   = z["score"]
        col = "#00e676" if s>=75 else ("#ffb300" if s>=60 else "#ef5350")
        cls = "score-high" if s>=75 else ("score-med" if s>=60 else "score-low")
        bars = "".join(f"""<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>
          <span style='font-size:11px;color:#aaa;width:70px;flex-shrink:0'>{labels[f]}</span>
          <div style='flex:1;background:#2a2d3e;border-radius:4px;height:7px;overflow:hidden'>
            <div style='width:{v}%;background:{bar_cols[f]};height:7px;border-radius:4px'></div></div>
          <span style='font-size:11px;color:#ccc;width:26px;text-align:right'>{v:.0f}</span></div>"""
          for f,v in z["sub_scores"].items())
        st.markdown(f"""<div class="metric-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div><span class="badge-rank">#{i+1}</span>
            <b style="font-size:17px;margin-left:8px">{z["name"]}</b></div>
            <div class="{cls}">{s}</div></div>
          <div style="color:#aaa;font-size:12px;margin:6px 0 10px">
            ⏱ {z["delivery_time"]} min &nbsp;|&nbsp; 📍 {z["coverage_km"]} km &nbsp;|&nbsp;
            📦 {z["monthly_orders"]:,}/mo &nbsp;|&nbsp; 🏪 {z["nearest_comp_km"]} km gap</div>
          {bars}</div>""", unsafe_allow_html=True)

    if len(scored) > top_n:
        nxt = scored[top_n]
        st.info(f"📉 Gap to #{top_n+1} ({nxt['name']}): **{round(scored[top_n-1]['score']-nxt['score'],1)} pts**")

# ─── Table ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 All zones")
df = pd.DataFrame([{"Rank":f"#{i+1}","Zone":z["name"],"Score":z["score"],
    "Restaurants":z.get("restaurants",0),"Offices":z.get("offices",0),
    "Transit":z.get("transit",0),"Shops":z.get("supermarkets",0),
    "Nearest comp":f"{z['nearest_comp_km']} km","Delivery":f"{z['delivery_time']} min",
    "Coverage":f"{z['coverage_km']} km","Orders/mo":f"{z['monthly_orders']:,}",
    "Revenue/mo":f"₹{z['monthly_revenue']:,}"} for i,z in enumerate(scored)])

def cs(val):
    try:
        v=float(val)
        return ("background:#1b3a2a;color:#00e676" if v>=75 else
                "background:#3a2e10;color:#ffb300" if v>=60 else "background:#3a1a1a;color:#ef5350")
    except: return ""

st.dataframe(df.style.applymap(cs,subset=["Score"]),use_container_width=True,hide_index=True)

# ─── What-if ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔀 What-if scenario comparison")
sc_list=[s for s in SCENARIOS if s!="🎛️ Custom"]
ca,cb=st.columns(2)
sa=ca.selectbox("Scenario A",sc_list,index=0,key="sa")
sb=cb.selectbox("Scenario B",sc_list,index=1,key="sb")
sa_s={z["name"]:score_zone(z,SCENARIOS[sa],competitors)["score"] for z in zones}
sb_s={z["name"]:score_zone(z,SCENARIOS[sb],competitors)["score"] for z in zones}
cdf=pd.DataFrame([{"Zone":n,f"A ({sa.split()[1]})":sa_s[n],f"B ({sb.split()[1]})":sb_s[n],"Diff":round(sa_s[n]-sb_s[n],1)} for n in sa_s]).sort_values(f"A ({sa.split()[1]})",ascending=False)
def cd(val):
    try:
        v=float(val)
        return "color:#00e676" if v>3 else ("color:#ef5350" if v<-3 else "color:#aaa")
    except: return ""
st.dataframe(cdf.style.applymap(cd,subset=["Diff"]),use_container_width=True,hide_index=True)

# ─── Export ───────────────────────────────────────────────────────────────────
st.markdown("---")
edf=pd.DataFrame([{"Rank":f"#{i+1}","City":city,"Zone":z["name"],"Score":z["score"],
    "Delivery (min)":z["delivery_time"],"Coverage (km)":z["coverage_km"],
    "Monthly orders":z["monthly_orders"],"Monthly revenue (₹)":z["monthly_revenue"],
    "Nearest competitor (km)":z["nearest_comp_km"],"Restaurants":z.get("restaurants",""),
    "Offices":z.get("offices",""),"Scenario":scenario} for i,z in enumerate(scored)])
st.download_button("⬇️ Download report (CSV)",data=edf.to_csv(index=False).encode(),
    file_name=f"darkiq_{city.lower().replace(' ','_')}.csv",mime="text/csv",use_container_width=True)
st.caption("DarkIQ v2 · 9 Indian cities · Bengaluru · Mumbai · Delhi · Hyderabad · Pune · Chennai · Kolkata · Surat · Jaipur")
