# DarkIQ v2 — Deployment Guide
### Real-time dark store placement engine · Works for ANY city worldwide

---

## What makes v2 different from v1

| Feature | v1 (yesterday) | v2 (this version) |
|---|---|---|
| City data | 6 hardcoded Indian cities | **ANY city in the world** |
| Zone data | Pre-loaded static data | **Live from OpenStreetMap** |
| POI data | Estimated numbers | **Real restaurant/office/transit counts** |
| Competitor detection | Manual pins | **Live OSM brand search** |
| Coverage | Fixed | **Calculated from real scores** |

---

## Files you have

- `app.py` — main Streamlit app (full real-time engine)
- `data_engine.py` — data fetching layer (OSM, Nominatim, OSRM)
- `scoring.py` — scoring and ranking engine
- `requirements.txt` — dependencies

---

## Deploy in 4 steps (same as before, 20 minutes)

### Step 1: GitHub
1. Go to github.com → create repo `darkiq-v2` (public)
2. Upload ALL 4 files

### Step 2: Streamlit Cloud
1. Go to share.streamlit.io
2. New app → point to `darkiq-v2` repo → `app.py`
3. Deploy → wait 3 minutes

### Step 3: Test
Try these cities:
- `Bengaluru, India`
- `Kolkata, India`
- `Surat, India`
- `Jaipur, India`
- `Dubai, UAE`
- `London, UK`

All will load REAL neighborhoods with REAL POI data.

### Step 4: Share
You now have a live link that works for any city a customer mentions.

---

## What the live data actually shows

When a customer picks "Koramangala, Bengaluru":
- Real count of restaurants in 2.5km radius (from OSM)
- Real count of offices, transit stops, ATMs (from OSM)
- Real road network density (primary + secondary roads)
- Competitor scan (searches for Blinkit, Zepto, Instamart names in OSM)
- Score calculated from all of the above

This is NOT demo data. This is real.

---

## Demo script for customers (5 minutes)

1. Open the app
2. Type their city → Load city
3. Watch it fetch live data ("📡 Analysing Koramangala...")
4. Show the map — point to top 3 locations
5. Click a marker → show the breakdown (restaurants: 47, offices: 12, etc.)
6. Switch scenario from "Balanced" to "Beat competitors" → show rankings shift
7. Download the CSV report → send it to them right there

Key line to say:
> "This data is live from OpenStreetMap — those restaurant and office counts
> are real, pulled 30 seconds ago. Your ops team can run this for any city
> before deciding where to sign a lease."

---

## Troubleshooting

**"No neighborhoods found"**
→ Add country to city name: "Pune, India" not just "Pune"

**Zones load slowly**
→ Normal — it's fetching real live data. First load takes 60-90 seconds.
→ After that, cached for 30 minutes.

**Some zones show ⚠️ instead of ✅**
→ Overpass API rate limit hit. Wait 60 seconds and reload.
→ All scores still calculated, some from cached data.

---

## Upgrade path (v3 — with paying customer)

Add to requirements.txt:
```
googlemaps  # real traffic data
```

Add to Streamlit Cloud secrets:
```
GOOGLE_MAPS_KEY = "your-key"
MAPPLS_KEY = "your-key"
```

This unlocks:
- Live traffic-adjusted delivery time estimates
- Real-time congestion data
- More accurate road speed calculations

Cost: ~$50-200/month Google Maps API at MVP scale (recover from first customer)

---

## For grant application

**Problem statement:**
"Quick commerce companies in India spend ₹2-5 crore per dark store placement,
yet most location decisions are made using spreadsheets and intuition. DarkIQ
is a real-time geo-intelligence platform that pulls live data from OpenStreetMap
to score candidate locations across six factors: population density, demand
signals, road accessibility, rental cost proxy, competitor gaps, and road
quality. The platform works for any city worldwide and delivers ranked
recommendations with estimated delivery times and monthly revenue projections."

**Traction to include:**
- Link to live app
- Screenshot of real data for a specific city
- CSV report downloaded from the app
- Quote from any ops manager who saw it

---

*DarkIQ v2 · Real-time · Any city · Built for operators*
