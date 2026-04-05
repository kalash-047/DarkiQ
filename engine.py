"""
DistrictIQ Data Engine v3
Real-time data from:
  - Valhalla (OSM public instance) — isochrones, real delivery zones
  - OSRM — actual road travel times between points
  - Overpass API — live outlet/kirana/shop density
  - Nominatim — geocoding any city/zone
All with multi-tier fallback so the app never breaks.
"""

import requests
import json
import math
import time
import hashlib
import numpy as np
from typing import Optional, List, Dict, Tuple
from functools import lru_cache
import streamlit as st

# ── API Endpoints ─────────────────────────────────────────────────────────────
VALHALLA_ENDPOINTS = [
    "https://valhalla1.openstreetmap.de",
    "https://valhalla.openstreetmap.de",
]

OSRM_ENDPOINTS = [
    "http://router.project-osrm.org",
    "https://routing.openstreetmap.de/routed-car",
]

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "DistrictIQ/3.0 (FMCG distribution intelligence; contact@districtiq.in)",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def circle_polygon(lat, lon, radius_km, n_points=32):
    """Generate circle polygon as GeoJSON coordinates."""
    coords = []
    for i in range(n_points + 1):
        angle = 2 * math.pi * i / n_points
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
        coords.append([lon + dlon * math.cos(angle), lat + dlat * math.sin(angle)])
    return coords

# ── Nominatim Geocoding ───────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def geocode(place: str) -> Optional[Dict]:
    """Geocode any place name to lat/lon."""
    try:
        r = requests.get(NOMINATIM_URL,
            params={"q": place, "format": "json", "limit": 1, "addressdetails": 1},
            headers=HEADERS, timeout=10)
        data = r.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0]["display_name"],
                "bbox": [float(x) for x in data[0].get("boundingbox", [])],
            }
    except Exception as e:
        pass
    return None

# ── Valhalla Isochrones ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_isochrone(lat: float, lon: float, minutes: int = 15,
                  costing: str = "motor_scooter") -> Dict:
    """
    Get real drivable isochrone polygon from Valhalla.
    motor_scooter = Indian delivery mode (2-wheeler through narrow lanes).
    Falls back to circle with road network efficiency factor.
    """
    payload = {
        "locations": [{"lat": lat, "lon": lon}],
        "costing": costing,
        "contours": [{"time": minutes, "color": "ff0000"}],
        "polygons": True,
        "generalize": 100,
        "show_locations": True,
    }

    for endpoint in VALHALLA_ENDPOINTS:
        try:
            r = requests.post(
                f"{endpoint}/isochrone",
                json=payload,
                headers=HEADERS,
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                if features:
                    geom = features[0]["geometry"]
                    coords = geom["coordinates"][0]
                    area_km2 = _polygon_area_km2(coords)
                    return {
                        "type": "isochrone",
                        "coordinates": coords,
                        "area_km2": area_km2,
                        "effective_radius_km": round(math.sqrt(area_km2 / math.pi), 2),
                        "source": "valhalla",
                        "costing": costing,
                        "minutes": minutes,
                    }
        except Exception:
            continue

    # Fallback: road-network-adjusted circle
    # Indian urban roads have ~65% efficiency vs straight-line
    road_efficiency = 0.65
    avg_speed_kmh   = 25  # motor scooter in Indian urban traffic
    radius_km       = (avg_speed_kmh * minutes / 60) * road_efficiency
    coords          = circle_polygon(lat, lon, radius_km)
    area_km2        = math.pi * radius_km ** 2

    return {
        "type": "circle_fallback",
        "coordinates": coords,
        "area_km2": round(area_km2, 2),
        "effective_radius_km": round(radius_km, 2),
        "source": "fallback",
        "costing": costing,
        "minutes": minutes,
    }

def get_multi_contour_isochrone(lat: float, lon: float,
                                 contours: List[int] = [10, 20, 30]) -> List[Dict]:
    """Get multiple time contours for a single point."""
    payload = {
        "locations": [{"lat": lat, "lon": lon}],
        "costing": "motor_scooter",
        "contours": [{"time": t, "color": "ff0000"} for t in contours],
        "polygons": True,
        "generalize": 100,
    }
    results = []
    for endpoint in VALHALLA_ENDPOINTS:
        try:
            r = requests.post(f"{endpoint}/isochrone", json=payload,
                              headers=HEADERS, timeout=25)
            if r.status_code == 200:
                data = r.json()
                for i, feature in enumerate(data.get("features", [])):
                    coords = feature["geometry"]["coordinates"][0]
                    results.append({
                        "minutes": contours[i] if i < len(contours) else contours[-1],
                        "coordinates": coords,
                        "area_km2": _polygon_area_km2(coords),
                        "source": "valhalla",
                    })
                return results
        except Exception:
            continue
    # Fallback
    for t in contours:
        r_km = (25 * t / 60) * 0.65
        results.append({
            "minutes": t,
            "coordinates": circle_polygon(lat, lon, r_km),
            "area_km2": round(math.pi * r_km**2, 2),
            "source": "fallback",
        })
    return results

# ── OSRM Travel Times ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_travel_time_matrix(origins: List[Tuple], destinations: List[Tuple]) -> np.ndarray:
    """
    Get real travel time matrix using OSRM table service.
    Returns matrix[i][j] = travel time in minutes from origin i to destination j.
    """
    all_coords = list(origins) + list(destinations)
    coord_str  = ";".join(f"{lon},{lat}" for lat, lon in all_coords)
    sources    = ";".join(str(i) for i in range(len(origins)))
    dests      = ";".join(str(i + len(origins)) for i in range(len(destinations)))

    for endpoint in OSRM_ENDPOINTS:
        try:
            url = f"{endpoint}/table/v1/driving/{coord_str}"
            r   = requests.get(url,
                params={"sources": sources, "destinations": dests, "annotations": "duration"},
                headers={"User-Agent": HEADERS["User-Agent"]},
                timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok":
                    matrix = np.array(data["durations"]) / 60  # convert to minutes
                    return matrix
        except Exception:
            continue

    # Fallback: haversine-based estimate (30 km/h average)
    matrix = np.zeros((len(origins), len(destinations)))
    for i, (olat, olon) in enumerate(origins):
        for j, (dlat, dlon) in enumerate(destinations):
            dist = haversine(olat, olon, dlat, dlon)
            matrix[i][j] = dist / 0.5  # 30 km/h = 0.5 km/min
    return matrix

@st.cache_data(ttl=3600, show_spinner=False)
def get_route(lat1, lon1, lat2, lon2) -> Dict:
    """Get actual road route between two points."""
    for endpoint in OSRM_ENDPOINTS:
        try:
            url = f"{endpoint}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
            r   = requests.get(url,
                params={"overview": "full", "geometries": "geojson", "annotations": "false"},
                headers={"User-Agent": HEADERS["User-Agent"]},
                timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok":
                    route = data["routes"][0]
                    return {
                        "duration_min": round(route["duration"] / 60, 1),
                        "distance_km":  round(route["distance"] / 1000, 2),
                        "geometry":     route["geometry"]["coordinates"],
                        "source":       "osrm",
                    }
        except Exception:
            continue
    dist = haversine(lat1, lon1, lat2, lon2)
    return {
        "duration_min": round(dist / 0.5, 1),
        "distance_km":  round(dist, 2),
        "geometry":     [[lon1, lat1], [lon2, lat2]],
        "source":       "fallback",
    }

# ── Overpass: Live Outlet Data ────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_outlets_in_bbox(min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
    """
    Fetch all retail outlets (kirana/grocery/supermarket/convenience)
    within a bounding box from OpenStreetMap.
    Returns list of {lat, lon, name, type, tags}
    """
    query = f"""
    [out:json][timeout:35];
    (
      node["shop"~"convenience|grocery|supermarket|general|kirana|provisions|food"](
        {min_lat},{min_lon},{max_lat},{max_lon});
      node["amenity"~"marketplace|pharmacy|fuel"](
        {min_lat},{min_lon},{max_lat},{max_lon});
      node["shop"="bakery"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    """
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": query},
                              headers={"User-Agent": HEADERS["User-Agent"]},
                              timeout=40)
            if r.status_code == 200:
                data  = r.json()
                elems = data.get("elements", [])
                outlets = []
                for el in elems:
                    tags = el.get("tags", {})
                    outlets.append({
                        "lat":  el["lat"],
                        "lon":  el["lon"],
                        "name": tags.get("name", "Unnamed"),
                        "type": tags.get("shop") or tags.get("amenity", "unknown"),
                        "brand":tags.get("brand", ""),
                    })
                return outlets
        except Exception:
            continue
    return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_poi_counts_radius(lat: float, lon: float, radius_m: int = 2000) -> Dict:
    """
    Fetch categorised POI counts around a point.
    Used for demand signal computation.
    """
    query = f"""
    [out:json][timeout:35];
    (
      node["shop"~"convenience|grocery|supermarket|kirana"](around:{radius_m},{lat},{lon});
      node["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
      node["highway"="bus_stop"](around:{radius_m},{lat},{lon});
      node["amenity"="atm"](around:{radius_m},{lat},{lon});
      node["amenity"~"school|college"](around:{radius_m},{lat},{lon});
      node["amenity"~"hospital|clinic|pharmacy"](around:{radius_m},{lat},{lon});
      way["building"~"residential|apartments"](around:{radius_m},{lat},{lon});
    );
    out count;
    """
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": query},
                              headers={"User-Agent": HEADERS["User-Agent"]},
                              timeout=40)
            if r.status_code == 200:
                data = r.json()
                elems = data.get("elements", [])
                counts = {}
                for el in elems:
                    tags = el.get("tags", {})
                    shop    = tags.get("shop", "")
                    amenity = tags.get("amenity", "")
                    highway = tags.get("highway", "")
                    building= tags.get("building", "")
                    if shop in ["convenience","grocery","supermarket","kirana","provisions"]:
                        counts["retail_outlets"] = counts.get("retail_outlets", 0) + 1
                    elif amenity in ["restaurant","cafe","fast_food"]:
                        counts["restaurants"] = counts.get("restaurants", 0) + 1
                    elif "office" in tags:
                        counts["offices"] = counts.get("offices", 0) + 1
                    elif highway == "bus_stop":
                        counts["transit"] = counts.get("transit", 0) + 1
                    elif amenity == "atm":
                        counts["atms"] = counts.get("atms", 0) + 1
                    elif amenity in ["school","college"]:
                        counts["schools"] = counts.get("schools", 0) + 1
                    elif amenity in ["hospital","clinic","pharmacy"]:
                        counts["healthcare"] = counts.get("healthcare", 0) + 1
                    elif building in ["residential","apartments"]:
                        counts["residential"] = counts.get("residential", 0) + 1
                return {"counts": counts, "source": "overpass", "success": True}
        except Exception:
            continue
    return {"counts": {}, "source": "fallback", "success": False}

# ── Distribution Unit (Hub) Analyser ─────────────────────────────────────────
def analyse_distribution_unit(
    hub_lat: float, hub_lon: float, hub_name: str,
    outlets: List[Dict], travel_minutes: int = 30
) -> Dict:
    """
    For a given distribution hub, compute:
    - Realistic delivery zone (Valhalla isochrone)
    - Outlets within zone
    - Travel times to all outlets (OSRM)
    - Coverage metrics
    - Network efficiency score
    """
    # Get real delivery zone
    isochrone = get_isochrone(hub_lat, hub_lon, minutes=travel_minutes)

    # Filter outlets within isochrone bounding box first
    if isochrone["coordinates"]:
        lons = [c[0] for c in isochrone["coordinates"]]
        lats = [c[1] for c in isochrone["coordinates"]]
        bbox = (min(lats), min(lons), max(lats), max(lons))
    else:
        r = isochrone["effective_radius_km"]
        bbox = (hub_lat - r/111, hub_lon - r/111,
                hub_lat + r/111, hub_lon + r/111)

    outlets_in_bbox = [
        o for o in outlets
        if bbox[0] <= o["lat"] <= bbox[2] and bbox[1] <= o["lon"] <= bbox[3]
    ]

    # Point-in-polygon check using ray casting
    outlets_in_zone = [
        o for o in outlets_in_bbox
        if _point_in_polygon(o["lat"], o["lon"], isochrone["coordinates"])
    ]

    # Travel times to outlets in zone
    if outlets_in_zone and len(outlets_in_zone) <= 50:
        origins = [(hub_lat, hub_lon)]
        dests   = [(o["lat"], o["lon"]) for o in outlets_in_zone[:50]]
        try:
            time_matrix = get_travel_time_matrix(origins, dests)
            travel_times = time_matrix[0].tolist()
        except Exception:
            travel_times = [
                haversine(hub_lat, hub_lon, o["lat"], o["lon"]) / 0.5
                for o in outlets_in_zone
            ]
    else:
        travel_times = [
            haversine(hub_lat, hub_lon, o["lat"], o["lon"]) / 0.5
            for o in outlets_in_zone
        ]

    # Metrics
    avg_tt   = round(np.mean(travel_times), 1) if travel_times else 0
    max_tt   = round(np.max(travel_times), 1) if travel_times else 0
    p90_tt   = round(np.percentile(travel_times, 90), 1) if travel_times else 0
    n_outlets = len(outlets_in_zone)

    # Orders/day capacity estimate (1 rep = 30 visits/day, avg 20min/visit)
    reps_needed = max(1, math.ceil(n_outlets / 25))
    orders_per_day = n_outlets * 0.7  # 70% active rate

    return {
        "hub_name":      hub_name,
        "hub_lat":       hub_lat,
        "hub_lon":       hub_lon,
        "isochrone":     isochrone,
        "outlets_in_zone": outlets_in_zone,
        "outlet_count":  n_outlets,
        "avg_travel_min":avg_tt,
        "max_travel_min":max_tt,
        "p90_travel_min":p90_tt,
        "reps_needed":   reps_needed,
        "orders_per_day":round(orders_per_day),
        "area_km2":      isochrone["area_km2"],
        "outlet_density":round(n_outlets / max(0.1, isochrone["area_km2"]), 1),
        "travel_times":  travel_times,
        "data_source":   isochrone["source"],
    }

# ── Utility ───────────────────────────────────────────────────────────────────
def _point_in_polygon(lat, lon, polygon_coords):
    """Ray casting algorithm for point-in-polygon."""
    x, y  = lon, lat
    inside = False
    n = len(polygon_coords)
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _polygon_area_km2(coords):
    """Shoelace formula for polygon area."""
    n   = len(coords)
    if n < 3: return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        # Convert to km
        x1 = coords[i][0] * 111.32 * math.cos(math.radians(coords[i][1]))
        y1 = coords[i][1] * 111.32
        x2 = coords[j][0] * 111.32 * math.cos(math.radians(coords[j][1]))
        y2 = coords[j][1] * 111.32
        area += x1 * y2 - x2 * y1
    return round(abs(area) / 2, 2)

def normalize(v, lo, hi, invert=False):
    if hi == lo: return 50.0
    n = max(0, min(100, ((v - lo) / (hi - lo)) * 100))
    return round(100 - n if invert else n, 1)
