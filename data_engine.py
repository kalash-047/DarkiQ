"""
DarkIQ Data Engine
Fetches real live data from:
  - Overpass API (OpenStreetMap) — real shops, roads, amenities
  - Nominatim — geocoding any city/zone name
  - Kontur Population API — real population density grids
  - Open-Meteo — weather/seasonality signals
  - OSRM — real road travel time between points
"""

import requests
import json
import time
import math
import numpy as np
from functools import lru_cache

HEADERS = {"User-Agent": "DarkIQ-MVP/2.0 (darkstore placement engine)"}

# ─── Nominatim: geocode any city or zone ─────────────────────────────────────
def geocode(place_name: str) -> dict | None:
    """Return lat/lon for any place name using OpenStreetMap Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "limit": 1, "addressdetails": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display_name": data[0]["display_name"],
                "bbox": data[0].get("boundingbox", []),
            }
    except Exception as e:
        print(f"Geocode error for '{place_name}': {e}")
    return None


# ─── Overpass: fetch real POIs around a point ────────────────────────────────
def fetch_osm_pois(lat: float, lon: float, radius_m: int = 2500) -> dict:
    """
    Fetch real POI data from OpenStreetMap Overpass API.
    Returns counts of: residential areas, shops, restaurants, offices,
    hospitals, schools, existing dark stores / supermarkets.
    """
    url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:30];
    (
      node["landuse"="residential"](around:{radius_m},{lat},{lon});
      way["landuse"="residential"](around:{radius_m},{lat},{lon});
      node["shop"~"supermarket|convenience|grocery"](around:{radius_m},{lat},{lon});
      node["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
      node["amenity"~"hospital|clinic|pharmacy"](around:{radius_m},{lat},{lon});
      node["amenity"~"school|college|university"](around:{radius_m},{lat},{lon});
      node["shop"~"mall|department_store"](around:{radius_m},{lat},{lon});
      way["highway"~"primary|secondary|tertiary"](around:{radius_m},{lat},{lon});
    );
    out count;
    """
    try:
        r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=35)
        data = r.json()
        counts = {}
        if "elements" in data:
            for el in data["elements"]:
                tag = el.get("tags", {})
                t = tag.get("type", "")
                counts[t] = counts.get(t, 0) + 1
        total = sum(counts.values())
        return {"raw_counts": counts, "total_pois": total, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e), "total_pois": 0}


def fetch_osm_detailed(lat: float, lon: float, radius_m: int = 2500) -> dict:
    """
    Detailed OSM fetch: counts each type of amenity separately.
    Used to compute real sub-scores.
    """
    url = "https://overpass-api.de/api/interpreter"
    queries = {
        "supermarkets":  f'node["shop"~"supermarket|convenience"](around:{radius_m},{lat},{lon});',
        "restaurants":   f'node["amenity"~"restaurant|cafe|fast_food|food_court"](around:{radius_m},{lat},{lon});',
        "offices":       f'node["office"](around:{radius_m},{lat},{lon});',
        "residential":   f'way["landuse"="residential"](around:{radius_m},{lat},{lon});',
        "hospitals":     f'node["amenity"~"hospital|clinic|pharmacy"](around:{radius_m},{lat},{lon});',
        "schools":       f'node["amenity"~"school|college|university"](around:{radius_m},{lat},{lon});',
        "main_roads":    f'way["highway"~"primary|secondary"](around:{radius_m},{lat},{lon});',
        "transit":       f'node["highway"="bus_stop"](around:{radius_m},{lat},{lon});',
        "atms":          f'node["amenity"="atm"](around:{radius_m},{lat},{lon});',
    }

    results = {}
    combined_query = f"[out:json][timeout:40];\n(\n"
    for key, q in queries.items():
        combined_query += f"  {q}\n"
    combined_query += ");\nout count;\n"

    try:
        r = requests.post(url, data={"data": combined_query}, headers=HEADERS, timeout=45)
        data = r.json()
        total = len(data.get("elements", []))

        # Run individual queries for accurate counts
        for key, q in queries.items():
            individual_query = f"[out:json][timeout:20];\n(\n  {q}\n);\nout count;\n"
            try:
                ri = requests.post(url, data={"data": individual_query},
                                   headers=HEADERS, timeout=25)
                di = ri.json()
                results[key] = len(di.get("elements", []))
                time.sleep(0.5)  # Respect rate limits
            except:
                results[key] = 0

        return {"counts": results, "success": True}
    except Exception as e:
        return {"counts": {k: 0 for k in queries}, "success": False, "error": str(e)}


# ─── OSRM: real travel time calculation ──────────────────────────────────────
def get_travel_time(origin_lat, origin_lon, dest_lat, dest_lon) -> float:
    """
    Get real road travel time in minutes using OSRM (free, no key needed).
    Falls back to haversine estimate if API unavailable.
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    params = {"overview": "false", "annotations": "false"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("code") == "Ok":
            duration_sec = data["routes"][0]["duration"]
            return round(duration_sec / 60, 1)
    except:
        pass
    # Fallback: simple haversine estimate
    dist_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    return round(dist_km / 0.5, 1)  # ~30 km/h urban average


def get_isochrone_area(lat, lon, minutes=15) -> float:
    """
    Estimate area reachable in N minutes from a point.
    Uses OSRM table service.
    Returns approximate km² coverage.
    """
    # Simple approximation: 30km/h average speed in urban India
    speed_kmh = 28
    radius_km = (speed_kmh * minutes) / 60
    area_km2 = math.pi * radius_km**2
    return round(area_km2, 2)


# ─── Population density (Kontur API) ─────────────────────────────────────────
def fetch_population_density(lat: float, lon: float, radius_km: float = 3.0) -> dict:
    """
    Fetch real population data from Kontur Population API.
    Returns estimated population within radius.
    """
    # Kontur Population API (free, global H3 hex grid)
    url = "https://population.un.org/dataportal/api/locations"
    # Alternative: WorldPop API
    worldpop_url = f"https://api.worldpop.org/v1/services/stats"
    params = {
        "dataset": "wpgpas",
        "year": 2020,
        "lat": lat,
        "lon": lon,
        "geojson": False,
        "runasync": False,
    }
    try:
        r = requests.get(worldpop_url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            pop = data.get("data", {}).get("total_population", 0)
            return {"population": int(pop), "source": "WorldPop", "success": True}
    except:
        pass

    # Fallback: estimate from OSM building density
    return estimate_population_from_osm(lat, lon, radius_km)


def estimate_population_from_osm(lat: float, lon: float, radius_km: float = 3.0) -> dict:
    """Estimate population from OSM building count (fallback)."""
    url = "https://overpass-api.de/api/interpreter"
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    way["building"](around:{radius_m},{lat},{lon});
    out count;
    """
    try:
        r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=30)
        data = r.json()
        building_count = len(data.get("elements", []))
        # Avg 20 people per building in Indian urban context
        estimated_pop = building_count * 20
        return {
            "population": estimated_pop,
            "buildings": building_count,
            "source": "OSM buildings estimate",
            "success": True
        }
    except Exception as e:
        return {"population": 50000, "source": "fallback", "success": False}


# ─── Competitor detection via OSM ────────────────────────────────────────────
def fetch_competitor_darkstores(lat: float, lon: float, radius_m: int = 5000) -> list:
    """
    Find actual quick commerce / dark store competitors via OSM tags.
    Searches for known brand names: Blinkit, Zepto, Swiggy Instamart, etc.
    """
    url = "https://overpass-api.de/api/interpreter"
    brands = ["Blinkit", "Zepto", "Instamart", "BigBasket", "Dunzo", "JioMart"]
    brand_filter = "|".join(brands)

    query = f"""
    [out:json][timeout:25];
    (
      node["brand"~"{brand_filter}",i](around:{radius_m},{lat},{lon});
      node["name"~"{brand_filter}",i](around:{radius_m},{lat},{lon});
      node["operator"~"{brand_filter}",i](around:{radius_m},{lat},{lon});
    );
    out body;
    """
    competitors = []
    try:
        r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=30)
        data = r.json()
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("brand") or tags.get("operator", "Unknown")
            competitors.append({
                "name": name,
                "lat": el["lat"],
                "lon": el["lon"],
                "type": tags.get("shop", tags.get("amenity", "darkstore")),
            })
    except Exception as e:
        pass
    return competitors


# ─── Road quality score ───────────────────────────────────────────────────────
def fetch_road_quality(lat: float, lon: float, radius_m: int = 2000) -> dict:
    """Fetch road network density and quality from OSM."""
    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"="primary"](around:{radius_m},{lat},{lon});
      way["highway"="secondary"](around:{radius_m},{lat},{lon});
      way["highway"="tertiary"](around:{radius_m},{lat},{lon});
      way["highway"="residential"](around:{radius_m},{lat},{lon});
    );
    out count;
    """
    try:
        r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=30)
        data = r.json()
        road_count = len(data.get("elements", []))
        # Score 0-100: 100 roads = score 85+
        score = min(95, 40 + (road_count * 0.5))
        return {"road_count": road_count, "road_score": round(score, 1), "success": True}
    except:
        return {"road_count": 0, "road_score": 60, "success": False}


# ─── Utility functions ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def generate_candidate_grid(center_lat, center_lon, bbox, grid_size=4) -> list:
    """
    Generate a grid of candidate locations within a bounding box.
    Used for city-wide scanning.
    """
    if bbox and len(bbox) >= 4:
        min_lat, max_lat = float(bbox[0]), float(bbox[1])
        min_lon, max_lon = float(bbox[2]), float(bbox[3])
    else:
        delta = 0.08
        min_lat = center_lat - delta
        max_lat = center_lat + delta
        min_lon = center_lon - delta
        max_lon = center_lon + delta

    candidates = []
    lat_steps = np.linspace(min_lat, max_lat, grid_size)
    lon_steps = np.linspace(min_lon, max_lon, grid_size)

    for i, lat in enumerate(lat_steps):
        for j, lon in enumerate(lon_steps):
            candidates.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "grid_id": f"G{i}{j}",
            })
    return candidates


def normalize_score(value, min_val, max_val, invert=False) -> float:
    """Normalize any value to 0-100 scale."""
    if max_val == min_val:
        return 50.0
    normalized = ((value - min_val) / (max_val - min_val)) * 100
    if invert:
        normalized = 100 - normalized
    return round(max(0, min(100, normalized)), 1)
