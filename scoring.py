"""
DarkIQ Scoring Engine
Converts real OSM/population data into placement scores.
Fully explainable — every score has a reason.
"""

import math
import numpy as np
from data_engine import (
    normalize_score, haversine_km, get_isochrone_area,
    fetch_osm_detailed, fetch_road_quality,
    fetch_competitor_darkstores, fetch_population_density,
)


# ─── Score weights per scenario ──────────────────────────────────────────────
SCENARIOS = {
    "⚖️ Balanced": {
        "description": "Best overall placement — balanced across all factors",
        "weights": {
            "population":    1.2,
            "demand_proxy":  1.5,
            "accessibility": 1.2,
            "rent_value":    0.8,
            "comp_gap":      1.0,
            "road_quality":  1.0,
        }
    },
    "🚀 Maximise order density": {
        "description": "Prioritise zones with highest existing demand signals",
        "weights": {
            "population":    0.8,
            "demand_proxy":  2.5,
            "accessibility": 1.5,
            "rent_value":    0.5,
            "comp_gap":      0.7,
            "road_quality":  1.2,
        }
    },
    "💰 Minimise cost": {
        "description": "Optimise for lowest rent + operational cost",
        "weights": {
            "population":    0.8,
            "demand_proxy":  1.0,
            "accessibility": 1.0,
            "rent_value":    2.5,
            "comp_gap":      0.7,
            "road_quality":  0.8,
        }
    },
    "⚔️ Beat competitors": {
        "description": "Find zones competitors haven't covered yet",
        "weights": {
            "population":    1.0,
            "demand_proxy":  1.5,
            "accessibility": 1.0,
            "rent_value":    0.8,
            "comp_gap":      2.5,
            "road_quality":  0.9,
        }
    },
    "🌱 Underserved areas": {
        "description": "Expand coverage to dense but unserved zones",
        "weights": {
            "population":    2.0,
            "demand_proxy":  0.8,
            "accessibility": 1.2,
            "rent_value":    1.5,
            "comp_gap":      2.0,
            "road_quality":  0.8,
        }
    },
    "🏢 Enterprise / large dark store": {
        "description": "Sites suitable for larger 3000+ sqft operations",
        "weights": {
            "population":    1.5,
            "demand_proxy":  2.0,
            "accessibility": 1.8,
            "rent_value":    1.0,
            "comp_gap":      0.8,
            "road_quality":  1.5,
        }
    },
}


# ─── Demand proxy from OSM ────────────────────────────────────────────────────
def compute_demand_proxy(osm_counts: dict) -> float:
    """
    Demand proxy score from OSM POI types.
    Restaurants + offices + transit = strong demand signal.
    High restaurant count = high footfall = high order potential.
    """
    c = osm_counts.get("counts", {})
    restaurants  = c.get("restaurants", 0)
    offices      = c.get("offices", 0)
    transit      = c.get("transit", 0)
    supermarkets = c.get("supermarkets", 0)
    schools      = c.get("schools", 0)
    atms         = c.get("atms", 0)

    # Weighted demand signals
    raw = (
        restaurants  * 2.5  +   # restaurants = high footfall demand
        offices      * 2.0  +   # offices = lunch + evening demand
        transit      * 1.5  +   # transit hubs = density
        supermarkets * 1.2  +   # existing shops = consumer zone
        schools      * 0.8  +   # schools = family demand
        atms         * 1.0      # ATMs = economic activity
    )
    # Normalize: 500 = excellent, 0 = empty zone
    return normalize_score(raw, 0, 500)


def compute_population_score(pop_data: dict) -> float:
    """Score population density 0-100."""
    pop = pop_data.get("population", 0)
    # In Indian urban context: 100k+ in 3km radius = excellent
    return normalize_score(pop, 5000, 200000)


def compute_accessibility_score(road_data: dict, osm_counts: dict) -> float:
    """
    Accessibility = road quality + transit access.
    """
    road_score  = road_data.get("road_score", 60)
    transit     = osm_counts.get("counts", {}).get("transit", 0)
    transit_score = normalize_score(transit, 0, 30)
    return round((road_score * 0.7) + (transit_score * 0.3), 1)


def compute_rent_score(osm_counts: dict, competitor_count: int) -> float:
    """
    Rent proxy: fewer shops + more residential = lower rent.
    We invert this — lower rent = higher score.
    Commercial density = higher rent = lower score.
    """
    c = osm_counts.get("counts", {})
    commercial_density = (
        c.get("supermarkets", 0) * 3 +
        c.get("restaurants", 0) * 1 +
        c.get("offices", 0) * 2
    )
    # High commercial = high rent = lower score
    rent_pressure = normalize_score(commercial_density, 0, 300)
    return round(100 - rent_pressure, 1)


def compute_competitor_gap(competitors: list, lat: float, lon: float) -> float:
    """
    Competitor gap score: how far is nearest competitor?
    > 3km away = 90+ score (open market)
    < 0.5km = 10 score (saturated)
    """
    if not competitors:
        return 85.0  # No competitors found = good opportunity

    distances = [
        haversine_km(lat, lon, c["lat"], c["lon"])
        for c in competitors
    ]
    min_dist = min(distances)

    if min_dist >= 4.0:   return 90.0
    elif min_dist >= 3.0: return 80.0
    elif min_dist >= 2.0: return 65.0
    elif min_dist >= 1.0: return 45.0
    elif min_dist >= 0.5: return 25.0
    else:                 return 10.0


# ─── Master scoring function ──────────────────────────────────────────────────
def score_location(
    lat: float,
    lon: float,
    zone_name: str,
    scenario: str,
    osm_data: dict,
    road_data: dict,
    pop_data: dict,
    competitors: list,
    custom_weights: dict = None,
) -> dict:
    """
    Score a single location across all factors.
    Returns full breakdown with explanation.
    """
    weights = custom_weights or SCENARIOS[scenario]["weights"]

    # Compute sub-scores
    pop_score    = compute_population_score(pop_data)
    demand_score = compute_demand_proxy(osm_data)
    access_score = compute_accessibility_score(road_data, osm_data)
    rent_score   = compute_rent_score(osm_data, len(competitors))
    comp_score   = compute_competitor_gap(competitors, lat, lon)
    road_score   = road_data.get("road_score", 60)

    sub_scores = {
        "population":    pop_score,
        "demand_proxy":  demand_score,
        "accessibility": access_score,
        "rent_value":    rent_score,
        "comp_gap":      comp_score,
        "road_quality":  road_score,
    }

    # Weighted total
    total_weight = sum(weights.values())
    weighted_sum = sum(sub_scores[k] * weights[k] for k in weights)
    final_score  = round(weighted_sum / total_weight, 1)

    # Weighted contribution breakdown (what % each factor contributed)
    contributions = {
        k: round(sub_scores[k] * weights[k] / total_weight, 1)
        for k in weights
    }

    # Delivery time estimate
    delivery_time = estimate_delivery_time(access_score, road_score)

    # Coverage area
    coverage_km2 = get_isochrone_area(lat, lon, minutes=15)
    coverage_radius = round(math.sqrt(coverage_km2 / math.pi), 2)

    # ROI estimate (rough)
    monthly_orders = estimate_monthly_orders(demand_score, pop_score, coverage_km2)
    monthly_revenue = round(monthly_orders * 350, 0)  # ₹350 avg order value

    # Generate human-readable recommendation
    recommendation = generate_recommendation(
        zone_name, final_score, sub_scores, competitors
    )

    return {
        "name":            zone_name,
        "lat":             lat,
        "lon":             lon,
        "score":           final_score,
        "sub_scores":      sub_scores,
        "contributions":   contributions,
        "delivery_time":   delivery_time,
        "coverage_radius": coverage_radius,
        "coverage_km2":    coverage_km2,
        "monthly_orders":  monthly_orders,
        "monthly_revenue": monthly_revenue,
        "competitor_count": len(competitors),
        "nearest_competitor": min(
            [haversine_km(lat, lon, c["lat"], c["lon"]) for c in competitors],
            default=None
        ),
        "osm_counts":      osm_data.get("counts", {}),
        "recommendation":  recommendation,
        "data_source":     "OpenStreetMap + OSRM + OSM population estimate",
    }


def estimate_delivery_time(access_score: float, road_score: float) -> int:
    """Estimate average delivery time in minutes."""
    base = 28
    access_bonus = (access_score / 100) * 8
    road_bonus   = (road_score / 100) * 5
    return max(8, round(base - access_bonus - road_bonus))


def estimate_monthly_orders(demand_score: float, pop_score: float, coverage_km2: float) -> int:
    """Rough monthly order estimate from a dark store at this location."""
    base_orders = 3000
    demand_mult = 0.5 + (demand_score / 100)
    pop_mult    = 0.6 + (pop_score / 100) * 0.8
    coverage_mult = min(2.0, coverage_km2 / 15)
    return round(base_orders * demand_mult * pop_mult * coverage_mult)


def generate_recommendation(
    name: str, score: float, sub_scores: dict, competitors: list
) -> str:
    """Generate a plain-English recommendation for this location."""
    lines = []

    if score >= 80:
        lines.append(f"✅ **{name} is a strong placement candidate.**")
    elif score >= 65:
        lines.append(f"🟡 **{name} is a viable option** with some trade-offs.")
    else:
        lines.append(f"🔴 **{name} is not recommended** at this time.")

    # Highlight strengths
    strengths = [k for k, v in sub_scores.items() if v >= 75]
    if strengths:
        labels = {
            "population": "high population density",
            "demand_proxy": "strong demand signals",
            "accessibility": "excellent road access",
            "rent_value": "affordable rent zone",
            "comp_gap": "low competitor presence",
            "road_quality": "good road infrastructure",
        }
        s_text = ", ".join(labels.get(s, s) for s in strengths[:3])
        lines.append(f"Strengths: {s_text}.")

    # Flag weaknesses
    weaknesses = [k for k, v in sub_scores.items() if v < 45]
    if weaknesses:
        w_labels = {
            "population": "low population density",
            "demand_proxy": "weak demand signals",
            "accessibility": "poor road access",
            "rent_value": "high real estate costs",
            "comp_gap": "heavy competitor presence",
            "road_quality": "limited road network",
        }
        w_text = ", ".join(w_labels.get(w, w) for w in weaknesses[:2])
        lines.append(f"Watch out for: {w_text}.")

    if competitors:
        lines.append(f"{len(competitors)} competitor node(s) detected within 5km.")

    return " ".join(lines)


# ─── Batch scorer for multiple zones ─────────────────────────────────────────
def rank_locations(scored_zones: list, scenario: str) -> list:
    """Sort and rank a list of scored zones."""
    return sorted(scored_zones, key=lambda x: x["score"], reverse=True)


def compute_city_summary(scored_zones: list) -> dict:
    """Summary statistics for a city scan."""
    if not scored_zones:
        return {}
    scores = [z["score"] for z in scored_zones]
    return {
        "best_zone":    scored_zones[0]["name"],
        "best_score":   scored_zones[0]["score"],
        "avg_score":    round(np.mean(scores), 1),
        "zones_above_70": sum(1 for s in scores if s >= 70),
        "zones_total":  len(scores),
        "total_monthly_orders": sum(z.get("monthly_orders", 0) for z in scored_zones[:3]),
    }
