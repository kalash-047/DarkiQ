"""
DistrictIQ ML Engine
Institutional-grade spatial analytics:
  - DBSCAN: cluster outlet density → identify natural territory boundaries
  - K-Means: optimal distributor territory partitioning
  - Voronoi: territory boundary generation from distributor locations
  - Gravity model: cannibalization and overlap prediction
  - Coverage gap scoring: demand × (1 - supply) formulation
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Voronoi
from scipy.optimize import linear_sum_assignment

# ── DBSCAN Outlet Clustering ──────────────────────────────────────────────────
def cluster_outlets_dbscan(
    outlets: List[Dict],
    eps_km: float = 0.4,
    min_samples: int = 5
) -> Dict:
    """
    Cluster outlet locations using DBSCAN.
    eps_km: maximum distance between outlets in same cluster
    min_samples: minimum outlets to form a cluster
    Returns cluster labels and cluster centroids.
    """
    if len(outlets) < min_samples:
        return {"clusters": [], "labels": [], "noise_count": 0}

    # Convert to radians for haversine metric
    coords = np.radians([[o["lat"], o["lon"]] for o in outlets])

    # eps in radians (1 km = 0.000157 radians on earth surface)
    eps_rad = eps_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree",
                metric="haversine")
    labels = db.fit_predict(coords)

    # Build cluster summaries
    clusters = []
    unique_labels = set(labels) - {-1}

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_outlets = [o for o, m in zip(outlets, mask) if m]
        lats = [o["lat"] for o in cluster_outlets]
        lons = [o["lon"] for o in cluster_outlets]

        # Estimate demand weight from outlet type
        demand_weights = {
            "supermarket": 3.0, "grocery": 2.0, "convenience": 1.5,
            "kirana": 1.5, "general": 1.2, "provisions": 1.2,
        }
        weighted_demand = sum(
            demand_weights.get(o.get("type",""), 1.0) for o in cluster_outlets
        )

        clusters.append({
            "id":           label,
            "centroid_lat": round(np.mean(lats), 6),
            "centroid_lon": round(np.mean(lons), 6),
            "outlet_count": len(cluster_outlets),
            "weighted_demand": round(weighted_demand, 1),
            "lat_spread":   round(np.std(lats) * 111.32, 2),  # km
            "lon_spread":   round(np.std(lons) * 111.32, 2),  # km
            "outlets":      cluster_outlets,
        })

    # Sort by weighted demand
    clusters.sort(key=lambda x: x["weighted_demand"], reverse=True)

    return {
        "clusters":    clusters,
        "labels":      labels.tolist(),
        "noise_count": int(np.sum(labels == -1)),
        "n_clusters":  len(clusters),
        "coverage_pct":round(100 * (1 - np.mean(labels == -1)), 1),
    }

# ── K-Means Territory Partitioning ───────────────────────────────────────────
def partition_territories_kmeans(
    outlets: List[Dict],
    n_distributors: int,
    existing_hubs: Optional[List[Dict]] = None,
    demand_weighted: bool = True
) -> Dict:
    """
    Partition outlet universe into N balanced territories using K-Means.
    Returns optimal territory assignments and centroids (= ideal hub locations).
    demand_weighted: if True, weight by outlet type importance.
    """
    if len(outlets) < n_distributors:
        return {"territories": [], "error": "Fewer outlets than distributors"}

    demand_weights = {
        "supermarket": 3.0, "grocery": 2.0, "convenience": 1.5,
        "kirana": 1.5, "general": 1.2, "provisions": 1.2,
    }

    coords = np.array([[o["lat"], o["lon"]] for o in outlets])
    weights = None

    if demand_weighted:
        weights = np.array([
            demand_weights.get(o.get("type",""), 1.0) for o in outlets
        ])

    # Use existing hub locations as initial centroids if provided
    init = "k-means++"
    if existing_hubs and len(existing_hubs) >= n_distributors:
        init = np.array([[h["lat"], h["lon"]] for h in existing_hubs[:n_distributors]])

    km = KMeans(n_clusters=n_distributors, init=init, n_init=10, random_state=42)

    if demand_weighted and weights is not None:
        # Weighted K-means: replicate points by weight
        int_weights = np.round(weights * 2).astype(int)
        expanded = np.repeat(coords, int_weights, axis=0)
        km.fit(expanded)
        # Predict on original coords
        labels = km.predict(coords)
    else:
        km.fit(coords)
        labels = km.labels_

    # Build territories
    territories = []
    for t in range(n_distributors):
        mask     = labels == t
        t_outlets = [o for o, m in zip(outlets, mask) if m]
        if not t_outlets:
            continue

        lats = [o["lat"] for o in t_outlets]
        lons = [o["lon"] for o in t_outlets]
        t_weights = [demand_weights.get(o.get("type",""), 1.0) for o in t_outlets]

        # Weighted centroid = optimal hub location
        total_w   = sum(t_weights)
        hub_lat   = sum(la * w for la, w in zip(lats, t_weights)) / total_w
        hub_lon   = sum(lo * w for lo, w in zip(lons, t_weights)) / total_w

        # Coverage area estimate
        lat_range = (max(lats) - min(lats)) * 111.32
        lon_range = (max(lons) - min(lons)) * 111.32 * math.cos(math.radians(np.mean(lats)))
        area_est  = lat_range * lon_range

        # Max travel distance from hub to furthest outlet
        max_dist  = max(
            _haversine(hub_lat, hub_lon, o["lat"], o["lon"]) for o in t_outlets
        )

        territories.append({
            "id":            t,
            "hub_lat":       round(hub_lat, 6),
            "hub_lon":       round(hub_lon, 6),
            "outlet_count":  len(t_outlets),
            "total_demand":  round(total_w, 1),
            "outlets":       t_outlets,
            "area_km2":      round(area_est, 2),
            "max_reach_km":  round(max_dist, 2),
            "avg_dist_km":   round(np.mean([
                _haversine(hub_lat, hub_lon, o["lat"], o["lon"])
                for o in t_outlets
            ]), 2),
        })

    # Balance score: std of territory sizes (lower = more balanced)
    outlet_counts = [t["outlet_count"] for t in territories]
    balance_score = round(100 - min(100, np.std(outlet_counts) / max(1, np.mean(outlet_counts)) * 100), 1)

    return {
        "territories":   territories,
        "n_territories": len(territories),
        "balance_score": balance_score,
        "total_outlets": len(outlets),
        "inertia":       round(km.inertia_, 2),
    }

# ── Voronoi Territory Boundaries ──────────────────────────────────────────────
def compute_voronoi_territories(hubs: List[Dict], bbox: Tuple) -> List[Dict]:
    """
    Compute Voronoi tessellation from hub locations.
    Each cell = territory of nearest hub.
    bbox = (min_lat, min_lon, max_lat, max_lon)
    Returns polygon boundaries for each territory.
    """
    if len(hubs) < 2:
        return []

    points = np.array([[h["lat"], h["lon"]] for h in hubs])

    # Add mirror points outside bbox to ensure all regions are bounded
    min_lat, min_lon, max_lat, max_lon = bbox
    padding = max(max_lat - min_lat, max_lon - min_lon) * 2
    mirrors = np.array([
        [min_lat - padding, min_lon - padding],
        [min_lat - padding, max_lon + padding],
        [max_lat + padding, min_lon - padding],
        [max_lat + padding, max_lon + padding],
    ])
    all_points = np.vstack([points, mirrors])

    try:
        vor = Voronoi(all_points)
        territories = []

        for i, hub in enumerate(hubs):
            region_idx = vor.point_region[i]
            region     = vor.regions[region_idx]

            if -1 in region or not region:
                continue

            vertices = vor.vertices[region]

            # Clip to bbox
            clipped = []
            for v in vertices:
                clipped_lat = max(min_lat, min(max_lat, v[0]))
                clipped_lon = max(min_lon, min(max_lon, v[1]))
                clipped.append([clipped_lon, clipped_lat])  # GeoJSON format [lon, lat]

            if len(clipped) >= 3:
                clipped.append(clipped[0])  # close polygon
                territories.append({
                    "hub_idx":   i,
                    "hub_name":  hub.get("name", f"Hub {i+1}"),
                    "polygon":   clipped,
                    "area_km2":  _polygon_area(clipped),
                })

        return territories
    except Exception as e:
        return []

# ── White Space Scoring Model ─────────────────────────────────────────────────
def score_white_space(
    outlet_clusters: List[Dict],
    existing_hubs: List[Dict],
    city_metadata: Dict,
    category: str = "FMCG"
) -> List[Dict]:
    """
    Score each outlet cluster for white space opportunity.
    Formula: WS = Demand × (1 - Supply) × Market_Potential × Growth
    """
    if not outlet_clusters:
        return []

    # Supply = proximity to nearest existing hub (0-1, 1=fully covered)
    def supply_score(cluster_lat, cluster_lon):
        if not existing_hubs:
            return 0.0
        dists = [
            _haversine(cluster_lat, cluster_lon, h["lat"], h["lon"])
            for h in existing_hubs
        ]
        min_dist = min(dists)
        # 0-2km = fully covered (1.0), 2-5km = partial, >5km = uncovered (0.0)
        if min_dist <= 1.5: return 1.0
        elif min_dist <= 5.0: return max(0, 1 - (min_dist - 1.5) / 3.5)
        return 0.0

    cat_multipliers = {
        "Personal Care":     1.15,
        "Food & Beverages":  1.25,
        "Home Care":         1.05,
        "Health & Wellness": 1.20,
        "Dairy & Staples":   1.30,
        "FMCG":              1.10,
    }
    cat_mult = cat_multipliers.get(category, 1.0)

    scored = []
    for cluster in outlet_clusters:
        lat = cluster["centroid_lat"]
        lon = cluster["centroid_lon"]

        supply   = supply_score(lat, lon)
        gap      = 1 - supply

        # Demand proxy from cluster characteristics
        demand   = min(100, cluster["weighted_demand"] * 4)

        # WS score
        ws_score = round(demand * gap * cat_mult, 1)

        # Revenue estimate
        monthly_rev = round(cluster["outlet_count"] * gap *
                            city_metadata.get("avg_revenue_per_outlet", 12000))

        scored.append({
            **cluster,
            "white_space_score": ws_score,
            "demand_score":      round(demand, 1),
            "supply_score":      round(supply * 100, 1),
            "coverage_gap_pct":  round(gap * 100, 1),
            "monthly_rev_opp":   monthly_rev,
            "annual_rev_opp":    monthly_rev * 12,
            "priority": (
                "critical" if ws_score >= 60 else
                "high"     if ws_score >= 40 else
                "medium"   if ws_score >= 20 else "low"
            ),
        })

    scored.sort(key=lambda x: x["white_space_score"], reverse=True)
    return scored

# ── Optimal Hub Count (Elbow Method) ─────────────────────────────────────────
def find_optimal_hub_count(outlets: List[Dict], max_k: int = 10) -> Dict:
    """
    Use elbow method on K-Means inertia to find optimal number of distributors.
    """
    if len(outlets) < 3:
        return {"optimal_k": 1, "inertias": []}

    coords  = np.array([[o["lat"], o["lon"]] for o in outlets])
    inertias = []
    k_range  = range(1, min(max_k + 1, len(outlets)))

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(coords)
        inertias.append(km.inertia_)

    # Find elbow: point of maximum curvature
    if len(inertias) >= 3:
        diffs  = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow  = int(np.argmin(diffs2)) + 2  # +2 because of double diff offset
    else:
        elbow = len(inertias)

    return {
        "optimal_k":  min(elbow, max_k),
        "inertias":   inertias,
        "k_range":    list(k_range),
        "method":     "elbow",
    }

# ── Gravity Model: Cannibalization ───────────────────────────────────────────
def gravity_cannibalization(
    candidate_hub: Dict,
    existing_hubs: List[Dict],
    alpha: float = 1.0,
    beta:  float = 2.0,
) -> Dict:
    """
    Huff Gravity Model: probability that each existing hub loses customers to candidate.
    P(candidate) = A_c^α / D_c^β  /  Σ(A_i^α / D_i^β)
    alpha = attractiveness exponent, beta = distance decay
    """
    if not existing_hubs:
        return {"transfer_probability": 0, "cannibalization_risk": "NONE"}

    # Attractiveness proxy (monthly outlet count or revenue)
    A_c = math.sqrt(candidate_hub.get("outlet_count", 300))

    transfers = []
    for hub in existing_hubs:
        dist = max(0.1, _haversine(
            candidate_hub["lat"], candidate_hub["lon"],
            hub["lat"], hub["lon"]
        ))
        A_i = math.sqrt(hub.get("outlet_count", 300))

        # Utility scores
        U_c = (A_c ** alpha) / (dist ** beta)
        U_i = (A_i ** alpha) / (dist ** beta)

        # Probability candidate captures this hub's customers
        transfer_p = U_c / (U_c + U_i) if (U_c + U_i) > 0 else 0

        transfers.append({
            "hub_name":    hub.get("name", "Existing hub"),
            "distance_km": round(dist, 2),
            "transfer_probability": round(transfer_p, 3),
            "revenue_at_risk": round(
                hub.get("monthly_revenue", 500000) * transfer_p
            ),
        })

    total_transfer = sum(t["transfer_probability"] for t in transfers)
    avg_transfer   = total_transfer / len(transfers) if transfers else 0

    risk = ("HIGH" if avg_transfer > 0.3 else
            "MEDIUM" if avg_transfer > 0.15 else "LOW")

    return {
        "transfer_probability": round(avg_transfer, 3),
        "cannibalization_risk": risk,
        "affected_hubs":       transfers,
        "total_revenue_at_risk": sum(t["revenue_at_risk"] for t in transfers),
    }

# ── Utility ───────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def _polygon_area(coords):
    """Shoelace formula."""
    n = len(coords)
    if n < 3: return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        x1 = coords[i][0] * 111.32
        y1 = coords[i][1] * 111.32
        x2 = coords[j][0] * 111.32
        y2 = coords[j][1] * 111.32
        area += x1 * y2 - x2 * y1
    return round(abs(area) / 2, 2)
