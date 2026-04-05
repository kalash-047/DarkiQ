# cannibalization_engine.py
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from typing import List, Dict, Tuple
import h3
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import requests

@dataclass
class Store:
    id: str
    lat: float
    lon: float
    monthly_revenue: float
    monthly_orders: int
    catchment_polygon: Polygon = None  # Valhalla isochrone
    customer_density_grid: Dict = None  # H3 hex -> order count

@dataclass
class CandidateLocation:
    lat: float
    lon: float
    setup_cost: float = 2500000  # ₹25L default
    operating_cost: float = 180000  # ₹1.8L/month

class NetworkCannibalizationAnalyzer:
    """
    Predicts net incremental revenue from new locations.
    The core algorithm that makes DarkIQ fundable.
    """
    
    def __init__(self, valhalla_url: str = "http://localhost:8002"):
        self.valhalla_url = valhalla_url
        self.h3_resolution = 9  # ~200m hexagons
        
    def analyze_expansion(
        self,
        existing_network: List[Store],
        candidate: CandidateLocation,
        customer_transfer_model: str = "gravity"
    ) -> Dict:
        """
        Returns the only number that matters: Net Incremental Revenue
        """
        # 1. Generate catchment area (15-min delivery zone)
        candidate_catchment = self._get_isochrone(
            candidate.lat, candidate.lon, minutes=15
        )
        
        if not candidate_catchment:
            return {"error": "Could not generate catchment area"}
        
        # 2. Find overlapping stores
        overlapping_stores = self._find_catchment_overlaps(
            candidate_catchment, existing_network
        )
        
        if not overlapping_stores:
            # Whitespace opportunity - no cannibalization risk
            whitespace_revenue = self._estimate_white_space_revenue(
                candidate, candidate_catchment
            )
            return {
                "net_incremental_revenue": whitespace_revenue,
                "cannibalization_loss": 0,
                "affected_stores": [],
                "risk_level": "LOW",
                "recommendation": "PROCEED - Pure whitespace",
                "roi_months": candidate.setup_cost / (whitespace_revenue * 0.12)
            }
        
        # 3. Calculate cannibalization for each overlapping store
        total_cannibalization = 0
        affected_stores = []
        
        for store in overlapping_stores:
            # Calculate transfer probability (Huff Gravity Model variant)
            transfer_rate = self._calculate_transfer_probability(
                store, candidate, candidate_catchment
            )
            
            revenue_at_risk = store.monthly_revenue * transfer_rate
            total_cannibalization += revenue_at_risk
            
            affected_stores.append({
                "store_id": store.id,
                "current_revenue": store.monthly_revenue,
                "transfer_probability": transfer_rate,
                "revenue_at_risk": revenue_at_risk,
                "distance_km": self._haversine(
                    store.lat, store.lon, candidate.lat, candidate.lon
                )
            })
        
        # 4. Estimate new revenue capture (from competition/untapped demand)
        whitespace_revenue = self._estimate_white_space_revenue(
            candidate, candidate_catchment, 
            competitor_overlap=len(overlapping_stores)
        )
        
        # 5. The fundable number
        net_incremental = whitespace_revenue - total_cannibalization
        
        # 6. Portfolio-level optimization check
        network_efficiency = self._calculate_network_efficiency(
            existing_network, candidate, net_incremental
        )
        
        return {
            "gross_opportunity": whitespace_revenue,
            "cannibalization_loss": total_cannibalization,
            "net_incremental_revenue": net_incremental,
            "cannibalization_ratio": total_cannibalization / whitespace_revenue if whitespace_revenue > 0 else 0,
            "affected_stores": affected_stores,
            "stores_at_risk": len(affected_stores),
            "network_efficiency_score": network_efficiency,
            "risk_level": self._classify_risk(net_incremental, total_cannibalization),
            "recommendation": self._generate_recommendation(
                net_incremental, candidate.setup_cost
            ),
            "roi_months": candidate.setup_cost / max(net_incremental * 0.12, 1),
            "breakeven_orders": candidate.setup_cost / 420,  # ₹420 AOV
            "optimal_timing": self._recommend_timing(
                existing_network, candidate
            )
        }
    
    def _get_isochrone(self, lat: float, lon: float, minutes: int = 15) -> Polygon:
        """Get actual drivable area from Valhalla."""
        try:
            resp = requests.post(
                f"{self.valhalla_url}/isochrone",
                json={
                    "locations": [{"lat": lat, "lon": lon}],
                    "costing": "motor_scooter",  # India delivery mode
                    "contours": [{"time": minutes, "color": "ff0000"}],
                    "polygons": True,
                    "generalize": 50
                },
                timeout=10
            )
            data = resp.json()
            
            # Extract polygon from GeoJSON
            features = data.get("features", [])
            if features:
                coords = features[0]["geometry"]["coordinates"][0]
                return Polygon(coords)
        except:
            pass
        
        # Fallback: Circle with 70% radius (road network efficiency factor)
        return self._circle_polygon(lat, lon, km=minutes/60 * 25 * 0.7)
    
    def _find_catchment_overlaps(
        self, 
        candidate_catchment: Polygon, 
        network: List[Store]
    ) -> List[Store]:
        """Find existing stores whose catchments overlap with candidate."""
        overlapping = []
        
        for store in network:
            if not store.catchment_polygon:
                # Generate on-the-fly if not cached
                store.catchment_polygon = self._get_isochrone(
                    store.lat, store.lon
                )
            
            if store.catchment_polygon and store.catchment_polygon.intersects(
                candidate_catchment
            ):
                # Calculate intersection area
                intersection = store.catchment_polygon.intersection(
                    candidate_catchment
                )
                overlap_ratio = intersection.area / store.catchment_polygon.area
                
                if overlap_ratio > 0.05:  # >5% overlap = cannibalization risk
                    overlapping.append(store)
        
        return overlapping
    
    def _calculate_transfer_probability(
        self,
        existing_store: Store,
        candidate: CandidateLocation,
        candidate_catchment: Polygon
    ) -> float:
        """
        Huff Gravity Model: Probability customer chooses new store over existing.
        P(j) = (S_j^α / D_j^β) / Σ(S_k^α / D_k^β)
        
        Where:
        S = Store attractiveness (assortment size, rating proxy)
        D = Drive time
        α, β = calibration parameters (default 1.0, 2.0)
        """
        # Calculate drive time between stores (approximate)
        drive_time_existing = self._estimate_drive_time(
            existing_store.lat, existing_store.lon,
            candidate.lat, candidate.lon
        )
        
        # Attractiveness proxy: Revenue = proxy for assortment/service quality
        attractiveness_existing = np.sqrt(existing_store.monthly_revenue)
        attractiveness_candidate = np.sqrt(200000)  # Assume ₹2L startup revenue
        
        # Distance decay function (inverse square law)
        utility_existing = attractiveness_existing / (drive_time_existing ** 2)
        utility_candidate = attractiveness_candidate / (drive_time_existing ** 2)
        
        # Transfer probability = likelihood of choosing new over old
        # Simplified: If utilities are equal, 50% transfer
        transfer_prob = utility_candidate / (utility_existing + utility_candidate)
        
        # Adjust for catchment overlap intensity
        overlap_area = existing_store.catchment_polygon.intersection(
            candidate_catchment
        ).area
        overlap_intensity = overlap_area / existing_store.catchment_polygon.area
        
        # Final transfer rate: Up to 40% of overlapping customers may switch
        return min(0.40, transfer_prob * overlap_intensity)
    
    def _estimate_white_space_revenue(
        self,
        candidate: CandidateLocation,
        catchment: Polygon,
        competitor_overlap: int = 0
    ) -> float:
        """
        Estimate revenue from truly new customers (not stolen from network).
        Uses H3 hexagon demand proxy.
        """
        # Convert catchment to H3 hexagons
        hexagons = self._polygon_to_h3(catchment)
        
        total_demand = 0
        
        for hex_id in hexagons:
            # Demand proxy: Population density × Income index × Commercial activity
            demand_score = self._get_hex_demand_score(hex_id)
            
            # If hex already served by competitor, reduce capture rate
            if competitor_overlap > 0:
                capture_rate = 0.15 / competitor_overlap  # Split market
            else:
                capture_rate = 0.35  # Monopoly capture in whitespace
            
            total_demand += demand_score * capture_rate
        
        # Convert to revenue (₹420 AOV, 30 days)
        estimated_orders = total_demand * 30  # Daily demand × month
        return estimated_orders * 420
    
    def _calculate_network_efficiency(
        self,
        existing_network: List[Store],
        candidate: CandidateLocation,
        net_incremental: float
    ) -> float:
        """
        Calculate if this location improves or hurts overall network efficiency.
        Metric: Revenue per sq km of total coverage area.
        """
        # Current network coverage (union of all catchments)
        existing_coverage = unary_union([
            s.catchment_polygon for s in existing_network 
            if s.catchment_polygon
        ])
        
        # Add candidate
        candidate_catchment = self._get_isochrone(candidate.lat, candidate.lon)
        new_coverage = unary_union([existing_coverage, candidate_catchment])
        
        # Calculate efficiency metrics
        current_revenue = sum(s.monthly_revenue for s in existing_network)
        new_total_revenue = current_revenue + net_incremental
        
        current_density = current_revenue / existing_coverage.area if existing_coverage.area > 0 else 0
        new_density = new_total_revenue / new_coverage.area if new_coverage.area > 0 else 0
        
        # Efficiency score: >1.0 means location improves portfolio density
        return new_density / current_density if current_density > 0 else 1.0
    
    def _classify_risk(
        self, 
        net_incremental: float, 
        cannibalization: float
    ) -> str:
        """Risk classification for investor reporting."""
        if net_incremental <= 0:
            return "CRITICAL - Value Destructive"
        elif cannibalization / max(net_incremental, 1) > 0.5:
            return "HIGH - Majority Value Transfer"
        elif cannibalization / max(net_incremental, 1) > 0.25:
            return "MEDIUM - Moderate Overlap"
        else:
            return "LOW - Pure Growth"
    
    def _generate_recommendation(
        self, 
        net_incremental: float, 
        setup_cost: float
    ) -> str:
        """Board-level recommendation."""
        roi = net_incremental * 12 / setup_cost  # Annual ROI
        
        if net_incremental <= 0:
            return "REJECT - Destroys shareholder value"
        elif roi < 0.5:  # <50% annual return
            return "REJECT - Insufficient return (ROI < 50%)"
        elif roi < 1.0:
            return "CONDITIONAL - Accept only if strategic (competitive blocking)"
        else:
            return "ACCEPT - Accretive to portfolio"
    
    def _recommend_timing(
        self,
        network: List[Store],
        candidate: CandidateLocation
    ) -> str:
        """
        Should you open now, or wait for existing stores to mature?
        """
        avg_store_age_months = np.mean([
            self._estimate_store_age(s) for s in network
        ])
        
        nearby_stores = [
            s for s in network 
            if self._haversine(s.lat, s.lon, candidate.lat, candidate.lon) < 3.0
        ]
        
        immature_nearby = [
            s for s in nearby_stores 
            if self._estimate_store_age(s) < 6  # <6 months old
        ]
        
        if immature_nearby:
            return f"DELAY 6 months - {len(immature_nearby)} nearby stores still maturing"
        else:
            return "PROCEED - Network ready for expansion"
    
    # --- Helper methods ---
    
    def _polygon_to_h3(self, polygon: Polygon, resolution: int = 9) -> List[str]:
        """Convert shapely polygon to H3 hexagon set."""
        # Get bounding box
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        
        hexagons = set()
        center = h3.latlng_to_cell((min_lat + max_lat)/2, (min_lon + max_lon)/2, resolution)
        
        # Expand from center until we cover the polygon
        for ring in range(1, 20):
            new_hexes = h3.grid_disk(center, ring)
            for h in new_hexes:
                lat, lon = h3.cell_to_latlng(h)
                point = Point(lon, lat)
                if polygon.contains(point):
                    hexagons.add(h)
            if len(hexagons) > 1000:  # Limit for performance
                break
        
        return list(hexagons)
    
    def _get_hex_demand_score(self, hex_id: str) -> float:
        """
        Fetch demand proxy for H3 hexagon.
        In production: Query PostGIS with OSM + Census data.
        """
        # Placeholder: In real implementation, query your demand database
        # Return daily order potential for this hex
        return np.random.uniform(5, 25)  # Mock 5-25 orders/day
    
    def _estimate_drive_time(
        self, 
        lat1: float, lon1: float, 
        lat2: float, lon2: float
    ) -> float:
        """Minutes driving between points (Valhalla or heuristic)."""
        dist_km = self._haversine(lat1, lon1, lat2, lon2)
        return dist_km / 0.5  # 30 km/h avg in city = 0.5 km/min
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Distance in km."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def _circle_polygon(self, lat, lon, km):
        """Fallback circular catchment."""
        angles = np.linspace(0, 2*np.pi, 32)
        lons = lon + (km/111.32) * np.cos(angles) / np.cos(np.radians(lat))
        lats = lat + (km/111.32) * np.sin(angles)
        return Polygon(zip(lons, lats))
    
    def _estimate_store_age(self, store: Store) -> int:
        """Mock: In reality, fetch from store metadata."""
        return 12  # months


class PortfolioOptimizer:
    """
    Given budget constraint, select optimal subset of candidates.
    Solves the 0-1 Knapsack problem with network effects.
    """
    
    def __init__(self, analyzer: NetworkCannibalizationAnalyzer):
        self.analyzer = analyzer
    
    def optimize_portfolio(
        self,
        existing_network: List[Store],
        candidates: List[CandidateLocation],
        budget: float,
        top_n: int = 10
    ) -> Dict:
        """
        Returns the optimal set of locations to open given capex constraint.
        Accounts for interdependencies (cannibalization between new stores).
        """
        # Calculate marginal value for each candidate independently
        candidate_metrics = []
        
        for i, cand in enumerate(candidates):
            print(f"Analyzing candidate {i+1}/{len(candidates)}...")
            metrics = self.analyzer.analyze_expansion(existing_network, cand)
            
            candidate_metrics.append({
                "candidate": cand,
                "metrics": metrics,
                "roi": metrics.get("net_incremental_revenue", 0) * 12 / cand.setup_cost,
                "efficiency": metrics.get("network_efficiency_score", 1.0)
            })
        
        # Greedy selection with network effect simulation
        selected = []
        remaining_budget = budget
        current_network = existing_network.copy()
        
        # Sort by ROI initially
        sorted_candidates = sorted(
            candidate_metrics, 
            key=lambda x: x["roi"], 
            reverse=True
        )
        
        for item in sorted_candidates:
            if remaining_budget < item["candidate"].setup_cost:
                continue
            
            # Re-analyze with current network state (includes previously selected)
            updated_metrics = self.analyzer.analyze_expansion(
                current_network, item["candidate"]
            )
            
            if updated_metrics.get("net_incremental_revenue", 0) > 0:
                selected.append({
                    "location": (item["candidate"].lat, item["candidate"].lon),
                    "setup_cost": item["candidate"].setup_cost,
                    "net_incremental": updated_metrics["net_incremental_revenue"],
                    "roi_months": updated_metrics["roi_months"],
                    "cannibalization": updated_metrics["cannibalization_loss"]
                })
                
                # Add to network for next iteration (simulation)
                new_store = Store(
                    id=f"NEW_{len(selected)}",
                    lat=item["candidate"].lat,
                    lon=item["candidate"].lon,
                    monthly_revenue=updated_metrics["net_incremental_revenue"],
                    monthly_orders=int(updated_metrics["net_incremental_revenue"] / 420),
                    catchment_polygon=self.analyzer._get_isochrone(
                        item["candidate"].lat, item["candidate"].lon
                    )
                )
                current_network.append(new_store)
                remaining_budget -= item["candidate"].setup_cost
        
        total_investment = budget - remaining_budget
        total_incremental = sum(s["net_incremental"] for s in selected)
        
        return {
            "selected_locations": selected,
            "total_capex": total_investment,
            "total_monthly_incremental": total_incremental,
            "portfolio_roi": total_incremental * 12 / total_investment if total_investment > 0 else 0,
            "avg_cannibalization_ratio": np.mean([
                s["cannibalization"] / (s["net_incremental"] + s["cannibalization"]) 
                for s in selected
            ]) if selected else 0,
            "rejected_due_to_cannibalization": len([
                c for c in candidate_metrics 
                if c["metrics"].get("net_incremental_revenue", 0) <= 0
            ])
        }


# --- Usage Example ---

if __name__ == "__main__":
    # Initialize with existing network
    analyzer = NetworkCannibalizationAnalyzer()
    
    existing_stores = [
        Store("BLR_001", 12.9352, 77.6245, 850000, 2024),
        Store("BLR_002", 12.9784, 77.6408, 1200000, 2857),
        Store("BLR_003", 12.9698, 77.7499, 650000, 1548),
    ]
    
    # Pre-compute catchments (do this once, cache in Redis)
    for store in existing_stores:
        store.catchment_polygon = analyzer._get_isochrone(store.lat, store.lon)
    
    # Analyze new candidate
    candidate = CandidateLocation(lat=12.9500, lon=77.6300, setup_cost=2500000)
    
    result = analyzer.analyze_expansion(existing_stores, candidate)
    
    print("\n" + "="*60)
    print("CANNIBALIZATION ANALYSIS")
    print("="*60)
    print(f"Gross Opportunity:     ₹{result['gross_opportunity']:,.0f}/month")
    print(f"Cannibalization Loss:  ₹{result['cannibalization_loss']:,.0f}/month")
    print(f"Net Incremental:       ₹{result['net_incremental_revenue']:,.0f}/month")
    print(f"Cannibalization Ratio: {result['cannibalization_ratio']:.1%}")
    print(f"Risk Level:            {result['risk_level']}")
    print(f"ROI Timeline:          {result['roi_months']:.1f} months")
    print(f"\nRecommendation: {result['recommendation']}")
    
    if result['affected_stores']:
        print("\nAffected Stores:")
        for aff in result['affected_stores']:
            print(f"  - {aff['store_id']}: ₹{aff['revenue_at_risk']:,.0f} at risk "
                  f"({aff['transfer_probability']:.1%} transfer rate)")
    
    # Portfolio optimization example
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION")
    print("="*60)
    
    candidates = [
        CandidateLocation(12.9500, 77.6300),
        CandidateLocation(12.9600, 77.6200),
        CandidateLocation(12.9400, 77.6500),
        CandidateLocation(12.9800, 77.6100),
    ]
    
    optimizer = PortfolioOptimizer(analyzer)
    portfolio = optimizer.optimize_portfolio(
        existing_stores, candidates, budget=10000000  # ₹1 Cr capex
    )
    
    print(f"Optimal portfolio: {len(portfolio['selected_locations'])} stores")
    print(f"Total Investment:  ₹{portfolio['total_capex']:,.0f}")
    print(f"Monthly Increment: ₹{portfolio['total_monthly_incremental']:,.0f}")
    print(f"Portfolio ROI:     {portfolio['portfolio_roi']:.1%}")
