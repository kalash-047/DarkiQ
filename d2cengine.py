"""
D2C Expansion Engine - Bangalore MVP
Optimized for Bangalore metro + Tier-2 expansion (Mysore, Coimbatore, Hubli)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
import json

from engine import geocode, get_isochrone, haversine, fetch_poi_counts_radius
from ml_engine import cluster_outlets_dbscan, partition_territories_kmeans

@dataclass
class BangaloreMicroMarket:
    pincode: str
    area_name: str
    lat: float
    lon: float
    expansion_score: float
    format: str  # 'flagship', 'community_store', 'pop_up', 'retail_partner'
    monthly_potential: float
    setup_cost: float
    break_even_months: int
    competitor_count: int
    affluence_index: float  # 0-1
    digital_demand: float   # Search volume proxy
    rent_per_sqft: float    # ₹/sqft
    cannibalization_risk: float
    catchment_population: int
    top_nearby_localities: List[str]

class BangaloreExpansionEngine:
    """
    Specialized for Bangalore D2C market:
    - Analyzes 50 key micro-markets (pincode level)
    - Compares Indiranagar vs Koramangala vs Whitefield vs Jayanagar
    - Predicts cannibalization from online (Bangalore has high digital penetration)
    """
    
    def __init__(self, brand_category: str, avg_order_value: int = 1500):
        self.category = brand_category
        self.aov = avg_order_value
        self.city_center = [12.9716, 77.5946]  # Bangalore
        
        # Bangalore-specific benchmarks
        self.format_costs = {
            'flagship': {'rent': 180, 'setup': 12000, 'sqft': 1200},      # ₹180/sqft (Indiranagar 100ft)
            'community_store': {'rent': 95, 'setup': 8000, 'sqft': 600},  # ₹95/sqft (Koramangala 5th block)
            'pop_up': {'rent': 60, 'setup': 3000, 'sqft': 400},           # ₹60/sqft (Whitefield)
            'retail_partner': {'rent': 0, 'setup': 200000, 'sqft': 0}     # One-time partnership cost
        }
        
        # Load Bangalore data
        from data.bangalore_pincodes import BANGALORE_MARKETS
        self.markets = BANGALORE_MARKETS
    
    def analyze_full_bangalore(self) -> Dict:
        """Analyze all 50 Bangalore micro-markets"""
        results = []
        
        for market_data in self.markets:
            analysis = self._analyze_micro_market(market_data)
            if analysis:
                results.append(analysis)
        
        # Sort by expansion score
        results.sort(key=lambda x: x.expansion_score, reverse=True)
        
        return {
            'city': 'Bangalore',
            'analysis_date': datetime.now().isoformat(),
            'total_markets_analyzed': len(results),
            'top_recommendations': results[:10],
            'investment_summary': self._calculate_investment_summary(results[:5]),
            'risk_heatmap': self._generate_risk_heatmap(results),
            'format_distribution': self._analyze_format_mix(results[:10])
        }
    
    def analyze_specific_corridor(self, corridor: str) -> Dict:
        """
        Analyze specific Bangalore corridors:
        'indiranagar-100ft', 'koramangala-80ft', 'whitefield-itpl', 
        'jayanagar-4thblock', 'hsr-sector7'
        """
        corridor_pins = {
            'indiranagar-100ft': ['560038', '560008'],
            'koramangala-80ft': ['560034', '560095'],
            'whitefield-itpl': ['560066', '560067'],
            'jayanagar-4thblock': ['560011', '560041'],
            'hsr-sector7': ['560102', '560068']
        }
        
        pins = corridor_pins.get(corridor, [])
        markets = [m for m in self.markets if m['pincode'] in pins]
        
        results = [self._analyze_micro_market(m) for m in markets]
        results = [r for r in results if r]
        
        return {
            'corridor': corridor,
            'markets': results,
            'corridor_score': np.mean([r.expansion_score for r in results]) if results else 0,
            'recommendation': 'Strong entry' if results and results[0].expansion_score > 75 else 'Test with pop-up'
        }
    
    def _analyze_micro_market(self, market: Dict) -> Optional[BangaloreMicroMarket]:
        """Deep analysis of single pincode"""
        pincode = market['pincode']
        lat = market['lat']
        lon = market['lon']
        area = market['area_name']
        
        # 1. Affluence scoring (Bangalore-specific weights)
        affluence = self._calculate_bangalore_affluence(lat, lon, area)
        
        # 2. Competition density (existing D2C stores in 2km)
        competitors = self._count_bangalore_competitors(lat, lon)
        
        # 3. Real estate efficiency (vs corridor average)
        rent_efficiency = self._get_bangalore_rent(area, pincode)
        
        # 4. Digital demand (Google Trends proxy)
        digital_demand = self._get_digital_demand_bangalore(area)
        
        # 5. Cannibalization (distance from existing online customer base)
        cannibalization = self._estimate_bangalore_cannibalization(lat, lon, pincode)
        
        # 6. Catchment quality (isochrone analysis)
        catchment = self._analyze_catchment(lat, lon)
        
        # Calculate scores
        scores = {
            'affluence': affluence * 25,
            'demand': digital_demand * 20,
            'competition_gap': max(0, 15 - competitors * 3),
            'real_estate': rent_efficiency * 15,
            'cannibalization_safety': (1 - cannibalization) * 10,
            'catchment': min(15, catchment['population'] / 5000)
        }
        
        total_score = sum(scores.values())
        
        # Recommend format based on score and affluence
        fmt = self._recommend_bangalore_format(total_score, affluence, competitors, rent_efficiency)
        cost = self.format_costs[fmt]
        
        # Financials
        monthly_orders = self._estimate_monthly_orders(affluence, digital_demand, competitors)
        monthly_revenue = monthly_orders * self.aov
        setup = cost['setup'] * cost['sqft'] if cost['sqft'] > 0 else cost['setup']
        break_even = int(setup / (monthly_revenue * 0.22)) if monthly_revenue > 0 else 99  # 22% net margin assumed
        
        return BangaloreMicroMarket(
            pincode=pincode,
            area_name=area,
            lat=lat,
            lon=lon,
            expansion_score=round(total_score, 1),
            format=fmt,
            monthly_potential=monthly_revenue,
            setup_cost=setup,
            break_even_months=break_even,
            competitor_count=competitors,
            affluence_index=round(affluence, 2),
            digital_demand=round(digital_demand, 2),
            rent_per_sqft=cost['rent'],
            cannibalization_risk=round(cannibalization, 2),
            catchment_population=catchment['population'],
            top_nearby_localities=catchment['nearby']
        )
    
    def _calculate_bangalore_affluence(self, lat: float, lon: float, area: str) -> float:
        """
        Bangalore-specific affluence signals:
        - Proximity to tech parks (Manyata, Ecoworld, ITPL)
        - Metro connectivity (purple/green line)
        - HNI residential pockets (Sadashivnagar, Indiranagar, Lavelle)
        """
        # Tech park proximity (high weight)
        tech_hubs = [
            (12.9965, 77.6618, 'Manyata'),      # Manyata Tech Park
            (12.9218, 77.6856, 'Ecoworld'),     # Ecoworld
            (12.9698, 77.7499, 'ITPL'),         # Whitefield ITPL
            (12.9346, 77.6120, 'Uber')          # Uber/JP Nagar
        ]
        
        min_dist_tech = min([haversine(lat, lon, t[0], t[1]) for t in tech_hubs])
        tech_score = max(0, 1 - (min_dist_tech / 10))  # Decay over 10km
        
        # Metro connectivity (Bangalore specific)
        metro_stations = [
            (12.9916, 77.5968, 'MG Road'),
            (12.9738, 77.6140, 'Indiranagar'),
            (12.9279, 77.6271, 'Jayanagar'),
            (12.9352, 77.6245, 'Koramangala')
        ]
        min_dist_metro = min([haversine(lat, lon, m[0], m[1]) for m in metro_stations])
        metro_score = max(0, 1 - (min_dist_metro / 3))  # 3km catchment
        
        # Premium area tags
        premium_areas = ['Indiranagar', 'Koramangala', 'Whitefield', 'Jayanagar', 'HSR Layout']
        area_premium = any(p in area for p in premium_areas)
        
        affluence = (tech_score * 0.5) + (metro_score * 0.3) + (0.2 if area_premium else 0)
        return min(affluence, 1.0)
    
    def _count_bangalore_competitors(self, lat: float, lon: float) -> int:
        """Count similar category stores in 2km radius using OSM"""
        try:
            poi = fetch_poi_counts_radius(lat, lon, radius_m=2000)
            
            if self.category == 'fashion':
                return poi['counts'].get('clothes', 0) + poi['counts'].get('boutique', 0)
            elif self.category == 'beauty':
                return poi['counts'].get('cosmetics', 0) + poi['counts'].get('chemist', 0)
            elif self.category == 'food':
                return poi['counts'].get('supermarket', 0) + poi['counts'].get('grocery', 0)
            else:
                return poi['counts'].get('retail_outlets', 0)
        except:
            return 0
    
    def _get_bangalore_rent(self, area: str, pincode: str) -> float:
        """
        Rent efficiency score (0-1)
        1.0 = market rate, >1.0 = premium/inefficient, <1.0 = value
        """
        # Bangalore rent ranges (₹/sqft/month)
        rent_tiers = {
            'Indiranagar': (160, 200),
            'Koramangala': (90, 130),
            'Whitefield': (50, 80),
            'Jayanagar': (80, 110),
            'HSR Layout': (70, 100),
            'Malleshwaram': (60, 90),
            'JP Nagar': (65, 95),
            'BTM Layout': (55, 85)
        }
        
        base_rent = 75  # City average
        for tier, (low, high) in rent_tiers.items():
            if tier in area:
                base_rent = (low + high) / 2
                break
        
        # Efficiency: Lower rent = higher score (up to a point)
        # Optimal is 80-100 range (established but not overheated)
        if 80 <= base_rent <= 100:
            return 1.0
        elif base_rent < 80:
            return 0.9  # Emerging area, slight risk
        else:
            return max(0.5, 1 - ((base_rent - 100) / 200))
    
    def _get_digital_demand_bangalore(self, area: str) -> float:
        """
        Proxy for digital demand in Bangalore area
        In production: Google Trends API for "fashion near me", etc.
        """
        # Bangalore digital penetration is high (90%+)
        # Base score by area sophistication
        digital_baselines = {
            'Indiranagar': 0.95,
            'Koramangala': 0.92,
            'Whitefield': 0.88,
            'Jayanagar': 0.90,
            'HSR Layout': 0.89,
            'Marathahalli': 0.85,
            'Malleshwaram': 0.87
        }
        
        for key, val in digital_baselines.items():
            if key in area:
                return val
        
        return 0.80  # Default for other areas
    
    def _estimate_bangalore_cannibalization(self, lat: float, lon: float, pincode: str) -> float:
        """
        Bangalore has high online penetration
        Risk: Your online customers switching to store (channel conflict)
        """
        # Distance from city center (online is stronger in center)
        dist_center = haversine(lat, lon, self.city_center[0], self.city_center[1])
        
        # Closer to center = higher online penetration = higher cannibalization risk
        if dist_center < 5:  # Central Bangalore
            base_risk = 0.6
        elif dist_center < 12:  # Middle ring
            base_risk = 0.4
        else:  # Outer (Whitefield, Electronic City)
            base_risk = 0.25
        
        # Category modifier (Fashion higher cannibalization than Food)
        category_mult = {
            'fashion': 1.2,
            'beauty': 1.1,
            'electronics': 1.3,
            'food': 0.8,
            'home': 0.9
        }
        
        return min(0.9, base_risk * category_mult.get(self.category, 1.0))
    
    def _analyze_catchment(self, lat: float, lon: float) -> Dict:
        """15-minute isochrone analysis"""
        iso = get_isochrone(lat, lon, minutes=15, costing="motor_scooter")
        
        # Estimate population (Bangalore density ~10k/sqkm in residential)
        area_km2 = iso.get('area_km2', 3.0)
        population = int(area_km2 * 8000)  # Conservative
        
        # Nearby localities (simplified - would use reverse geocoding)
        return {
            'population': population,
            'area_km2': area_km2,
            'nearby': ['Adjacent localities within 2km']  # Simplified
        }
    
    def _recommend_bangalore_format(self, score: float, affluence: float, 
                                   competitors: int, rent_eff: float) -> str:
        """Bangalore-specific format logic"""
        if score < 45:
            return 'retail_partner'
        elif affluence > 0.8 and competitors < 3:
            return 'flagship'
        elif competitors > 6 or rent_eff < 0.7:
            return 'pop_up'
        else:
            return 'community_store'
    
    def _estimate_monthly_orders(self, affluence: float, demand: float, competitors: int) -> int:
        """Projected monthly transactions"""
        base = 400  # Bangalore baseline for D2C
        affluence_mult = 1 + (affluence - 0.5)  # +/- 50%
        demand_mult = demand
        competition_penalty = max(0.3, 1 - (competitors * 0.1))
        
        return int(base * affluence_mult * demand_mult * competition_penalty)
    
    def _calculate_investment_summary(self, top_markets: List[BangaloreMicroMarket]) -> Dict:
        """ROI calculation for top 5 locations"""
        total_setup = sum([m.setup_cost for m in top_markets])
        annual_revenue = sum([m.monthly_potential * 12 for m in top_markets])
        avg_break_even = np.mean([m.break_even_months for m in top_markets])
        
        return {
            'total_setup_cost_lakhs': round(total_setup / 100000, 1),
            'projected_annual_revenue_cr': round(annual_revenue / 10000000, 2),
            'average_break_even_months': int(avg_break_even),
            'roi_24_months': round(((annual_revenue * 2 - total_setup) / total_setup) * 100, 0) if total_setup > 0 else 0,
            'recommended_sequence': [f"{m.area_name} ({m.format})" for m in top_markets[:3]]
        }
    
    def _generate_risk_heatmap(self, all_markets: List[BangaloreMicroMarket]) -> List[Dict]:
        """Risk factors for Bangalore market"""
        high_cannibalization = [m for m in all_markets if m.cannibalization_risk > 0.6]
        high_competition = [m for m in all_markets if m.competitor_count > 5]
        
        risks = []
        if high_cannibalization:
            risks.append({
                'type': 'Channel Cannibalization',
                'count': len(high_cannibalization),
                'areas': [m.area_name for m in high_cannibalization[:3]],
                'mitigation': 'Start with retail partners in high-digital areas'
            })
        
        return risks
    
    def _analyze_format_mix(self, markets: List[BangaloreMicroMarket]) -> Dict:
        """Optimal store format distribution"""
        formats = {}
        for m in markets:
            formats[m.format] = formats.get(m.format, 0) + 1
        
        return {
            'recommended_mix': formats,
            'investment_breakdown': {
                fmt: sum([m.setup_cost for m in markets if m.format == fmt])
                for fmt in formats.keys()
            }
        }
