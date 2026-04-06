"""
Google Trends API Wrapper
Note: For MVP, uses mock data. Replace with real pytrends integration.
"""

from typing import Dict, List
import random

class GoogleTrendsBangalore:
    """
    Analyzes search trends for D2C categories in Bangalore localities
    """
    
    def __init__(self):
        # For MVP: Simulated data based on real Bangalore patterns
        # In production: Use pytrends library with geo='IN-KA' and city filters
        self.baseline_scores = {
            'Indiranagar': {'fashion': 95, 'beauty': 88, 'food': 75, 'home': 70},
            'Koramangala': {'fashion': 92, 'beauty': 90, 'food': 85, 'home': 65},
            'Whitefield': {'fashion': 75, 'beauty': 70, 'food': 80, 'home': 85},
            'Jayanagar': {'fashion': 85, 'beauty': 82, 'food': 90, 'home': 78},
            'HSR Layout': {'fashion': 88, 'beauty': 85, 'food': 82, 'home': 72},
            'Marathahalli': {'fashion': 70, 'beauty': 65, 'food': 75, 'home': 80},
            'Malleshwaram': {'fashion': 80, 'beauty': 75, 'food': 88, 'home': 75},
            'Electronic City': {'fashion': 65, 'beauty': 60, 'food': 70, 'home': 90},
            'Yelahanka': {'fashion': 60, 'beauty': 55, 'food': 65, 'home': 70},
            'JP Nagar': {'fashion': 82, 'beauty': 78, 'food': 85, 'home': 68}
        }
    
    def get_local_interest(self, area: str, category: str) -> float:
        """
        Returns 0-1 score for category interest in area
        """
        area_key = None
        for key in self.baseline_scores.keys():
            if key in area:
                area_key = key
                break
        
        if area_key:
            score = self.baseline_scores[area_key].get(category, 50)
            # Add some noise for realism
            noise = random.uniform(-5, 5)
            return min(100, max(0, score + noise)) / 100.0
        else:
            # Default for unknown areas (emerging localities)
            return 0.65
    
    def get_trending_categories(self, area: str) -> List[Dict]:
        """What categories are trending in this area"""
        interests = []
        for cat in ['fashion', 'beauty', 'food', 'home', 'electronics']:
            score = self.get_local_interest(area, cat)
            interests.append({
                'category': cat,
                'interest_score': score,
                'trend': 'up' if score > 0.8 else 'stable' if score > 0.6 else 'down'
            })
        return sorted(interests, key=lambda x: x['interest_score'], reverse=True)
    
    def compare_localities(self, area1: str, area2: str, category: str) -> Dict:
        """Compare demand between two Bangalore localities"""
        s1 = self.get_local_interest(area1, category)
        s2 = self.get_local_interest(area2, category)
        
        return {
            'area1': {'name': area1, 'score': s1},
            'area2': {'name': area2, 'score': s2},
            'winner': area1 if s1 > s2 else area2,
            'difference': abs(s1 - s2)
        }

# Usage in d2c_engine:
# trends = GoogleTrendsBangalore()
# demand
