"""
Bangalore Micro-Market Data
Top 50 pincodes for D2C expansion analysis
Coordinates approximated for privacy
"""

BANGALORE_MARKETS = [
    # Tier 1: Premium (High affluence, high rent)
    {"pincode": "560038", "area_name": "Indiranagar 100ft Road", "lat": 12.9784, "lon": 77.6408, "tier": "premium"},
    {"pincode": "560034", "area_name": "Koramangala 80ft Road", "lat": 12.9352, "lon": 77.6245, "tier": "premium"},
    {"pincode": "560001", "area_name": "MG Road/Brigade", "lat": 12.9716, "lon": 77.5946, "tier": "premium"},
    {"pincode": "560042", "area_name": "Commercial Street", "lat": 12.9815, "lon": 77.6087, "tier": "premium"},
    
    # Tier 2: Established (Good affluence, moderate rent)
    {"pincode": "560011", "area_name": "Jayanagar 4th Block", "lat": 12.9308, "lon": 77.5832, "tier": "established"},
    {"pincode": "560041", "area_name": "Jayanagar 9th Block", "lat": 12.9242, "lon": 77.5948, "tier": "established"},
    {"pincode": "560102", "area_name": "HSR Layout Sector 7", "lat": 12.9116, "lon": 77.6389, "tier": "established"},
    {"pincode": "560068", "area_name": "HSR Layout Sector 1", "lat": 12.9180, "lon": 77.6480, "tier": "established"},
    {"pincode": "560095", "area_name": "Koramangala 5th Block", "lat": 12.9330, "lon": 77.6150, "tier": "established"},
    {"pincode": "560003", "area_name": "Malleshwaram 8th Cross", "lat": 13.0034, "lon": 77.5660, "tier": "established"},
    {"pincode": "560004", "area_name": "Malleshwaram 15th Cross", "lat": 13.0100, "lon": 77.5700, "tier": "established"},
    {"pincode": "560560", "area_name": "JP Nagar 3rd Phase", "lat": 12.9100, "lon": 77.6000, "tier": "established"},
    {"pincode": "560078", "area_name": "JP Nagar 6th Phase", "lat": 12.9050, "lon": 77.5850, "tier": "established"},
    {"pincode": "560029", "area_name": "BTM Layout 2nd Stage", "lat": 12.9166, "lon": 77.6101, "tier": "established"},
    {"pincode": "560076", "area_name": "Bannerghatta Road", "lat": 12.8750, "lon": 77.5950, "tier": "established"},
    
    # Tier 3: Emerging (Tech hubs, lower rent, growing)
    {"pincode": "560066", "area_name": "Whitefield ITPL", "lat": 12.9698, "lon": 77.7499, "tier": "emerging"},
    {"pincode": "560067", "area_name": "Whitefield Marathahalli", "lat": 12.9591, "lon": 77.6971, "tier": "emerging"},
    {"pincode": "560103", "area_name": "Electronic City Phase 1", "lat": 12.8399, "lon": 77.6770, "tier": "emerging"},
    {"pincode": "560100", "area_name": "Electronic City Phase 2", "lat": 12.8450, "lon": 77.6700, "tier": "emerging"},
    {"pincode": "560037", "area_name": "Bellandur", "lat": 12.9261, "lon": 77.6785, "tier": "emerging"},
    {"pincode": "560035", "area_name": "Sarjapur Road", "lat": 12.9000, "lon": 77.6900, "tier": "emerging"},
    {"pincode": "560048", "area_name": "Kadubeesanahalli", "lat": 12.9500, "lon": 77.7000, "tier": "emerging"},
    {"pincode": "560045", "area_name": "Mahadevapura", "lat": 12.9900, "lon": 77.6900, "tier": "emerging"},
    {"pincode": "560016", "area_name": "Murugeshpalya", "lat": 12.9600, "lon": 77.6500, "tier": "emerging"},
    {"pincode": "560017", "area_name": "Old Airport Road", "lat": 12.9600, "lon": 77.6400, "tier": "emerging"},
    {"pincode": "560008", "area_name": "Ulsoor", "lat": 12.9800, "lon": 77.6300, "tier": "emerging"},
    
    # Tier 4: Suburban (Future growth, low rent now)
    {"pincode": "560064", "area_name": "Yelahanka New Town", "lat": 13.1007, "lon": 77.5963, "tier": "suburban"},
    {"pincode": "560063", "area_name": "Jakkur", "lat": 13.0800, "lon": 77.6100, "tier": "suburban"},
    {"pincode": "560077", "area_name": "Hebbal", "lat": 13.0353, "lon": 77.5970, "tier": "suburban"},
    {"pincode": "560024", "area_name": "RT Nagar", "lat": 13.0200, "lon": 77.5900, "tier": "suburban"},
    {"pincode": "560032", "area_name": "Vidyaranyapura", "lat": 13.0800, "lon": 77.5600, "tier": "suburban"},
    {"pincode": "560097", "area_name": "Thanisandra", "lat": 13.0600, "lon": 77.6400, "tier": "suburban"},
    {"pincode": "560049", "area_name": "Banashankari 3rd Stage", "lat": 12.9250, "lon": 77.5600, "tier": "suburban"},
    {"pincode": "560050", "area_name": "Uttarahalli", "lat": 12.9000, "lon": 77.5400, "tier": "suburban"},
    {"pincode": "560091", "area_name": "Kengeri", "lat": 12.9000, "lon": 77.4800, "tier": "suburban"},
    {"pincode": "560060", "area_name": "Nagarbhavi", "lat": 12.9600, "lon": 77.5200, "tier": "suburban"},
    
    # Tier 5: Micro-markets (Niche, specific audiences)
    {"pincode": "560025", "area_name": "Richmond Town", "lat": 12.9650, "lon": 77.6050, "tier": "niche"},
    {"pincode": "560005", "area_name": "Fraser Town", "lat": 12.9950, "lon": 77.6100, "tier": "niche"},
    {"pincode": "560047", "area_name": "Cox Town", "lat": 12.9900, "lon": 77.6200, "tier": "niche"},
    {"pincode": "560084", "area_name": "CV Raman Nagar", "lat": 12.9850, "lon": 77.6600, "tier": "niche"},
    {"pincode": "560043", "area_name": "Ramamurthy Nagar", "lat": 13.0100, "lon": 77.6700, "tier": "niche"},
    {"pincode": "560036", "area_name": "Kalyan Nagar", "lat": 13.0300, "lon": 77.6400, "tier": "niche"},
    {"pincode": "560094", "area_name": "Kammanahalli", "lat": 13.0150, "lon": 77.6350, "tier": "niche"},
    {"pincode": "560033", "area_name": "HRBR Layout", "lat": 13.0250, "lon": 77.6300, "tier": "niche"},
    {"pincode": "560022", "area_name": "Domlur", "lat": 12.9550, "lon": 77.6350, "tier": "niche"},
    {"pincode": "560071", "area_name": "Basavanagudi", "lat": 12.9450, "lon": 77.5750, "tier": "niche"}
]
