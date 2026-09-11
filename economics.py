"""
Real crop economics — replaces the earlier fabricated price/yield figures.

Yields are the REAL median yields (quintal/ha) computed from the Government of
India district-wise Area-Production-Yield statistics (1997-2020, last decade
used), in data/real/crop_yield.json.

Prices are the REAL Minimum Support Prices (MSP) announced by the Government of
India (CACP) for the 2024-25 season, in ₹/quintal, for the mandated crops.
Crops without an MSP (fruits, coconut) are marked as indicative market prices.
"""
import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))
_YIELD_PATH = os.path.join(BASE, "data", "real", "crop_yield.json")

try:
    REAL_YIELD = json.load(open(_YIELD_PATH, encoding="utf-8"))   # {crop: q/ha}
except Exception:
    REAL_YIELD = {}

# Real MSP 2024-25 (₹/quintal), Government of India / CACP.
MSP = {
    "rice": 2300, "maize": 2225, "chickpea": 5650, "pigeonpeas": 7550,
    "mungbean": 8682, "blackgram": 7400, "lentil": 6700, "mothbeans": 5550,
    "cotton": 7121, "jute": 5335,
}
# Indicative market prices (₹/quintal) for non-MSP crops (fruits/plantation) —
# clearly not MSP; averaged wholesale, for illustration only.
MARKET = {
    "banana": 1800, "coconut": 3000, "pomegranate": 6000, "mango": 4000,
    "grapes": 5000, "apple": 7000, "orange": 3000, "papaya": 1800,
    "watermelon": 1000, "muskmelon": 1500, "coffee": 9000, "kidneybeans": 6000,
}
# fallback indicative yields (q/ha) for crops not in the govt APY dataset
FALLBACK_YIELD = {
    "pomegranate": 100, "mango": 90, "grapes": 210, "apple": 110, "orange": 140,
    "papaya": 400, "watermelon": 240, "muskmelon": 190, "coffee": 9,
    "kidneybeans": 12, "coconut": 90,
}


def yield_q_ha(crop):
    """Real median yield (q/ha) if available, else an indicative fallback."""
    if crop in REAL_YIELD:
        return REAL_YIELD[crop], "real (Govt of India APY)"
    if crop in FALLBACK_YIELD:
        return FALLBACK_YIELD[crop], "indicative"
    return 0, "n/a"


def price_q(crop):
    """Price ₹/quintal — real MSP where mandated, else indicative market price."""
    if crop in MSP:
        return MSP[crop], "MSP 2024-25"
    if crop in MARKET:
        return MARKET[crop], "market (indicative)"
    return 0, "n/a"
