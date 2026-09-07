"""
Central crop knowledge base used by the optimization modules
(season filtering, irrigation advice, yield/profit ranking, rotation).

Values are approximate, India-oriented, and meant to be easy to adjust.
They are illustrative agronomic/market figures for a final-year prototype,
not live market data. Prices are indicative ₹ per quintal; yields are
indicative quintal per hectare (perennial fruit yields are per-year).
"""

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# season: one of Kharif (monsoon, Jun-Oct), Rabi (winter, Oct-Mar),
#         Zaid (summer, Mar-Jun), Perennial (year-round)
# family: cereal | pulse | fruit | fibre | beverage  (pulses fix nitrogen)
# water:  low | medium | high
# price:  ₹/quintal (indicative)   yield: quintal/ha (indicative)
CROPS = {
    "rice":        {"season": "Kharif",    "family": "cereal",   "water": "high",   "price": 2200,  "yield": 40},
    "maize":       {"season": "Kharif",    "family": "cereal",   "water": "medium", "price": 2000,  "yield": 55},
    "chickpea":    {"season": "Rabi",      "family": "pulse",    "water": "low",    "price": 5400,  "yield": 15},
    "kidneybeans": {"season": "Rabi",      "family": "pulse",    "water": "medium", "price": 6000,  "yield": 12},
    "pigeonpeas":  {"season": "Kharif",    "family": "pulse",    "water": "low",    "price": 6600,  "yield": 12},
    "mothbeans":   {"season": "Kharif",    "family": "pulse",    "water": "low",    "price": 5500,  "yield": 8},
    "mungbean":    {"season": "Kharif",    "family": "pulse",    "water": "low",    "price": 7800,  "yield": 10},
    "blackgram":   {"season": "Kharif",    "family": "pulse",    "water": "low",    "price": 6800,  "yield": 10},
    "lentil":      {"season": "Rabi",      "family": "pulse",    "water": "low",    "price": 6000,  "yield": 12},
    "pomegranate": {"season": "Perennial", "family": "fruit",    "water": "medium", "price": 8000,  "yield": 100},
    "banana":      {"season": "Perennial", "family": "fruit",    "water": "high",   "price": 2000,  "yield": 350},
    "mango":       {"season": "Perennial", "family": "fruit",    "water": "medium", "price": 4000,  "yield": 100},
    "grapes":      {"season": "Perennial", "family": "fruit",    "water": "medium", "price": 5000,  "yield": 220},
    "watermelon":  {"season": "Zaid",      "family": "fruit",    "water": "medium", "price": 1000,  "yield": 250},
    "muskmelon":   {"season": "Zaid",      "family": "fruit",    "water": "medium", "price": 1500,  "yield": 200},
    "apple":       {"season": "Perennial", "family": "fruit",    "water": "medium", "price": 8000,  "yield": 120},
    "orange":      {"season": "Perennial", "family": "fruit",    "water": "medium", "price": 3000,  "yield": 150},
    "papaya":      {"season": "Perennial", "family": "fruit",    "water": "high",   "price": 2000,  "yield": 450},
    "coconut":     {"season": "Perennial", "family": "fruit",    "water": "high",   "price": 3000,  "yield": 100},
    "cotton":      {"season": "Kharif",    "family": "fibre",    "water": "medium", "price": 7000,  "yield": 20},
    "jute":        {"season": "Kharif",    "family": "fibre",    "water": "high",   "price": 5000,  "yield": 25},
    "coffee":      {"season": "Perennial", "family": "beverage", "water": "medium", "price": 10000, "yield": 10},
}

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🫑", "coconut": "🥥", "cotton": "🌸",
    "jute": "🧵", "coffee": "☕",
}

# approximate irrigation depth per season for each water class (mm)
WATER_MM = {"low": 350, "medium": 650, "high": 1100}


def emoji(crop):
    return CROP_EMOJI.get(crop, "🌱")


def meta(crop):
    return CROPS.get(crop, {})
