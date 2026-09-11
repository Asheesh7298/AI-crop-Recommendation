"""
Stage 1 - Soil fertility assessment (standard soil-test rating + Parker's
Nutrient Index).

Each macronutrient is rated Low / Medium / High using the standard three-tier
soil-test rating method, and the three ratings are combined into an overall
Soil Fertility Index using **Parker's Nutrient Index** (Parker et al., 1951):

        NI = (n_low·1 + n_medium·2 + n_high·3) / n_total
        NI < 1.67  -> Low ;  1.67-2.33 -> Medium ;  > 2.33 -> High

Units note: the reference soil-test critical limits are defined for available
nutrients in kg/ha (e.g. N: Low <280, Medium 280-560, High >560; P: <10/10-25/
>25; K: <120/120-280/>280 — Indian Soil Health Card standard). The soil
sensors / dataset used here report N, P and K on a compressed operating scale
(N 0-140, P 5-145, K 5-205), so the class limits below are the standard
three-tier ratings **calibrated to that operating range** (dataset tertiles).
The rating method and the Parker Nutrient Index formula are unchanged.
"""

# (low_upper, high_lower): value < low_upper -> Low; > high_lower -> High; else Medium
# Calibrated to the sensor/dataset operating range (33rd/67th percentiles).
LIMITS = {
    "N": (30, 70),
    "P": (35, 60),
    "K": (25, 45),
}

# Reference Soil Health Card critical limits (available nutrient, kg/ha) — kept
# for documentation / the report; used when inputs are supplied in kg/ha.
LIMITS_KG_HA = {
    "N": (280, 560),
    "P": (10, 25),
    "K": (120, 280),
}

_RANK = {"Low": 1, "Medium": 2, "High": 3}


def _rate(value, low_upper, high_lower):
    if value < low_upper:
        return "Low"
    if value > high_lower:
        return "High"
    return "Medium"


def ph_class(ph):
    if ph < 6.5:
        return "Acidic"
    if ph > 7.5:
        return "Alkaline"
    return "Neutral"


def assess(n, p, k, ph):
    """Return a dict of soil-fertility labels for the dashboard."""
    return {
        "Nitrogen (N)": _rate(n, *LIMITS["N"]),
        "Phosphorus (P)": _rate(p, *LIMITS["P"]),
        "Potassium (K)": _rate(k, *LIMITS["K"]),
        "Soil pH": ph_class(ph),
    }


def nutrient_index(n, p, k):
    """Parker's Nutrient Index over the three macronutrients.
    Returns (value, class) where class is Low / Medium / High."""
    ratings = [_rate(n, *LIMITS["N"]), _rate(p, *LIMITS["P"]), _rate(k, *LIMITS["K"])]
    ni = round(sum(_RANK[r] for r in ratings) / len(ratings), 2)
    cls = "Low" if ni < 1.67 else ("Medium" if ni <= 2.33 else "High")
    return ni, cls


def overall(report):
    """One-line fertility summary driven by Parker's Nutrient Index.
    Accepts the assess() report (to read back the N/P/K ratings)."""
    ratings = [report.get("Nitrogen (N)"), report.get("Phosphorus (P)"),
               report.get("Potassium (K)")]
    ni = round(sum(_RANK.get(r, 2) for r in ratings) / len(ratings), 2)
    if ni < 1.67:
        return f"Poor — low nutrient status (Nutrient Index {ni:.2f})"
    if ni <= 2.33:
        return f"Moderate — balanced fertility (Nutrient Index {ni:.2f})"
    return f"Rich — good nutrient status (Nutrient Index {ni:.2f})"
