"""
Stage 1 - Soil fertility assessment.

Classifies each soil parameter into Low / Medium / High using the
rating tables from the reference papers (Sheeba et al. 2022 potassium/pH
tables; Suchithra & Pai 2020 six-tier fertility indices). This mirrors
what the papers do BEFORE we go on to recommend a crop.
"""

# (low_upper, high_lower): value < low_upper -> Low; > high_lower -> High; else Medium
_THRESHOLDS = {
    "N":   (50, 90),     # available nitrogen (dataset units)
    "P":   (30, 60),     # phosphorus
    "K":   (30, 60),     # potassium
    "ph":  (5.5, 7.5),   # handled specially below (acidic / neutral / alkaline)
}


def _level(value, low_upper, high_lower):
    if value < low_upper:
        return "Low"
    if value > high_lower:
        return "High"
    return "Medium"


def ph_class(ph):
    if ph < 5.5:
        return "Acidic"
    if ph > 7.5:
        return "Alkaline"
    return "Neutral"


def assess(n, p, k, ph):
    """Return a dict of soil-fertility labels for the dashboard."""
    report = {
        "Nitrogen (N)": _level(n, *_THRESHOLDS["N"]),
        "Phosphorus (P)": _level(p, *_THRESHOLDS["P"]),
        "Potassium (K)": _level(k, *_THRESHOLDS["K"]),
        "Soil pH": ph_class(ph),
    }
    return report


def overall(report):
    """A one-line summary of fertility."""
    highs = sum(1 for v in report.values() if v == "High")
    lows = sum(1 for v in report.values() if v == "Low")
    if lows >= 2:
        return "Poor — several nutrients deficient"
    if highs >= 2:
        return "Rich — good nutrient status"
    return "Moderate — balanced fertility"
