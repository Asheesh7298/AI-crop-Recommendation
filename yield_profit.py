"""
Yield + profitability ranking.

For each candidate crop we take an indicative yield (quintal/ha) and market
price (₹/quintal) from crops_meta, subtract the fertilizer cost, and rank
crops by expected net profit. This turns 'most suitable' into 'most
profitable' - the optimization the user asked for.
"""

from crops_meta import CROPS, emoji


def gross_income(crop):
    m = CROPS.get(crop, {})
    return m.get("yield", 0) * m.get("price", 0)


def rank_by_profit(candidate_crops, fert_cost_by_crop=None):
    """
    candidate_crops: list of crop names (e.g. model's top-N)
    fert_cost_by_crop: optional {crop: ₹ fertilizer cost} to subtract
    Returns list of dicts sorted by net profit desc.
    """
    fert_cost_by_crop = fert_cost_by_crop or {}
    rows = []
    for crop in candidate_crops:
        m = CROPS.get(crop, {})
        income = gross_income(crop)
        cost = fert_cost_by_crop.get(crop, 0)
        rows.append({
            "crop": crop,
            "emoji": emoji(crop),
            "yield_q_ha": m.get("yield", 0),
            "price_q": m.get("price", 0),
            "gross": income,
            "fert_cost": cost,
            "net_profit": income - cost,
        })
    rows.sort(key=lambda r: r["net_profit"], reverse=True)
    return rows
