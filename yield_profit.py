"""
Yield + profitability ranking (real yields + real MSP prices).

For each *suitable* candidate crop (the model's confident matches) we take the
REAL median yield (quintal/ha, Govt of India APY statistics) and the REAL price
(MSP where mandated, else an indicative market price), subtract the fertilizer
cost, and rank by net profit. Crops the model is not actually suggesting
(confidence below a floor) are excluded so a low-confidence high-value crop can
never "win".
"""
from economics import yield_q_ha, price_q
from crops_meta import emoji

MIN_CONF = 0.05   # ignore crops the model gives < 5% confidence


def rank_by_profit(candidates, fert_cost_by_crop=None):
    """
    candidates: list of (crop, confidence) tuples (model's ranked matches)
    fert_cost_by_crop: optional {crop: ₹ fertilizer cost} to subtract
    Returns rows sorted by net profit desc, each flagged with data sources.
    """
    fert_cost_by_crop = fert_cost_by_crop or {}
    rows = []
    for crop, conf in candidates:
        if conf < MIN_CONF:
            continue
        y, y_src = yield_q_ha(crop)
        price, p_src = price_q(crop)
        gross = y * price
        cost = fert_cost_by_crop.get(crop, 0)
        rows.append({
            "crop": crop,
            "emoji": emoji(crop),
            "confidence": round(conf, 4),
            "yield_q_ha": y,
            "price_q": price,
            "gross": round(gross),
            "fert_cost": cost,
            "net_profit": round(gross - cost),
            "yield_source": y_src,
            "price_source": p_src,
        })
    rows.sort(key=lambda r: r["net_profit"], reverse=True)
    return rows
