"""
Stage 3 - Fertilizer advice.

Compares the sensed N/P/K against the average N/P/K the recommended crop
grows in (learned from the dataset, stored in crop_model.pkl as
'crop_means'), then suggests which nutrient to add or reduce.

Agronomy logic (which fertilizer supplies which nutrient, role of FYM /
PSB) follows Rekha et al. 2018 (green-gram FYM/P/PSB study).
"""

# nutrient -> (common straight fertilizer, note)
_SOURCES = {
    "N": ("Urea", "nitrogen for leafy/vegetative growth"),
    "P": ("DAP / Single Super Phosphate", "phosphorus for root & flowering; PSB improves uptake"),
    "K": ("Muriate of Potash (MOP)", "potassium for grain filling & disease resistance"),
}

# how far off before we flag it (dataset units)
_TOLERANCE = 12


def advise(crop, n, p, k, crop_means):
    """Return a list of advice strings comparing sensed vs required NPK."""
    target = crop_means.get(crop)
    if not target:
        return ["No nutrient profile available for this crop."]

    sensed = {"N": n, "P": p, "K": k}
    tips = []
    for nut in ("N", "P", "K"):
        need = target[nut]
        have = sensed[nut]
        diff = need - have
        src, role = _SOURCES[nut]
        if diff > _TOLERANCE:
            tips.append(f"⬆ Add {nut}: ~{diff:.0f} units short. Apply {src} ({role}).")
        elif diff < -_TOLERANCE:
            tips.append(f"⬇ {nut} is high (~{abs(diff):.0f} above need). Reduce {nut} input to save cost.")
        else:
            tips.append(f"✅ {nut} is adequate for {crop}.")

    tips.append("🌱 Add well-rotted FYM (~10 t/ha) to improve nutrient balance and soil health (Rekha et al. 2018).")
    return tips


def estimated_cost(crop, n, p, k, crop_means):
    """Very rough ₹/ha estimate of the extra fertilizer to buy (indicative)."""
    target = crop_means.get(crop)
    if not target:
        return 0
    # indicative ₹ per 'unit' shortfall per nutrient
    unit_cost = {"N": 8, "P": 12, "K": 10}
    cost = 0
    sensed = {"N": n, "P": p, "K": k}
    for nut in ("N", "P", "K"):
        diff = target[nut] - sensed[nut]
        if diff > _TOLERANCE:
            cost += diff * unit_cost[nut]
    return round(cost)
