"""
Stage 3 - Fertilizer advice (real ICAR recommended doses, soil-test adjusted).

Uses each crop's standard recommended dose of N-P2O5-K2O (kg/ha) from ICAR /
State Agricultural University packages of practices, adjusts it by the soil-test
fertility class (Low -> apply more, High -> apply less, per soil-test-based
recommendation), and converts the nutrients into actual fertilizer products
(urea, DAP, MOP) with an indicative cost.
"""
import fertility

# Standard recommended dose (kg/ha) as N : P2O5 : K2O  (ICAR / SAU packages).
RECO_DOSE = {
    "rice": (100, 50, 50), "maize": (120, 60, 40), "chickpea": (20, 50, 20),
    "pigeonpeas": (25, 50, 25), "mothbeans": (15, 40, 20), "mungbean": (20, 40, 20),
    "blackgram": (20, 40, 20), "lentil": (20, 40, 20), "banana": (200, 60, 200),
    "coconut": (50, 30, 120), "cotton": (100, 50, 50), "jute": (60, 30, 30),
    "pomegranate": (50, 25, 50), "mango": (100, 50, 100), "grapes": (100, 50, 100),
    "watermelon": (100, 50, 50), "muskmelon": (100, 50, 50), "apple": (70, 35, 70),
    "orange": (60, 30, 60), "papaya": (200, 100, 200), "coffee": (40, 30, 40),
    "kidneybeans": (20, 40, 20),
}
_DEFAULT_DOSE = (80, 40, 40)

# soil-test class -> dose multiplier (apply more when Low, less when High)
_ADJ = {"Low": 1.25, "Medium": 1.0, "High": 0.75}

# fertilizer nutrient contents and indicative subsidised price (₹/kg)
UREA_N, DAP_N, DAP_P, MOP_K = 0.46, 0.18, 0.46, 0.60
PRICE = {"urea": 5.9, "dap": 27.0, "mop": 34.0}


def _adjusted_dose(crop, n, p, k):
    base = RECO_DOSE.get(crop, _DEFAULT_DOSE)
    rep = fertility.assess(n, p, k, 7.0)   # pH not needed for N/P/K class
    classes = (rep["Nitrogen (N)"], rep["Phosphorus (P)"], rep["Potassium (K)"])
    return tuple(round(d * _ADJ.get(cls, 1.0)) for d, cls in zip(base, classes)), classes


def _products(N, P2O5, K2O):
    """Convert an N-P2O5-K2O dose (kg/ha) into urea/DAP/MOP quantities (kg/ha)."""
    dap = P2O5 / DAP_P if P2O5 > 0 else 0.0          # DAP supplies the phosphorus
    n_from_dap = dap * DAP_N
    urea = max(0.0, (N - n_from_dap)) / UREA_N       # urea tops up remaining N
    mop = K2O / MOP_K if K2O > 0 else 0.0            # MOP supplies the potassium
    return {"urea": round(urea), "dap": round(dap), "mop": round(mop)}


def advise(crop, n, p, k):
    """Return real, soil-adjusted fertilizer advice for the recommended crop."""
    (N, P2O5, K2O), classes = _adjusted_dose(crop, n, p, k)
    prod = _products(N, P2O5, K2O)
    tips = [
        f"Recommended dose for {crop} (soil-test adjusted): "
        f"N {N} · P₂O₅ {P2O5} · K₂O {K2O} kg/ha "
        f"(soil rated N-{classes[0]}, P-{classes[1]}, K-{classes[2]}).",
        f"Apply about {prod['urea']} kg Urea + {prod['dap']} kg DAP + {prod['mop']} kg MOP per hectare.",
        "Split N: half at sowing/planting, the rest as top-dressing. "
        "Add well-rotted FYM (~10 t/ha) to build soil health (ICAR package of practices).",
    ]
    return tips


def estimated_cost(crop, n, p, k):
    """Indicative ₹/ha cost of the recommended fertilizer (real subsidised rates)."""
    (N, P2O5, K2O), _ = _adjusted_dose(crop, n, p, k)
    prod = _products(N, P2O5, K2O)
    cost = (prod["urea"] * PRICE["urea"] + prod["dap"] * PRICE["dap"]
            + prod["mop"] * PRICE["mop"])
    return round(cost)
