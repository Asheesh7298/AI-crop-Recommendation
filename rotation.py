"""
Crop-rotation suggestion.

Good rotation preserves soil fertility: follow a nutrient-hungry cereal /
cash crop with a nitrogen-fixing pulse, and vice-versa. Logic grounded in
the fertility papers and the FYM/PSB green-gram study (pulses + PSB restore
soil N/P).
"""

from crops_meta import CROPS, emoji

# nitrogen-fixing pulses that restore soil
_PULSES = [c for c, m in CROPS.items() if m["family"] == "pulse"]
_CEREALS = [c for c, m in CROPS.items() if m["family"] == "cereal"]


def suggest(current_crop):
    fam = CROPS.get(current_crop, {}).get("family", "")

    if fam in ("cereal", "fibre", "beverage"):
        picks = _PULSES[:3]
        reason = ("This crop draws heavily on soil nitrogen. Follow it with a "
                  "nitrogen-fixing pulse to naturally restore fertility.")
    elif fam == "pulse":
        picks = _CEREALS[:2] + ["cotton"]
        reason = ("Pulses enrich the soil with nitrogen. Follow with a cereal / "
                  "cash crop that benefits from the added nitrogen.")
    else:  # fruit / perennial
        return {
            "reason": "Perennial/fruit crop — occupies land for years, so classic "
                      "annual rotation does not apply. Intercrop with legumes when young.",
            "picks": [{"crop": c, "emoji": emoji(c)} for c in _PULSES[:2]],
        }

    return {
        "reason": reason,
        "picks": [{"crop": c, "emoji": emoji(c)} for c in picks],
    }
