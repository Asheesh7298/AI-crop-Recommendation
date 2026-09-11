"""
Real-world cultivation cross-check.

Grounds every recommendation in reality: is this crop actually grown in the
farmer's state? Data is the Government of India district-wise Area-Production
statistics (last decade), pre-aggregated per state in data/real/state_crops.json
as each mapped crop's share (%) of the state's cropped area.
"""
import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(BASE, "data", "real", "state_crops.json")

try:
    STATE_CROPS = json.load(open(_PATH, encoding="utf-8"))
except Exception:
    STATE_CROPS = {}

# normalise geocoder state names -> APY state names
_ALIASES = {
    "national capital territory of delhi": "Delhi",
    "nct": "Delhi", "delhi": "Delhi",
    "odisha": "Odisha", "orissa": "Odisha",
    "uttarakhand": "Uttarakhand", "uttaranchal": "Uttarakhand",
}


def _match_state(state):
    if not state:
        return None
    s = state.strip()
    if s in STATE_CROPS:
        return s
    low = s.lower()
    if low in _ALIASES and _ALIASES[low] in STATE_CROPS:
        return _ALIASES[low]
    for name in STATE_CROPS:                       # loose contains match
        if name.lower() == low or low in name.lower() or name.lower() in low:
            return name
    return None


def check(crop, state):
    """Return a real-world grounding dict for `crop` in `state`, or None if we
    have no cultivation record for that crop/state."""
    st = _match_state(state)
    if not st:
        return None
    rec = STATE_CROPS.get(st, {})
    shares = rec.get("shares", {})
    if crop not in shares:
        return {"state": st, "grown": False,
                "message": f"Rarely grown in {st} — not in the state's major-crop records."}
    share = shares[crop]
    rank = rec.get("ranks", {}).get(crop)
    if share >= 15:
        verdict = f"Major crop in {st} — grown on ~{share}% of the state's tracked cropland (rank #{rank})."
    elif share >= 3:
        verdict = f"Commonly grown in {st} — ~{share}% of tracked cropland (rank #{rank})."
    else:
        verdict = f"Grown in {st} but only on ~{share}% of tracked cropland (rank #{rank})."
    return {"state": st, "grown": True, "share": share, "rank": rank, "message": verdict}
