"""
Season awareness and sowing-time advice.

Maps a month to the Indian cropping season, flags which recommended crops
are actually in season, and — given the month the farmer intends to start
farming — tells them whether it is the right time to sow the recommended
crop and, if not, when to start.

Kharif  : Jun-Oct (monsoon)
Rabi    : Nov-Mar (winter)
Zaid    : Apr-May (summer)
Perennial crops are always fine.
"""

from datetime import date
from crops_meta import CROPS, sow_months, month_name


def season_of_month(m):
    """Indian cropping season for a given month number (1-12)."""
    if 6 <= m <= 10:
        return "Kharif"
    if m in (11, 12, 1, 2, 3):
        return "Rabi"
    return "Zaid"          # Apr, May


def current_season(today=None):
    return season_of_month((today or date.today()).month)


def is_in_season(crop, season):
    s = CROPS.get(crop, {}).get("season", "Perennial")
    return s == "Perennial" or s == season


def season_note(crop, season):
    s = CROPS.get(crop, {}).get("season", "Perennial")
    if s == "Perennial":
        return "Perennial — can be grown year-round."
    if s == season:
        return f"In season now ({season})."
    return f"Best grown in {s}, not the current {season} season."


# ---------------- sowing-time advice ----------------

def sow_window_str(crop):
    """Human-readable ideal sowing window, e.g. 'June–July'."""
    ms = sow_months(crop)
    if not ms:
        return "year-round"
    if len(ms) == 1:
        return month_name(ms[0])
    return f"{month_name(ms[0])}–{month_name(ms[-1])}"


def is_right_time(crop, month):
    """True if `month` falls within the crop's ideal sowing window
    (perennials are always fine)."""
    ms = sow_months(crop)
    return (not ms) or (month in ms)


def timing_verdict(crop, month):
    """Whether `month` is a good time to sow `crop`, with a message and the
    ideal window. Returned to the dashboard as the 'right time?' badge."""
    ms = sow_months(crop)
    window = sow_window_str(crop)
    name = crop.title()
    if not ms:
        return {"ok": True, "window": "year-round",
                "message": f"{name} is perennial — it can be established year-round."}
    if month in ms:
        return {"ok": True, "window": window,
                "message": f"✔ Right time to sow {crop} — "
                           f"{month_name(month)} is within the ideal {window} window."}
    return {"ok": False, "window": window,
            "message": f"⚠ {name} is best sown in {window}; sowing in "
                       f"{month_name(month)} may reduce yield. Consider starting in {window}."}


def in_season_crops(ranked, month):
    """From a model-ranked [(crop, prob), ...] list, keep those suited to the
    season of `month`. Falls back to the full list if none match."""
    season = season_of_month(month)
    kept = [(c, p) for c, p in ranked if is_in_season(c, season)]
    return kept if kept else ranked
