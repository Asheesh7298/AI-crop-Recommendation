"""
Season awareness - maps the current month to the Indian cropping season
and flags which recommended crops are actually in season.

Kharif  : Jun-Oct (monsoon)
Rabi    : Oct-Mar (winter)
Zaid    : Mar-Jun (summer)
Perennial crops are always fine.
"""

from datetime import date
from crops_meta import CROPS


def current_season(today=None):
    m = (today or date.today()).month
    if 6 <= m <= 9:
        return "Kharif"
    if m == 10:
        return "Kharif"        # tail of Kharif / start of Rabi
    if m in (11, 12, 1, 2, 3):
        return "Rabi"
    return "Zaid"              # Apr, May


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
