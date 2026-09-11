"""
Weather-by-location as SEASONAL CLIMATE NORMALS, not a live snapshot.

Why this matters: the crop model was trained on each crop's growing-season
climate — temperature and humidity averages and *monthly* rainfall in mm
(dataset rainfall ranges ~20-300 mm). A single live reading ("Delhi, 3pm,
38 C, 20% humidity") is out-of-distribution and produces wrong crops. So for
the month the farmer intends to start farming, we return that month's
multi-year normals:

    temperature  = mean daily-mean temperature for that month (deg C)
    humidity     = mean relative humidity for that month (%)
    rainfall     = mean TOTAL monthly precipitation for that month (mm)

Primary source: Open-Meteo Archive API (free, no API key). We geocode the
city, pull ~10 years of daily history, and aggregate to the chosen month.
Fallback: a bundled monthly-normals table (data/climate_normals.csv) so the
app still works offline / during a demo.
"""

import os
import csv
import datetime
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
YEARS = 10  # how many past years of history to average

BASE = os.path.dirname(os.path.abspath(__file__))
NORMALS_CSV = os.path.join(BASE, "data", "climate_normals.csv")

_cache = {}           # (city_lower, month) -> result dict
_normals_table = None  # lazily loaded offline table


def has_key():
    """Kept for API compatibility. Open-Meteo needs no key, so the app can
    always fetch weather; True signals the frontend that auto-weather works."""
    return True


# ---------------- offline fallback table ----------------
def _load_normals():
    global _normals_table
    if _normals_table is not None:
        return _normals_table
    table = defaultdict(dict)  # region -> {month: (t, h, r)}
    try:
        with open(NORMALS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                table[row["region"].strip().lower()][int(row["month"])] = (
                    float(row["temperature"]), float(row["humidity"]), float(row["rainfall"]))
    except Exception:
        pass
    _normals_table = table
    return table


def _offline_normals(city, month):
    table = _load_normals()
    if not table:
        return None
    key = None
    cl = (city or "").strip().lower()
    for region in table:
        if region != "default" and (region in cl or cl in region):
            key = region
            break
    if key is None:
        key = "default" if "default" in table else None
    if key is None or month not in table[key]:
        return None
    t, h, r = table[key][month]
    label = "Default (India)" if key == "default" else key.title()
    return {"temperature": t, "humidity": h, "rainfall": r,
            "location": label, "month": month,
            "source": "Offline climate table"}


# ---------------- online climate normals ----------------
_geo_cache = {}


def _geocode(city):
    key = (city or "").strip().lower()
    if key in _geo_cache:
        return _geo_cache[key]
    if requests is None:
        return None
    try:
        r = requests.get(GEO_URL, params={"name": city, "count": 5,
                                          "language": "en", "format": "json"}, timeout=10)
        if r.status_code != 200:
            return None
        results = r.json().get("results")
        if not results:
            return None
        # prefer an Indian match (many city names are ambiguous, e.g. Kochi/Japan)
        g = next((x for x in results if x.get("country_code") == "IN"), results[0])
        name = g.get("name", city)
        admin = g.get("admin1")
        out = (g["latitude"], g["longitude"], (f"{name}, {admin}" if admin else name), admin)
        _geo_cache[key] = out
        return out
    except Exception:
        return None


def get_state(city):
    """Return the state/region (admin1) for a city, or None. Cached."""
    geo = _geocode(city)
    return geo[3] if geo else None


def annual_climate(lat, lon):
    """Annual climate normals at a coordinate (for the real-data model):
    mean temperature (°C), mean humidity (%), total annual rainfall (mm)."""
    if requests is None:
        return None
    end = datetime.date.today().year - 1
    params = {"latitude": lat, "longitude": lon,
              "start_date": f"{end-4}-01-01", "end_date": f"{end}-12-31",
              "daily": "temperature_2m_mean,precipitation_sum",
              "hourly": "relative_humidity_2m", "timezone": "auto"}
    try:
        d = requests.get(ARCHIVE_URL, params=params, timeout=30).json()
        dd = d.get("daily", {})
        temps = [t for t in dd.get("temperature_2m_mean", []) if t is not None]
        precs = [p for p in dd.get("precipitation_sum", []) if p is not None]
        rhs = [h for h in d.get("hourly", {}).get("relative_humidity_2m", []) if h is not None]
        if not temps:
            return None
        return {"clim_temp": round(sum(temps) / len(temps), 1),
                "clim_rainfall": round(sum(precs) / 5),          # 5-yr avg annual mm
                "clim_humidity": round(sum(rhs) / len(rhs), 1) if rhs else 60.0}
    except Exception:
        return None


def _online_normals(city, month):
    if requests is None:
        return {"error": "requests_not_installed"}
    geo = _geocode(city)
    if not geo:
        return {"error": "geocode_failed",
                "message": f"Could not locate '{city}'. Check the spelling."}
    lat, lon, name, _admin = geo

    end_year = datetime.date.today().year - 1
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{end_year - (YEARS - 1)}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "hourly": "relative_humidity_2m",
        "timezone": "auto",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    except Exception as e:
        return {"error": "network", "message": str(e)}
    if r.status_code != 200:
        return {"error": f"http_{r.status_code}"}
    d = r.json()

    # daily temperature + per-year monthly rainfall totals
    dd = d.get("daily", {})
    dates = dd.get("time", []) or []
    temps = dd.get("temperature_2m_mean", []) or []
    precs = dd.get("precipitation_sum", []) or []
    temp_vals = []
    year_rain = defaultdict(float)
    year_seen = set()
    for dt, t, pr in zip(dates, temps, precs):
        if int(dt[5:7]) != month:
            continue
        yr = dt[:4]
        if t is not None:
            temp_vals.append(t)
        if pr is not None:
            year_rain[yr] += pr
            year_seen.add(yr)

    # hourly relative humidity for the month
    hh = d.get("hourly", {})
    htimes = hh.get("time", []) or []
    rhs = hh.get("relative_humidity_2m", []) or []
    rh_vals = [rh for ht, rh in zip(htimes, rhs) if rh is not None and int(ht[5:7]) == month]

    monthly_totals = [year_rain[y] for y in year_seen]
    if not temp_vals or not monthly_totals:
        return {"error": "insufficient_data"}

    temperature = round(sum(temp_vals) / len(temp_vals), 1)
    rainfall = round(sum(monthly_totals) / len(monthly_totals), 1)
    humidity = round(sum(rh_vals) / len(rh_vals), 1) if rh_vals else None
    if humidity is None:
        fb = _offline_normals(city, month)
        humidity = fb["humidity"] if fb else 60.0

    return {"temperature": temperature, "humidity": humidity, "rainfall": rainfall,
            "location": name, "month": month,
            "source": f"Open-Meteo climate normals ({end_year - (YEARS - 1)}–{end_year})"}


def get_weather(city, month=None):
    """Return climate normals for `city` in `month` (1-12; default = current
    month) as {temperature, humidity, rainfall, location, month, source}.
    Tries Open-Meteo, then the offline table, then an error dict the UI can
    use to fall back to manual sliders."""
    month = int(month) if month else datetime.date.today().month
    if not 1 <= month <= 12:
        month = datetime.date.today().month
    key = ((city or "").strip().lower(), month)
    if key in _cache:
        return _cache[key]

    result = _online_normals(city, month)
    if "error" in result:
        offline = _offline_normals(city, month)
        if offline:
            result = offline
    if "error" not in result:
        _cache[key] = result
    return result
