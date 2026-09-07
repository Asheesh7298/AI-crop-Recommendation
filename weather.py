"""
Weather-by-location - fetches live temperature, humidity and recent
rainfall from the OpenWeather API so the farmer only has to enter soil
values. Falls back gracefully to manual entry when there is no API key
or no internet.

Set the key in an environment variable (never hard-code it):
    setx OPENWEATHER_API_KEY "your_key"     (Windows, new terminal after)
Get a free key at https://openweathermap.org/api
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # read OPENWEATHER_API_KEY from a local .env file
except ImportError:
    pass

try:
    import requests
except ImportError:  # requests may not be installed in every env
    requests = None

API_KEY_ENV = "OPENWEATHER_API_KEY"


def has_key():
    return bool(os.environ.get(API_KEY_ENV))


def get_weather(city):
    """
    Return dict {temperature, humidity, rainfall, source} for a city.
    rainfall is the last-1h/3h reading scaled to a rough seasonal mm proxy;
    if the API is unavailable, returns {'error': ...} so the UI can fall
    back to manual sliders.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        return {"error": "no_api_key"}
    if requests is None:
        return {"error": "requests_not_installed"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": key, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 401:
            return {"error": "http_401",
                    "message": "Key rejected (401). A new OpenWeather key can take 1-2 hours "
                               "to activate — check the key in .env, then try again later."}
        if r.status_code == 404:
            return {"error": "http_404", "message": f"City '{city}' not found — check the spelling."}
        if r.status_code != 200:
            return {"error": f"http_{r.status_code}", "message": f"Weather service returned {r.status_code}."}
        d = r.json()
        temp = d["main"]["temp"]
        humidity = d["main"]["humidity"]
        # OpenWeather gives short-window rain; scale to a rough monthly proxy
        rain_1h = d.get("rain", {}).get("1h") or d.get("rain", {}).get("3h") or 0
        rainfall = round(rain_1h * 30 * 8, 1) if rain_1h else 100.0
        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "rainfall": rainfall,
            "source": "OpenWeather (live)",
        }
    except Exception as e:  # network error, bad city, etc.
        return {"error": str(e)}
