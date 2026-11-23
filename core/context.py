"""Optional context: self-reported fatigue, duration, weather (Open-Meteo, network optional)."""
import json, urllib.request, urllib.parse

def fetch_weather(lat, lon, timeout=5):
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, current="temperature_2m,relative_humidity_2m,wind_speed_10m"))
    try:
        with urllib.request.urlopen("https://api.open-meteo.com/v1/forecast?" + q, timeout=timeout) as r: cur = json.load(r)["current"]
    except Exception: return None          # offline or API change: the app keeps working
    return dict(source="open-meteo.com", **cur)

def session_context(fatigue=None, duration_min=None, weather=None):
    if fatigue is not None and not 1 <= fatigue <= 10: raise ValueError("fatigue must be 1-10")
    return dict(fatigue=fatigue, duration_min=duration_min, weather=weather, provenance="fatigue/duration self-reported; weather from API if available")

def notes(ctx):
    """General, non-medical suggestions. Context is NOT used to infer causes of performance."""
    n = []
    if ctx.get("fatigue") and ctx["fatigue"] >= 8: n.append("High self-reported fatigue: consider a lighter session.")
    t = (ctx.get("weather") or {}).get("temperature_2m")
    if t is not None and t >= 32: n.append(f"Hot conditions ({t} C): plan hydration and rest breaks.")
    return n
