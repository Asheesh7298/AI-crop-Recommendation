"""
IoT device simulator — mimics an ESP32 field sensor rig.

It generates realistic, slowly-drifting soil + climate readings and POSTs
them to the app's /api/ingest endpoint every few seconds, exactly like a
real ESP32 would. The server stores each reading (so the dashboard's live
mode shows it) and returns the crop recommendation, which we print.

Use it to demo the full IoT data path with NO hardware.

Examples
--------
Local server:
    python iot/iot_simulator.py

Deployed app:
    python iot/iot_simulator.py --url https://your-app.onrender.com

Options:
    --url       API base URL (default http://localhost:5000)
    --interval  seconds between readings (default 5)
    --device    device id (default esp32-01)
    --token     X-Device-Token, only if the server sets DEVICE_TOKEN
    --profile   which crop's conditions to imitate (default rice)
"""

import argparse
import random
import time

import requests

# rough per-crop sensor centres (N,P,K,temp,humidity,pH,rainfall) — same
# ballpark as the training data, so recommendations come out sensible.
PROFILES = {
    "rice":   (80, 48, 40, 24, 82, 6.4, 236),
    "maize":  (78, 48, 20, 22, 65, 6.2, 84),
    "cotton": (118, 46, 20, 24, 80, 6.9, 80),
    "banana": (100, 82, 50, 27, 80, 6.0, 105),
    "coffee": (101, 29, 30, 26, 59, 6.8, 158),
}
KEYS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
JITTER = [6, 6, 6, 1.5, 4, 0.2, 15]  # per-reading noise


def make_reading(centre, moisture):
    r = {}
    for k, c, j in zip(KEYS, centre, JITTER):
        v = max(0, c + random.uniform(-j, j))
        r[k] = round(v, 1)
    r["ph"] = round(min(9.5, max(3.5, r["ph"])), 1)
    r["soil_moisture"] = round(moisture, 0)
    return r


def main():
    ap = argparse.ArgumentParser(description="ESP32 sensor-rig simulator")
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--device", default="esp32-01")
    ap.add_argument("--token", default=None)
    ap.add_argument("--profile", default="rice", choices=list(PROFILES))
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/api/ingest"
    centre = PROFILES[args.profile]
    moisture = 45.0
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["X-Device-Token"] = args.token

    print(f"[{args.device}] streaming '{args.profile}' readings to {endpoint}")
    print(f"[{args.device}] every {args.interval}s — Ctrl+C to stop\n")

    while True:
        # soil moisture drifts and slowly dries out, then a 'rain' event refills
        moisture = max(8, moisture - random.uniform(0, 3))
        if random.random() < 0.15:
            moisture = min(95, moisture + random.uniform(20, 40))

        reading = make_reading(centre, moisture)
        payload = {**reading, "device_id": args.device}
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                best = resp.json().get("result", {}).get("best", {})
                print(f"  sent N={reading['N']:.0f} P={reading['P']:.0f} K={reading['K']:.0f} "
                      f"pH={reading['ph']:.1f} moist={reading['soil_moisture']:.0f}%  "
                      f"->  {best.get('emoji','')} {best.get('crop','?')} "
                      f"({best.get('confidence',0)*100:.0f}%)")
            else:
                print(f"  server returned HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"  send failed: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
