# IoT Sensor Module

The IoT layer of **System using IoT and Crop Recommendation AI**. A field
node reads soil + climate sensors and streams them to the app, which returns
a live crop recommendation.

## Data flow

```
 ┌─────────── Field node (ESP32) ───────────┐
 │  NPK sensor (RS485/Modbus) ─┐             │
 │  Soil pH sensor  ───────────┤             │        HTTPS POST JSON
 │  DHT22 (temp + humidity) ───┼─► ESP32 ────┼──── /api/ingest ──►  Crop Recommendation AI
 │  Soil-moisture probe  ──────┘   (WiFi)    │                        (FastAPI server)
 └───────────────────────────────────────────┘                              │
                                                                             ▼
                          Dashboard "Live sensor" mode  ◄──── /api/latest ── stores reading
                          (auto-fills + re-runs the recommendation)          + returns advice
```

- **Device → server:** `POST /api/ingest` with the 8 readings + `device_id`.
  The server stores it and returns the full recommendation to the device.
- **Server → dashboard:** the web page's *Live sensor feed* polls `GET /api/latest`,
  auto-fills the sliders from the newest reading, and re-runs the recommendation.

## Try it now — no hardware needed

Run the app, then run the simulator (it behaves exactly like the ESP32):

```bash
# terminal 1
python main.py
# terminal 2
python iot/iot_simulator.py                 # local
# or against the deployed app:
python iot/iot_simulator.py --url https://your-app.onrender.com
```

Open the dashboard, click **Live sensor feed**, and watch readings + the
recommendation update automatically as the "device" streams.

## Real hardware

Flash `esp32_sensor.ino` (Arduino IDE, ESP32 board support). Set `WIFI_SSID`,
`WIFI_PASS`, and `SERVER_URL` at the top, then upload. Adjust the sensor
register/calibration in `readNPK()`, `readPh()`, `readMoisture()` to your parts.

### Bill of materials (indicative, India)

| Component | Purpose | Approx ₹ |
|-----------|---------|---------:|
| ESP32 DevKit v1 | WiFi microcontroller | 400 |
| RS485 7-in-1 NPK soil sensor | N, P, K (also temp/moist/pH on some) | 2,500 |
| MAX485 / TTL-RS485 module | ESP32 ↔ RS485 | 80 |
| Soil pH sensor | soil acidity | 1,500 |
| DHT22 | air temperature + humidity | 250 |
| Capacitive soil-moisture sensor | moisture % | 120 |
| Jumpers, 5 V supply, enclosure | wiring & power | 400 |
| **Total** | | **~5,250** |

### Wiring (ESP32 pins used in the sketch)

| Signal | ESP32 pin |
|--------|-----------|
| DHT22 data | GPIO 4 |
| Soil-moisture (analog) | GPIO 34 (ADC1) |
| pH sensor (analog) | GPIO 35 (ADC1) |
| RS485 RX / TX | GPIO 16 / 17 |
| RS485 DE/RE (driver enable) | GPIO 5 |

## Security

`/api/ingest` is open by default (fine for a demo). To require auth, set a
`DEVICE_TOKEN` env var on the server and the same value in the sketch /
simulator (`--token`); requests must then send a matching `X-Device-Token`.
