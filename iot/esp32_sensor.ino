/*
 * Crop Recommendation AI — ESP32 field sensor node
 * Final Year Project: System using IoT and Crop Recommendation AI
 *
 * Reads a soil NPK sensor (RS485/Modbus), a soil pH sensor, a DHT22
 * (air temperature + humidity) and a capacitive soil-moisture probe,
 * then POSTs the readings as JSON to the app's /api/ingest endpoint over
 * WiFi. The server stores the reading and returns the crop recommendation.
 *
 * Libraries (install via Arduino Library Manager):
 *   - ArduinoJson         (JSON build/parse)
 *   - DHT sensor library  (Adafruit, for DHT22)
 *   - ModbusMaster        (RS485 NPK sensor)
 * Board: any ESP32 dev board (e.g. ESP32 DevKit v1).
 *
 * NOTE: sensor register addresses / calibration differ by vendor — adjust
 * readNPK(), readPh() and readMoisture() to match your hardware.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <ModbusMaster.h>

// ---------- CONFIG: edit these ----------
const char* WIFI_SSID   = "YOUR_WIFI";
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
// Your deployed app (or PC's LAN IP during testing), no trailing slash:
const char* SERVER_URL  = "https://your-app.onrender.com";
const char* DEVICE_ID   = "esp32-01";
const char* DEVICE_TOKEN = "";          // set if the server sets DEVICE_TOKEN
const unsigned long SEND_EVERY_MS = 10000;  // 10 s

// ---------- pins ----------
#define DHTPIN     4
#define DHTTYPE    DHT22
#define MOISTURE_PIN 34     // analog (ADC1)
#define PH_PIN       35     // analog (ADC1)
#define RS485_RX     16
#define RS485_TX     17
#define RS485_DE_RE  5      // driver enable for the RS485 transceiver

DHT dht(DHTPIN, DHTTYPE);
ModbusMaster npc;           // NPK sensor over Modbus

void preTx()  { digitalWrite(RS485_DE_RE, HIGH); }
void postTx() { digitalWrite(RS485_DE_RE, LOW); }

void setup() {
  Serial.begin(115200);
  pinMode(RS485_DE_RE, OUTPUT);
  digitalWrite(RS485_DE_RE, LOW);
  dht.begin();

  Serial2.begin(9600, SERIAL_8N1, RS485_RX, RS485_TX);
  npc.begin(1, Serial2);      // slave id 1
  npc.preTransmission(preTx);
  npc.postTransmission(postTx);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.printf("\nConnected: %s\n", WiFi.localIP().toString().c_str());
}

// --- sensor reads (adjust registers/calibration to your hardware) ---
void readNPK(float &n, float &p, float &k) {
  // Common 7-in-1 NPK register layout: N=0x1E, P=0x1F, K=0x20
  n = p = k = 0;
  if (npc.readHoldingRegisters(0x1E, 3) == npc.ku8MBSuccess) {
    n = npc.getResponseBuffer(0);
    p = npc.getResponseBuffer(1);
    k = npc.getResponseBuffer(2);
  }
}

float readMoisture() {
  int raw = analogRead(MOISTURE_PIN);          // 0..4095
  // calibrate: dry ~3200, wet ~1200 (typical capacitive probe)
  float pct = map(raw, 3200, 1200, 0, 100);
  return constrain(pct, 0, 100);
}

float readPh() {
  int raw = analogRead(PH_PIN);
  float voltage = raw * (3.3 / 4095.0);
  // calibrate against pH 4.0 / 7.0 buffer solutions:
  return constrain(7.0 + (2.5 - voltage) / 0.18, 3.5, 9.5);
}

void loop() {
  float n, p, k;
  readNPK(n, p, k);
  float moisture = readMoisture();
  float ph = readPh();
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  if (isnan(temperature)) temperature = 25.0;
  if (isnan(humidity))    humidity = 60.0;
  float rainfall = 100.0;   // no rain gauge -> the server can use a weather API

  // build JSON
  StaticJsonDocument<256> doc;
  doc["device_id"]     = DEVICE_ID;
  doc["N"]             = n;
  doc["P"]             = p;
  doc["K"]             = k;
  doc["ph"]            = ph;
  doc["soil_moisture"] = moisture;
  doc["temperature"]   = temperature;
  doc["humidity"]      = humidity;
  doc["rainfall"]      = rainfall;
  String body;
  serializeJson(doc, body);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(String(SERVER_URL) + "/api/ingest");
    http.addHeader("Content-Type", "application/json");
    if (strlen(DEVICE_TOKEN) > 0) http.addHeader("X-Device-Token", DEVICE_TOKEN);

    int code = http.POST(body);
    if (code == 200) {
      StaticJsonDocument<2048> resp;
      if (deserializeJson(resp, http.getString()) == DeserializationError::Ok) {
        const char* crop = resp["result"]["best"]["crop"] | "?";
        float conf = resp["result"]["best"]["confidence"] | 0.0;
        Serial.printf("Recommended: %s (%.0f%%)\n", crop, conf * 100);
      }
    } else {
      Serial.printf("POST failed: HTTP %d\n", code);
    }
    http.end();
  }

  delay(SEND_EVERY_MS);
}
