#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const uint8_t YAW_CHANNEL   = 0;
const uint8_t PITCH_CHANNEL = 1;

const int          SERVO_FREQ_HZ      = 50;
const int          NEUTRAL_US         = 1500;
const int          MIN_US             = 1200;
const int          MAX_US             = 1800;
const int          MAX_DELTA_US       = 600;
const unsigned long WATCHDOG_MS       = 500;
const unsigned long SERVO_INTERVAL_MS = 5;   // 200 Hz servo updates

int          yawUs       = NEUTRAL_US;
int          pitchUs     = NEUTRAL_US;
int          targetYawUs   = NEUTRAL_US;
int          targetPitchUs = NEUTRAL_US;
unsigned long lastCommandMs    = 0;
unsigned long lastServoUpdateMs = 0;

// Non-blocking line buffer — avoids readStringUntil blocking latency
String  rxBuf = "";

// ── helpers ──────────────────────────────────────────────────────────────────

int usToTicks(int pulseUs) {
  return (int)((pulseUs / (1000000.0f / SERVO_FREQ_HZ)) * 4096.0f);
}

int clampPulse(int us) {
  return us < MIN_US ? MIN_US : (us > MAX_US ? MAX_US : us);
}

int rampToward(int current, int target) {
  target = clampPulse(target);
  if (target > current + MAX_DELTA_US) return current + MAX_DELTA_US;
  if (target < current - MAX_DELTA_US) return current - MAX_DELTA_US;
  return target;
}

void writeServos() {
  pwm.setPWM(YAW_CHANNEL,   0, usToTicks(yawUs));
  pwm.setPWM(PITCH_CHANNEL, 0, usToTicks(pitchUs));
}

int normalizedToPulse(float v) {
  if (v < -1.0f) v = -1.0f;
  if (v >  1.0f) v =  1.0f;
  return v >= 0.0f
    ? NEUTRAL_US + (int)(v * (MAX_US   - NEUTRAL_US))
    : NEUTRAL_US + (int)(v * (NEUTRAL_US - MIN_US));
}

void requestNeutral() {
  targetYawUs   = NEUTRAL_US;
  targetPitchUs = NEUTRAL_US;
}

// ── command processor ─────────────────────────────────────────────────────────

void processLine(String &line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    lastCommandMs = millis();

  } else if (line == "NEUTRAL") {
    requestNeutral();
    lastCommandMs = millis();
    Serial.println("OK NEUTRAL");

  } else if (line.startsWith("SET ")) {
    int s1 = line.indexOf(' ');
    int s2 = line.indexOf(' ', s1 + 1);
    if (s2 > 0) {
      float yaw   = line.substring(s1 + 1, s2).toFloat();
      float pitch = line.substring(s2 + 1).toFloat();
      targetYawUs   = normalizedToPulse(yaw);
      targetPitchUs = normalizedToPulse(pitch);
      lastCommandMs = millis();
      // Skip "OK SET" echo — saves ~0.15 ms of serial TX per command
    } else {
      Serial.println("ERR BAD_SET");
    }

  } else {
    Serial.println("ERR UNKNOWN");
  }
}

// ── setup / loop ──────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(500000);          // 500 kbaud — ~0.3 ms per command vs ~1.7 ms at 115200
  Wire.begin();
  Wire.setClock(400000);         // 400 kHz Fast Mode — 4× faster I2C to PCA9685
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ_HZ);
  delay(10);
  yawUs = pitchUs = targetYawUs = targetPitchUs = NEUTRAL_US;
  writeServos();
  lastCommandMs = lastServoUpdateMs = millis();
  Serial.println("READY");
}

void loop() {
  // ── non-blocking serial read ──────────────────────────────────────────────
  // Reads one character at a time; never blocks; processes full lines immediately.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processLine(rxBuf);
      rxBuf = "";
    } else if (c != '\r') {
      rxBuf += c;
    }
  }

  // ── servo update at 200 Hz ────────────────────────────────────────────────
  unsigned long now = millis();
  if (now - lastServoUpdateMs >= SERVO_INTERVAL_MS) {
    lastServoUpdateMs = now;

    if (now - lastCommandMs > WATCHDOG_MS) {
      requestNeutral();
    }

    yawUs   = rampToward(yawUs,   targetYawUs);
    pitchUs = rampToward(pitchUs, targetPitchUs);
    writeServos();
  }
}
