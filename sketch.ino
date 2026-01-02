#include <Arduino.h>
const int pinR = 9;
const int pinG = 6;
const int pinB = 5;
const int buttonPin = 2;

int mode = 0;
int hp = 0, hpMax = 100;
int mana = 0, manaMax = 100;

unsigned long lastDebounce = 0;
int lastButtonState = HIGH;
int buttonState = HIGH;
const unsigned long debounceDelay = 50;

unsigned long lastFlashUntil = 0;
bool flashing = false;

void setup() {
  Serial.begin(115200);
  pinMode(pinR, OUTPUT);
  pinMode(pinG, OUTPUT);
  pinMode(pinB, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);

  analogWrite(pinR, 0);
  analogWrite(pinG, 0);
  analogWrite(pinB, 0);
}

void loop() {
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    parseLine(line);
  }

  int reading = digitalRead(buttonPin);
  if (reading != lastButtonState) {
    lastDebounce = millis();
  }
  if ((millis() - lastDebounce) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == LOW) {
        mode = 1 - mode;
      }
    }
  }
  lastButtonState = reading;

  if (flashing) {
    if (millis() < lastFlashUntil) {
      analogWrite(pinR, 255);
      analogWrite(pinG, 255);
      analogWrite(pinB, 255);
      return;
    } else {
      flashing = false;
    }
  }

  if (mode == 0) {
    int g = 0;
    if (hpMax > 0) g = map(constrain(hp, 0, hpMax), 0, hpMax, 0, 255);
    analogWrite(pinR, 0);
    analogWrite(pinG, g);
    analogWrite(pinB, 0);
  } else {
    int b = 0;
    if (manaMax > 0) b = map(constrain(mana, 0, manaMax), 0, manaMax, 0, 255);
    analogWrite(pinR, 0);
    analogWrite(pinG, 0);
    analogWrite(pinB, b);
  }
}

void parseLine(String s) {
  s.trim();
  if (s.length() < 2) return;
  char t = s.charAt(0);
  if (t == 'H') {
    int colon = s.indexOf(':');
    int comma = s.indexOf(',');
    if (colon >= 0 && comma > colon) {
      String a = s.substring(colon + 1, comma);
      String b = s.substring(comma + 1);
      hp = a.toInt();
      hpMax = b.toInt();
    }
  } else if (t == 'M') {
    int colon = s.indexOf(':');
    int comma = s.indexOf(',');
    if (colon >= 0 && comma > colon) {
      String a = s.substring(colon + 1, comma);
      String b = s.substring(comma + 1);
      mana = a.toInt();
      manaMax = b.toInt();
    }
  } else if (t == 'E') {
    int colon = s.indexOf(':');
    String ev = (colon >= 0) ? s.substring(colon + 1) : s.substring(1);
    ev.trim();
    if (ev == "drake") {
      flashWhite();
    }
  } else {
    int comma1 = s.indexOf(',');
    int comma2 = s.indexOf(',', comma1 + 1);
    int comma3 = s.indexOf(',', comma2 + 1);
    if (comma1 > 0 && comma2 > comma1 && comma3 > comma2) {
      int opcode = s.substring(0, comma1).toInt();
      int r = s.substring(comma1 + 1, comma2).toInt();
      int g = s.substring(comma2 + 1, comma3).toInt();
      int dot = s.indexOf('.', comma3 + 1);
      int b = 0;
      if (dot > comma3) {
        b = s.substring(comma3 + 1, dot).toInt();
      } else {
        b = s.substring(comma3 + 1).toInt();
      }
      analogWrite(pinR, r);
      analogWrite(pinG, g);
      analogWrite(pinB, b);
      delay(350);
    }
  }
}

void flashWhite() {
  flashing = true;
  lastFlashUntil = millis() + 450;
}
