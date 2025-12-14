import copy
import json
import operator
import re
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract
import serial


SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600

CONFIG_PATH = Path("crops.json")

MONITOR_INDEX = 1

EVENT_THRESHOLD = 0.8

FULL_FRAMES_REQUIRED = 3
FULL_SNAP_PERMILLE = 995

SHOW_DEBUG_WINDOWS = True
PRINT_OCR_DEBUG = True

HEALTH_FOCUS_X = (0.40, 0.62)
HEALTH_FOCUS_Y = (0.32, 0.78)

EVENTS_CROP = {"top": 233, "left": 1550, "width": 160, "height": 292}

HEALTH_CROP = {"top": 984, "left": 775, "width": 280, "height": 27}

OBJECTIVE_HOLD_SEC = 3.6
DEAD_FRAMES_REQUIRED = 4
ALIVE_FRAMES_REQUIRED = 2

TEMPLATES = {
    0: ("Baron", "template/baron.jpg"),
    1: ("Rift", "template/rift.jpg"),
    2: ("Cloud", "template/cloud.jpg"),
    3: ("Infernal", "template/infernal.png"),
    4: ("Mountain", "template/mountain.jpg"),
    5: ("Ocean", "template/ocean.jpg"),
    6: ("Elder", "template/elder.jpg"),
}


def make_tesseract_path():
    dash = chr(45)
    base = r"C:\Program Files\Tesseract"
    tail = r"OCR\tesseract.exe"
    return base + dash + tail


pytesseract.pytesseract.tesseract_cmd = make_tesseract_path()


def safe_serial_open():
    try:
        return serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    except Exception as exc:
        print("Serial open failed:", exc)
        return None


def safe_serial_write(arduino, text):
    if arduino is None:
        return
    try:
        arduino.write(text.encode())
    except Exception as exc:
        print("Serial write failed:", exc)


def build_region(monitor, crop):
    return {
        "mon": MONITOR_INDEX,
        "top": monitor["top"] + crop["top"],
        "left": monitor["left"] + crop["left"],
        "width": crop["width"],
        "height": crop["height"],
    }


def bgra_to_bgr(img_bgra):
    return img_bgra[:, :, :3]


def bgr_to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def prep_health_for_ocr(health_bgra):
    h, w, _ = health_bgra.shape

    x1 = max(0, int(w * HEALTH_FOCUS_X[0]))
    x2 = min(w, int(w * HEALTH_FOCUS_X[1]))
    y1 = max(0, int(h * HEALTH_FOCUS_Y[0]))
    y2 = min(h, int(h * HEALTH_FOCUS_Y[1]))
    focus = health_bgra[y1:y2, x1:x2]

    health_bgr = bgra_to_bgr(focus)
    gray = bgr_to_gray(health_bgr)

    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return bw


def ocr_health_text(health_bw):
    dash = chr(45)
    cfg = f"{dash}{dash}psm 7 {dash}c tessedit_char_whitelist=0123456789/"
    txt = pytesseract.image_to_string(health_bw, config=cfg)
    return txt.strip().replace(" ", "")


def parse_hp_flexible(txt, last_max):
    m = re.search(r"(\d+)\s*/\s*(\d+)", txt)
    if m:
        cur = int(m.group(1))
        mx = int(m.group(2))
        return cur, mx

    m2 = re.search(r"(\d{2,4})", txt)
    if m2:
        cur = int(m2.group(1))
        if last_max > 0:
            return cur, last_max
        return cur, cur

    return None


def clamp_int(v, low, high):
    if v < low:
        return low
    if v > high:
        return high
    return v


def quantize_permille(permille, step=50):
    permille = clamp_int(permille, 0, 1000)
    return int(round(permille / step) * step)


def accept_hp_reading(cur, mx, last_hp):
    last_cur, last_max = last_hp
    if mx <= 0:
        return False, cur, mx

    cur = min(cur, mx)

    if last_max > 0:
        if mx > last_max * 1.35 or mx < last_max * 0.65:
            return False, cur, mx

        jump_cap = max(50, int(last_max * 0.6))
        if abs(cur - last_cur) > jump_cap:
            return False, cur, mx

    return True, cur, mx


def hsv_to_rgb(h, s, v):
    h = float(h % 360.0)
    s = float(clamp_int(int(s * 1000), 0, 1000)) / 1000.0
    v = float(clamp_int(int(v * 1000), 0, 1000)) / 1000.0

    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c

    if h < 60:
        r1, g1, b1 = c, x, 0
    elif h < 120:
        r1, g1, b1 = x, c, 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x

    return (
        int((r1 + m) * 255),
        int((g1 + m) * 255),
        int((b1 + m) * 255),
    )


def color_from_permille(permille, dead_now):
    if dead_now:
        return (255, 0, 0)

    hp = clamp_int(permille, 0, 1000) / 1000.0
    hue = 120.0 * hp
    return hsv_to_rgb(hue, 1.0, 1.0)


def send_still_color(arduino, rgb):
    safe_serial_write(arduino, f"10,{rgb[0]},{rgb[1]},{rgb[2]}.")


def load_crop_config():
    if not CONFIG_PATH.exists():
        return MONITOR_INDEX, EVENTS_CROP, HEALTH_CROP
    try:
        data = json.loads(CONFIG_PATH.read_text())
        mon = int(data.get("monitor_index", MONITOR_INDEX))
        events = data.get("events_crop", EVENTS_CROP)
        health = data.get("health_crop", HEALTH_CROP)
        for key in ("top", "left", "width", "height"):
            if key not in events or key not in health:
                raise ValueError("crop missing keys")
        return mon, events, health
    except Exception as exc:
        print("Failed to load crop config, using defaults:", exc)
        return MONITOR_INDEX, EVENTS_CROP, HEALTH_CROP


def load_templates():
    loaded = {}
    for event_id, pair in TEMPLATES.items():
        name, path = pair
        img = cv2.imread(path, 0)
        if img is None:
            raise RuntimeError("Missing template: " + path)
        loaded[event_id] = (name, img)
    return loaded


def detect_event(events_gray, templates):
    best_event = None
    best_loc = None
    best_score = 0.0

    for event_id, pair in templates.items():
        _, tpl = pair
        res = cv2.matchTemplate(events_gray, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(np.amax(res))
        if score > best_score:
            best_score = score
            best_event = event_id
            best_loc = np.where(res >= EVENT_THRESHOLD)

    if best_event is None:
        return None, None

    if best_score < EVENT_THRESHOLD:
        return None, None

    return best_event, best_loc


def detect_team(events_bgr, loc):
    if loc is None:
        return None

    team = None

    for pt in zip(*loc[::-1]):
        x = int(pt[0])
        y = int(pt[1])

        x2 = operator.sub(x, 2)
        if x2 < 0:
            continue

        b = int(events_bgr[y, x2, 0])
        r = int(events_bgr[y, x2, 2])

        if b > 100 and r < 80:
            team = 1
        elif b < 80 and r > 120:
            team = 0

    return team


def main():
    monitor_index, events_crop, health_crop = load_crop_config()

    arduino = safe_serial_open()
    templates = load_templates()

    currentEvent = [-1, -1]
    lastEvent = [-1, -1]
    lastEventTime = 0.0

    good_hp = [0, 0]
    good_permille = 1000
    smooth_permille = 1000

    was_dead = False
    dead_confirm = 0
    alive_confirm = 0

    led_lock_until = 0.0
    last_led_send_time = 0.0
    last_rgb = None

    objective_hold_until = 0.0

    full_confirm = 0

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            monitor = sct.monitors[monitor_index]

            events_region = build_region(monitor, events_crop)
            events_bgra = np.array(sct.grab(events_region))
            events_bgr = bgra_to_bgr(events_bgra)
            events_gray = bgr_to_gray(events_bgr)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Events Debug", events_bgr)
                cv2.waitKey(1)

            health_region = build_region(monitor, health_crop)
            health_bgra = np.array(sct.grab(health_region))

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Health Crop RAW", bgra_to_bgr(health_bgra))
                cv2.waitKey(1)

            health_bw = prep_health_for_ocr(health_bgra)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Health OCR Input", health_bw)
                cv2.waitKey(1)

            healthTxt = ocr_health_text(health_bw)

            parsed = parse_hp_flexible(healthTxt, good_hp[1])
            if parsed is not None:
                cur, mx = parsed
                ok, cur, mx = accept_hp_reading(cur, mx, good_hp)

                if ok:
                    if mx < 0:
                        mx = 0
                    if cur < 0:
                        cur = 0
                    if mx > 0 and cur > mx:
                        cur = mx

                    good_hp[0] = cur
                    good_hp[1] = mx

                    if mx > 0:
                        good_permille = int(cur * 1000 / mx)
                    else:
                        good_permille = 0

                    if good_hp[1] > 0:
                        if cur == 0:
                            dead_confirm = dead_confirm + 1
                            alive_confirm = 0
                        else:
                            alive_confirm = alive_confirm + 1
                            dead_confirm = 0
                    else:
                        dead_confirm = 0
                        alive_confirm = 0

            dead_now = was_dead
            if dead_confirm >= DEAD_FRAMES_REQUIRED:
                dead_now = True
            elif alive_confirm >= ALIVE_FRAMES_REQUIRED:
                dead_now = False
            was_dead = dead_now

            smooth_permille = int((smooth_permille * 6 + good_permille * 4) / 10)
            target_rgb = color_from_permille(smooth_permille, dead_now)

            if PRINT_OCR_DEBUG:
                print(
                    "OCR:",
                    repr(healthTxt),
                    "HP:",
                    good_hp,
                    "permille:",
                    good_permille,
                    "smooth:",
                    smooth_permille,
                    "dead:",
                    dead_now,
                    "rgb:",
                    target_rgb,
                )

            event_id, loc = detect_event(events_gray, templates)

            if event_id is not None:
                currentEvent[1] = event_id
                team = detect_team(events_bgr, loc)
                currentEvent[0] = team
            else:
                currentEvent[0] = -1
                currentEvent[1] = -1

            team_changed = int(currentEvent[0]) != int(lastEvent[0])
            event_changed = int(currentEvent[1]) != int(lastEvent[1])
            timed_out = (time.time() - lastEventTime) > 3.0

            if (team_changed or timed_out) and event_changed and int(currentEvent[1]) != -1:
                if currentEvent[0] == 1:
                    if currentEvent[1] == 0:
                        safe_serial_write(arduino, "3,255,0,255.")
                        safe_serial_write(arduino, "4,255,0,255.")
                        print("Baron")
                    if currentEvent[1] == 1:
                        safe_serial_write(arduino, "2,255,0,255.")
                        print("Rift")
                    elif currentEvent[1] == 2:
                        safe_serial_write(arduino, "1,255,255,255.")
                        print("Cloud")
                    elif currentEvent[1] == 3:
                        safe_serial_write(arduino, "1,255,10,0.")
                        print("Infernal")
                    elif currentEvent[1] == 4:
                        safe_serial_write(arduino, "1,255,150,0.")
                        print("Mountain")
                    elif currentEvent[1] == 5:
                        safe_serial_write(arduino, "1,0,150,255.")
                        print("Ocean")
                    elif currentEvent[1] == 6:
                        safe_serial_write(arduino, "3,255,150,255.")
                        safe_serial_write(arduino, "4,255,150,255.")
                        print("Elder")

                lastEvent = copy.deepcopy(currentEvent)
                lastEventTime = time.time()

                objective_hold_until = now + OBJECTIVE_HOLD_SEC

            if now >= led_lock_until and now >= objective_hold_until:
                if operator.sub(now, last_led_send_time) >= 0.08:
                    if target_rgb != last_rgb:
                        send_still_color(arduino, target_rgb)
                        last_rgb = target_rgb
                        last_led_send_time = now


if __name__ == "__main__":
    main()
