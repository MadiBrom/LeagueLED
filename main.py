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

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None


print("RUNNING:", Path(__file__).resolve())

SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
AUTO_DETECT_SERIAL = True

CONFIG_PATH = Path("crops.json")

MONITOR_INDEX = 1

EVENT_THRESHOLD = 0.8

SHOW_DEBUG_WINDOWS = True
PRINT_OCR_DEBUG = True
DEBUG_PRINT_EVERY_SEC = 0.35

HEALTH_FOCUS_X = (0.40, 0.62)
HEALTH_FOCUS_Y = (0.32, 0.78)

EVENTS_CROP = {"top": 233, "left": 1550, "width": 160, "height": 292}
HEALTH_CROP = {"top": 984, "left": 775, "width": 280, "height": 27}

OBJECTIVE_HOLD_SEC = 3.6

DEAD_FRAMES_REQUIRED = 4
ALIVE_FRAMES_REQUIRED = 2

FULL_FRAMES_REQUIRED = 3
FULL_SNAP_PERMILLE = 995

MAX_CANDIDATE_FRAMES_REQUIRED = 6

USE_THIN_DIGITS = True
THIN_ITERATIONS = 1

HSV_WHITE_LOW = (0, 0, 160)
HSV_WHITE_HIGH = (179, 80, 255)

SOLID_COLOR_OPCODE = 10

# Your Arduino test proved G and B are swapped on output.
# So we send R, B, G instead of R, G, B.
SWAP_GB_ON_SEND = True

# New behavior for OCR glitches
OCR_GLITCH_HOLD_SEC = 0.40
OCR_FALLBACK_YELLOW_AFTER_SEC = 1.20
FALLBACK_YELLOW_RGB = (255, 255, 0)

# Reduce jitter around mid health colors
COLOR_STEP_PERMILLE = 25

TEMPLATES = {
    0: ("Baron", "template/baron.jpg"),
    1: ("Rift", "template/rift.jpg"),
    2: ("Cloud", "template/cloud.jpg"),
    3: ("Infernal", "template/infernal.png"),
    4: ("Mountain", "template/mountain.jpg"),
    5: ("Ocean", "template/ocean.jpg"),
    6: ("Elder", "template/elder.jpg"),
}

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not Path(TESSERACT_EXE).exists():
    raise RuntimeError("Tesseract exe not found at: " + TESSERACT_EXE)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
print("tesseract_cmd:", pytesseract.pytesseract.tesseract_cmd)


def pick_serial_port(preferred):
    if not AUTO_DETECT_SERIAL:
        return preferred
    if list_ports is None:
        return preferred

    ports = list(list_ports.comports())
    if not ports:
        return preferred

    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return p.device

    return ports[0].device


def safe_serial_open():
    port = pick_serial_port(SERIAL_PORT)
    try:
        print("Serial trying:", port)
        return serial.Serial(port, SERIAL_BAUD, timeout=1)
    except Exception as exc:
        print("Serial open failed:", exc)
        return None


_last_serial_fail = 0.0


def safe_serial_write(arduino, text):
    global _last_serial_fail
    if arduino is None:
        return
    try:
        arduino.write(text.encode())
    except Exception as exc:
        now = time.monotonic()
        if now - _last_serial_fail > 1.0:
            print("Serial write failed:", exc)
            _last_serial_fail = now


def to_hw_rgb(rgb):
    r, g, b = rgb
    if SWAP_GB_ON_SEND:
        return (r, b, g)
    return (r, g, b)


def send_color_cmd(arduino, opcode, rgb):
    r, g, b = to_hw_rgb(rgb)
    safe_serial_write(arduino, f"{int(opcode)},{int(r)},{int(g)},{int(b)}.")


def send_still_color(arduino, rgb):
    send_color_cmd(arduino, SOLID_COLOR_OPCODE, rgb)


def build_region(monitor, crop):
    return {
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

    # Upscale for crisp OCR on tiny HUD digits
    health_bgr = cv2.resize(
        health_bgr, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC
    )

    # HSV white filter: keeps the digits, dumps the colored bar noise
    hsv = cv2.cvtColor(health_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_WHITE_LOW, dtype=np.uint8)
    upper = np.array(HSV_WHITE_HIGH, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if USE_THIN_DIGITS:
        mask = cv2.erode(mask, kernel, iterations=THIN_ITERATIONS)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def ocr_health_text(health_bw):
    cfg = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=0123456789/ "
        "-c classify_bln_numeric_mode=1"
    )
    txt = pytesseract.image_to_string(health_bw, config=cfg)
    return txt.strip().replace(" ", "")


def is_ocr_glitch(txt):
    if not txt:
        return True
    if txt.count("/") != 1:
        return True
    if len(txt) > 10:
        return True
    return False


def parse_hp_flexible(txt, last_max):
    m = re.search(r"(\d+)\s*/\s*(\d+)", txt)
    if m:
        cur = int(m.group(1))
        mx = int(m.group(2))
        return cur, mx, True

    m2 = re.search(r"(\d{2,4})", txt)
    if m2:
        cur = int(m2.group(1))
        if last_max > 0:
            return cur, last_max, False
        return cur, cur, False

    return None


def repair_dropped_digit(cur, mx, last_cur, last_max):
    if mx <= 0:
        return cur

    s_cur = str(cur)

    # Do NOT try to "repair" single digit reads.
    # This preserves real death reads like 0/951,
    # and real low hp like 5/951, 9/951, etc.
    if len(s_cur) == 1:
        return cur

    candidates = [cur]

    if last_cur > 0:
        s_last = str(last_cur)
        if len(s_cur) < len(s_last):
            diff = len(s_last) - len(s_cur)
            if diff <= 2:
                try:
                    pref = s_last[:diff]
                    cand = int(pref + s_cur)
                    if 0 <= cand <= mx:
                        candidates.append(cand)
                except Exception:
                    pass

    if last_max > 0:
        s_lastm = str(last_max)
        if len(s_cur) < len(s_lastm):
            diff = len(s_lastm) - len(s_cur)
            if diff <= 2:
                try:
                    pref = s_lastm[:diff]
                    cand = int(pref + s_cur)
                    if 0 <= cand <= mx:
                        candidates.append(cand)
                except Exception:
                    pass

    if cur * 10 <= mx:
        candidates.append(cur * 10)
    if cur * 100 <= mx:
        candidates.append(cur * 100)

    if last_cur > 0:
        return min(candidates, key=lambda c: abs(c - last_cur))

    return max(candidates)


def repair_extra_digit(cur, mx, last_cur):
    if mx <= 0:
        return cur

    candidates = [cur]

    if cur // 10 <= mx:
        candidates.append(cur // 10)
    if cur // 100 <= mx:
        candidates.append(cur // 100)

    s = str(cur)
    if len(s) > 1:
        try:
            candidates.append(int(s[1:]))
        except Exception:
            pass
        try:
            candidates.append(int(s[:-1]))
        except Exception:
            pass

    if cur % 1000 <= mx:
        candidates.append(cur % 1000)
    if cur % 10000 <= mx:
        candidates.append(cur % 10000)

    if last_cur > 0:
        return min(candidates, key=lambda c: abs(c - last_cur))

    return min(candidates, key=lambda c: abs(mx - c))


def clamp_int(v, low, high):
    if v < low:
        return low
    if v > high:
        return high
    return v


def quantize_permille(permille, step):
    permille = clamp_int(int(permille), 0, 1000)
    return int(round(permille / float(step)) * step)


def accept_hp_reading(cur, mx, last_hp, strong, max_candidate, max_candidate_hits):
    last_cur, last_max = last_hp

    if mx <= 0:
        return False, cur, mx, max_candidate, max_candidate_hits

    orig_cur = cur
    cur = min(cur, mx)

    # Critical: if we were in weak parse mode and OCR produced a number bigger than max,
    # do not clamp it to max and accidentally go full green.
    if not strong and orig_cur > mx:
        return False, cur, mx, max_candidate, max_candidate_hits

    if strong:
        if mx < 200 or mx > 9999:
            return False, cur, mx, max_candidate, max_candidate_hits

        if last_max > 0:
            ratio = mx / float(last_max)
            if ratio > 1.35 or ratio < 0.65:
                if max_candidate == mx:
                    max_candidate_hits += 1
                else:
                    max_candidate = mx
                    max_candidate_hits = 1

                if max_candidate_hits >= MAX_CANDIDATE_FRAMES_REQUIRED:
                    max_candidate_hits = 0
                    return True, cur, mx, None, 0

                return False, cur, mx, max_candidate, max_candidate_hits

    if last_max > 0:
        jump_cap = max(50, int(last_max * 0.6))
        if abs(cur - last_cur) > jump_cap and not strong:
            return False, cur, mx, max_candidate, max_candidate_hits

    return True, cur, mx, None, 0


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
    good_permille = 500
    smooth_permille = 500

    was_dead = False
    dead_confirm = 0
    alive_confirm = 0
    full_confirm = 0

    last_led_send_time = 0.0
    last_rgb = None
    objective_hold_until = 0.0

    max_candidate = None
    max_candidate_hits = 0

    last_debug_time = 0.0
    last_debug_tuple = None

    ocr_lock_until = 0.0
    ocr_bad_since = None

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            monitor = sct.monitors[monitor_index]

            events_region = build_region(monitor, events_crop)
            events_bgra = np.array(sct.grab(events_region))
            events_bgr = bgra_to_bgr(events_bgra)
            events_gray = bgr_to_gray(events_bgr)

            health_region = build_region(monitor, health_crop)
            health_bgra = np.array(sct.grab(health_region))
            health_bw = prep_health_for_ocr(health_bgra)
            healthTxt = ocr_health_text(health_bw)

            frame_ok = False
            glitch = False

            if is_ocr_glitch(healthTxt):
                glitch = True
            else:
                parsed = parse_hp_flexible(healthTxt, good_hp[1])
                if parsed is None:
                    glitch = True
                else:
                    cur, mx, strong = parsed

                    cur = repair_dropped_digit(cur, mx, good_hp[0], good_hp[1])

                    if strong and mx > 0 and cur > mx:
                        fixed = repair_extra_digit(cur, mx, good_hp[0])
                        if fixed > mx:
                            glitch = True
                        else:
                            cur = fixed

                    if not glitch:
                        ok, cur, mx, max_candidate, max_candidate_hits = accept_hp_reading(
                            cur, mx, good_hp, strong, max_candidate, max_candidate_hits
                        )

                        if ok and mx > 0 and 0 <= cur <= mx:
                            good_hp[0] = cur
                            good_hp[1] = mx
                            good_permille = int(cur * 1000 / mx)

                            if strong and mx > 0 and cur == 0:
                                dead_confirm += 1
                                alive_confirm = 0
                            elif mx > 0 and cur > 0:
                                alive_confirm += 1
                                dead_confirm = 0
                            else:
                                dead_confirm = 0
                                alive_confirm = 0

                            if strong and mx > 0 and good_permille >= FULL_SNAP_PERMILLE:
                                full_confirm += 1
                            else:
                                full_confirm = 0

                            frame_ok = True
                        else:
                            glitch = True

            if not frame_ok:
                ocr_lock_until = max(ocr_lock_until, now + OCR_GLITCH_HOLD_SEC)
                if ocr_bad_since is None:
                    ocr_bad_since = now
            else:
                ocr_bad_since = None

            dead_now = was_dead
            if dead_confirm >= DEAD_FRAMES_REQUIRED:
                dead_now = True
            elif alive_confirm >= ALIVE_FRAMES_REQUIRED:
                dead_now = False
            was_dead = dead_now

            smooth_permille = int((smooth_permille * 6 + good_permille * 4) / 10)
            smooth_permille = quantize_permille(smooth_permille, COLOR_STEP_PERMILLE)

            if dead_now:
                target_rgb = (255, 0, 0)
            elif full_confirm >= FULL_FRAMES_REQUIRED:
                target_rgb = (0, 255, 0)
            else:
                target_rgb = color_from_permille(smooth_permille, dead_now)

            if now < ocr_lock_until:
                if last_rgb is not None:
                    target_rgb = last_rgb
                else:
                    target_rgb = FALLBACK_YELLOW_RGB

            if ocr_bad_since is not None and (now - ocr_bad_since) >= OCR_FALLBACK_YELLOW_AFTER_SEC:
                target_rgb = FALLBACK_YELLOW_RGB

            if PRINT_OCR_DEBUG:
                debug_tuple = (
                    healthTxt,
                    good_hp[0],
                    good_hp[1],
                    good_permille,
                    smooth_permille,
                    dead_now,
                    target_rgb,
                    max_candidate,
                    max_candidate_hits,
                    glitch,
                    round(max(0.0, ocr_lock_until - now), 2),
                )

                if (now - last_debug_time) >= DEBUG_PRINT_EVERY_SEC and debug_tuple != last_debug_tuple:
                    print(
                        "OCR:", repr(healthTxt),
                        "HP:", good_hp,
                        "permille:", good_permille,
                        "smooth:", smooth_permille,
                        "dead:", dead_now,
                        "rgb:", target_rgb,
                        "maxCandidate:", max_candidate,
                        "hits:", max_candidate_hits,
                        "glitch:", glitch,
                        "lock:", round(max(0.0, ocr_lock_until - now), 2),
                    )
                    last_debug_time = now
                    last_debug_tuple = debug_tuple

            event_id, loc = detect_event(events_gray, templates)

            if event_id is not None:
                currentEvent[1] = event_id
                currentEvent[0] = detect_team(events_bgr, loc)
            else:
                currentEvent[0] = -1
                currentEvent[1] = -1

            team_changed = int(currentEvent[0]) != int(lastEvent[0])
            event_changed = int(currentEvent[1]) != int(lastEvent[1])
            timed_out = (now - lastEventTime) > 3.0

            if (team_changed or timed_out) and event_changed and int(currentEvent[1]) != -1:
                if currentEvent[0] == 1:
                    if currentEvent[1] == 0:
                        send_color_cmd(arduino, 3, (255, 0, 255))
                        send_color_cmd(arduino, 4, (255, 0, 255))
                        print("Baron")
                    elif currentEvent[1] == 1:
                        send_color_cmd(arduino, 2, (255, 0, 255))
                        print("Rift")
                    elif currentEvent[1] == 2:
                        send_color_cmd(arduino, 1, (255, 255, 255))
                        print("Cloud")
                    elif currentEvent[1] == 3:
                        send_color_cmd(arduino, 1, (255, 10, 0))
                        print("Infernal")
                    elif currentEvent[1] == 4:
                        send_color_cmd(arduino, 1, (255, 150, 0))
                        print("Mountain")
                    elif currentEvent[1] == 5:
                        send_color_cmd(arduino, 1, (0, 150, 255))
                        print("Ocean")
                    elif currentEvent[1] == 6:
                        send_color_cmd(arduino, 3, (255, 150, 255))
                        send_color_cmd(arduino, 4, (255, 150, 255))
                        print("Elder")

                lastEvent = copy.deepcopy(currentEvent)
                lastEventTime = now
                objective_hold_until = now + OBJECTIVE_HOLD_SEC

            if now >= objective_hold_until:
                if (now - last_led_send_time) >= 0.08:
                    if target_rgb != last_rgb:
                        send_still_color(arduino, target_rgb)
                        last_rgb = target_rgb
                        last_led_send_time = now

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Events Debug", events_bgr)
                cv2.imshow("Health Crop RAW", bgra_to_bgr(health_bgra))
                cv2.imshow("Health OCR Input", health_bw)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
