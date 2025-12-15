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

# ================= SERIAL CONFIG =================

SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
AUTO_DETECT_SERIAL = True

# ================= OCR CONFIG =================

TESSERACT_EXE = r"C:\Users\tjfle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if not Path(TESSERACT_EXE).exists():
    raise RuntimeError("Tesseract exe not found at: " + TESSERACT_EXE)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
print("tesseract_cmd:", pytesseract.pytesseract.tesseract_cmd)

HSV_WHITE_LOW = (0, 0, 160)
HSV_WHITE_HIGH = (179, 80, 255)

USE_THIN_DIGITS = True
THIN_ITERATIONS = 1

# ================= SCREEN CONFIG =================

CONFIG_PATH = Path("crops.json")
MONITOR_INDEX = 1

EVENTS_CROP = {"top": 233, "left": 1550, "width": 160, "height": 292}
HEALTH_CROP = {"top": 1000, "left": 606, "width": 350, "height": 30 }
MANA_CROP = {"top": 1017, "left": 610, "width": 345, "height": 30 }
HEALTH_FOCUS_X = (0.40, 0.62)
HEALTH_FOCUS_Y = (0.32, 0.78)
MANA_FOCUS_X = (0.40, 0.62)
MANA_FOCUS_Y = (0.32, 0.78)

SHOW_DEBUG_WINDOWS = True
PRINT_DEBUG = True
DEBUG_PRINT_EVERY_SEC = 0.35

# ================= LED SEND CONFIG =================

SOLID_COLOR_OPCODE = 10
SWAP_GB_ON_SEND = True
SEND_EVERY_SEC = 0.08

COLOR_STEP_PERMILLE = 25

# HP glitch fallback behavior
HP_OCR_GLITCH_HOLD_SEC = 0.40
HP_OCR_FALLBACK_YELLOW_AFTER_SEC = 1.20
FALLBACK_YELLOW_RGB = (255, 255, 0)

DEAD_FRAMES_REQUIRED = 4
ALIVE_FRAMES_REQUIRED = 2

FULL_FRAMES_REQUIRED = 3
FULL_SNAP_PERMILLE = 995

MAX_CANDIDATE_FRAMES_REQUIRED = 6

# Mana dead like behavior
MANA_OCR_GLITCH_HOLD_SEC = 0.40
MANA_DEAD_DETECT_AFTER_SEC = 1.20
MANA_REVIVE_FORCE_BLUE_SEC = 1.00
MANA_ALIVE_FRAMES_REQUIRED = 2

RGB_STEP = 5
MANA_GRADIENT_LOW_RGB = (0, 255, 255)
MANA_GRADIENT_HIGH_RGB = (0, 0, 255)

# Objective detection
EVENT_THRESHOLD = 0.8
OBJECTIVE_PULSE_SEC = 2.0
OBJECTIVE_PRINT_EVERY_SEC = 0.25

TEMPLATES = {
    0: ("Baron", "template/baron.png"),
    1: ("Rift", "template/rift.png"),
    2: ("Cloud", "template/cloud.png"),
    3: ("Infernal", "template/infernal.png"),
    4: ("Mountain", "template/mountain.png"),
    5: ("Ocean", "template/ocean.png"),
    6: ("Elder", "template/elder.png"),
    7: ("Chem", "template/chem.png"),
    8: ("Hex", "template/hex.png"),

}


# ================= UTIL =================

def pick_serial_port(preferred: str) -> str:
    if not AUTO_DETECT_SERIAL or list_ports is None:
        return preferred

    ports = list(list_ports.comports())
    if not ports:
        return preferred

    devices = [p.device for p in ports]
    if preferred in devices:
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
        ar = serial.Serial(port, SERIAL_BAUD, timeout=0.05)
        time.sleep(2.0)
        return ar
    except Exception as exc:
        print("Serial open failed:", exc)
        return None


_last_serial_fail = 0.0


def safe_serial_write(arduino, text: str):
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
    return (r, b, g) if SWAP_GB_ON_SEND else (r, g, b)


def send_color_cmd(arduino, opcode: int, rgb):
    r, g, b = to_hw_rgb(rgb)
    safe_serial_write(arduino, f"{int(opcode)},{int(r)},{int(g)},{int(b)}.")


def send_still_color(arduino, rgb):
    send_color_cmd(arduino, SOLID_COLOR_OPCODE, rgb)


def clamp_int(v: int, low: int, high: int) -> int:
    if v < low:
        return low
    if v > high:
        return high
    return v


def quantize_permille(permille: int, step: int) -> int:
    permille = clamp_int(int(permille), 0, 1000)
    return int(round(permille / float(step)) * step)


def clamp_rgb(rgb):
    return (
        clamp_int(int(rgb[0]), 0, 255),
        clamp_int(int(rgb[1]), 0, 255),
        clamp_int(int(rgb[2]), 0, 255),
    )


def quantize_rgb(rgb, step: int):
    if step <= 1:
        return clamp_rgb(rgb)
    return clamp_rgb(
        (
            int(round(rgb[0] / float(step)) * step),
            int(round(rgb[1] / float(step)) * step),
            int(round(rgb[2] / float(step)) * step),
        )
    )


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


def load_crop_config():
    if not CONFIG_PATH.exists():
        return MONITOR_INDEX, EVENTS_CROP, HEALTH_CROP, MANA_CROP

    try:
        data = json.loads(CONFIG_PATH.read_text())
        mon = int(data.get("monitor_index", MONITOR_INDEX))
        events = data.get("events_crop", EVENTS_CROP)
        health = data.get("health_crop", HEALTH_CROP)
        mana = data.get("mana_crop", MANA_CROP)
        return mon, events, health, mana
    except Exception as exc:
        print("Failed to load crop config, using defaults:", exc)
        return MONITOR_INDEX, EVENTS_CROP, HEALTH_CROP, MANA_CROP


# ================= OCR PIPE =================

def prep_bar_for_ocr(bar_bgra, focus_x, focus_y):
    h, w, _ = bar_bgra.shape

    x1 = max(0, int(w * focus_x[0]))
    x2 = min(w, int(w * focus_x[1]))
    y1 = max(0, int(h * focus_y[0]))
    y2 = min(h, int(h * focus_y[1]))
    focus = bar_bgra[y1:y2, x1:x2]

    bar_bgr = bgra_to_bgr(focus)
    bar_bgr = cv2.resize(bar_bgr, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_WHITE_LOW, dtype=np.uint8)
    upper = np.array(HSV_WHITE_HIGH, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if USE_THIN_DIGITS:
        mask = cv2.erode(mask, kernel, iterations=THIN_ITERATIONS)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def ocr_bar_text(bar_bw):
    cfg = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=0123456789/ "
        "-c classify_bln_numeric_mode=1"
    )
    txt = pytesseract.image_to_string(bar_bw, config=cfg)
    return txt.strip().replace(" ", "")


def is_ocr_glitch(txt: str) -> bool:
    if not txt:
        return True
    if txt.count("/") != 1:
        return True
    if len(txt) > 10:
        return True
    return False


def parse_pair_flexible(txt: str, last_max: int):
    m = re.search(r"(\d+)\s*/\s*(\d+)", txt)
    if m:
        cur = int(m.group(1))
        mx = int(m.group(2))
        return cur, mx, True

    m2 = re.search(r"(\d{1,4})", txt)
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


def accept_reading(cur, mx, last_pair, strong, max_candidate, max_candidate_hits):
    last_cur, last_max = last_pair

    if mx <= 0:
        return False, cur, mx, max_candidate, max_candidate_hits

    orig_cur = cur
    cur = min(cur, mx)

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


# ================= COLORS =================

def hsv_to_rgb(h, s, v):
    h = float(h % 360.0)
    s = float(s)
    v = float(v)

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


def hp_color_from_permille(permille: int, dead_now: bool):
    if dead_now:
        return (255, 0, 0)
    p = clamp_int(permille, 0, 1000) / 1000.0
    hue = 120.0 * p
    return hsv_to_rgb(hue, 1.0, 1.0)


def mana_gradient_from_permille(permille: int):
    t = 1.0 - (clamp_int(permille, 0, 1000) / 1000.0)
    r0, g0, b0 = MANA_GRADIENT_LOW_RGB
    r1, g1, b1 = MANA_GRADIENT_HIGH_RGB
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


# ================= OBJECTIVES =================

def load_templates():
    loaded = {}
    for event_id, (name, path) in TEMPLATES.items():
        img = cv2.imread(path, 0)
        if img is None:
            raise RuntimeError("Missing template: " + path)
        loaded[event_id] = (name, img)
    return loaded


def detect_event(events_gray, templates):
    best_event = None
    best_name = None
    best_loc = None
    best_score = 0.0

    for event_id, pair in templates.items():
        name, tpl = pair
        res = cv2.matchTemplate(events_gray, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(np.amax(res))
        if score > best_score:
            best_score = score
            best_event = event_id
            best_name = name
            best_loc = np.where(res >= EVENT_THRESHOLD)

    if best_event is None or best_score < EVENT_THRESHOLD:
        return None, None, best_score, best_name

    return best_event, best_loc, best_score, best_name


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


def get_objective_payload(team, event_id):
    if team != 1:
        return None, []

    if event_id == 0:
        return "Baron", [(3, (255, 0, 255)), (4, (255, 0, 255))]
    if event_id == 1:
        return "Rift", [(2, (255, 0, 255))]
    if event_id == 2:
        return "Cloud", [(1, (255, 255, 255))]
    if event_id == 3:
        return "Infernal", [(1, (255, 10, 0))]
    if event_id == 4:
        return "Mountain", [(1, (255, 150, 0))]
    if event_id == 5:
        return "Ocean", [(1, (0, 150, 255))]
    if event_id == 6:
        return "Elder", [(3, (255, 150, 255)), (4, (255, 150, 255))]
    if event_id == 7:
        return "Chem", [(1, (0, 255, 0))]
    if event_id == 8:
        return "Hex", [(1, (255, 255, 0))]
    return None, []


def send_objective(arduino, team, event_id):
    name, payload = get_objective_payload(team, event_id)
    if not payload:
        return name

    for opcode, rgb in payload:
        send_color_cmd(arduino, opcode, rgb)

    return name


# ================= MODE INPUT =================

def pump_mode_messages(arduino, show_mana: bool) -> bool:
    if arduino is None:
        return show_mana

    # read any full lines sitting in the buffer
    while True:
        try:
            if arduino.in_waiting <= 0:
                break
            line = arduino.readline().decode(errors="ignore").strip()
        except Exception:
            break

        if not line:
            break

        if line == "MODE:MANA":
            show_mana = True
            print(">> MODE set to MANA")
        elif line == "MODE:HP":
            show_mana = False
            print(">> MODE set to HP")

    return show_mana


# ================= MAIN =================

def main():
    monitor_index, events_crop, health_crop, mana_crop = load_crop_config()

    arduino = safe_serial_open()
    templates = load_templates()

    show_mana = False

    # objective state
    last_event = [-1, -1]
    last_event_time = 0.0
    active_obj_name = None
    active_obj_team = -1
    active_obj_event_id = -1
    active_obj_until = 0.0
    last_obj_send_time = 0.0
    last_obj_print_time = 0.0

    # HP state
    good_hp = [0, 0]
    hp_permille = 500
    hp_smooth = 500
    hp_max_candidate = None
    hp_max_hits = 0
    hp_ocr_lock_until = 0.0
    hp_bad_since = None
    was_dead = False
    dead_confirm = 0
    alive_confirm = 0
    full_confirm = 0
    hp_last_rgb = None

    # Mana state
    good_mana = [0, 0]
    mana_permille = 500
    mana_smooth = 500
    mana_max_candidate = None
    mana_max_hits = 0
    mana_lock_until = 0.0
    mana_bad_since = None
    mana_dead_now = False
    mana_alive_confirm = 0
    revive_blue_until = 0.0
    last_stable_mana_rgb = None

    # send throttling
    last_sent_rgb = None
    last_send_time = 0.0

    last_debug_time = 0.0

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            show_mana = pump_mode_messages(arduino, show_mana)

            monitor = sct.monitors[monitor_index]

            # ========= EVENTS =========
            events_region = build_region(monitor, events_crop)
            events_bgra = np.array(sct.grab(events_region))
            events_bgr = bgra_to_bgr(events_bgra)
            events_gray = bgr_to_gray(events_bgr)

            event_id, loc, event_score, event_name = detect_event(events_gray, templates)

            current_event = [-1, -1]
            if event_id is not None:
                current_event[1] = event_id
                team = detect_team(events_bgr, loc)
                current_event[0] = team if team is not None else -1

            team_changed = int(current_event[0]) != int(last_event[0])
            event_changed = int(current_event[1]) != int(last_event[1])
            timed_out = (now - last_event_time) > 3.0

            if (team_changed or timed_out) and event_changed and int(current_event[1]) != -1:
                obj_name = send_objective(arduino, int(current_event[0]), int(current_event[1]))

                if obj_name is not None:
                    active_obj_name = obj_name
                    active_obj_team = int(current_event[0])
                    active_obj_event_id = int(current_event[1])
                    active_obj_until = now + OBJECTIVE_PULSE_SEC
                    last_obj_send_time = 0.0
                    last_obj_print_time = 0.0

                    print(
                        "OBJECTIVE TRIGGERED:",
                        obj_name,
                        "team",
                        active_obj_team,
                        "score",
                        round(float(event_score), 3),
                        "pulseSec",
                        OBJECTIVE_PULSE_SEC,
                    )

                last_event = copy.deepcopy(current_event)
                last_event_time = now

            # ========= HP OCR =========
            hp_region = build_region(monitor, health_crop)
            hp_bgra = np.array(sct.grab(hp_region))
            hp_bw = prep_bar_for_ocr(hp_bgra, HEALTH_FOCUS_X, HEALTH_FOCUS_Y)
            hp_txt = ocr_bar_text(hp_bw)

            hp_frame_ok = False
            hp_glitch = False

            if is_ocr_glitch(hp_txt):
                hp_glitch = True
            else:
                parsed = parse_pair_flexible(hp_txt, good_hp[1])
                if parsed is None:
                    hp_glitch = True
                else:
                    cur, mx, strong = parsed
                    cur = repair_dropped_digit(cur, mx, good_hp[0], good_hp[1])

                    if strong and mx > 0 and cur > mx:
                        fixed = repair_extra_digit(cur, mx, good_hp[0])
                        if fixed > mx:
                            hp_glitch = True
                        else:
                            cur = fixed

                    if not hp_glitch:
                        ok, cur, mx, hp_max_candidate, hp_max_hits = accept_reading(
                            cur, mx, good_hp, strong, hp_max_candidate, hp_max_hits
                        )
                        if ok and mx > 0 and 0 <= cur <= mx:
                            good_hp[0] = cur
                            good_hp[1] = mx
                            hp_permille = int(cur * 1000 / mx)
                            hp_frame_ok = True
                        else:
                            hp_glitch = True

            if not hp_frame_ok:
                hp_ocr_lock_until = max(hp_ocr_lock_until, now + HP_OCR_GLITCH_HOLD_SEC)
                if hp_bad_since is None:
                    hp_bad_since = now
            else:
                hp_bad_since = None

            if hp_frame_ok:
                cur, mx = good_hp
                strong = True if ("/" in hp_txt) else False
                if strong and mx > 0 and cur == 0:
                    dead_confirm += 1
                    alive_confirm = 0
                elif mx > 0 and cur > 0:
                    alive_confirm += 1
                    dead_confirm = 0
                else:
                    dead_confirm = 0
                    alive_confirm = 0

                if strong and mx > 0 and hp_permille >= FULL_SNAP_PERMILLE:
                    full_confirm += 1
                else:
                    full_confirm = 0

            dead_now = was_dead
            if dead_confirm >= DEAD_FRAMES_REQUIRED:
                dead_now = True
            elif alive_confirm >= ALIVE_FRAMES_REQUIRED:
                dead_now = False
            was_dead = dead_now

            hp_smooth = int((hp_smooth * 6 + hp_permille * 4) / 10)
            hp_smooth = quantize_permille(hp_smooth, COLOR_STEP_PERMILLE)

            hp_rgb = hp_color_from_permille(hp_smooth, dead_now)
            if (not dead_now) and (full_confirm >= FULL_FRAMES_REQUIRED):
                hp_rgb = (0, 255, 0)

            if now < hp_ocr_lock_until:
                hp_rgb = hp_last_rgb if hp_last_rgb is not None else FALLBACK_YELLOW_RGB

            if hp_bad_since is not None and (now - hp_bad_since) >= HP_OCR_FALLBACK_YELLOW_AFTER_SEC:
                hp_rgb = FALLBACK_YELLOW_RGB

            hp_last_rgb = hp_rgb

            # ========= MANA OCR =========
            mana_region = build_region(monitor, mana_crop)
            mana_bgra = np.array(sct.grab(mana_region))
            mana_bw = prep_bar_for_ocr(mana_bgra, MANA_FOCUS_X, MANA_FOCUS_Y)
            mana_txt = ocr_bar_text(mana_bw)

            mana_frame_ok = False
            mana_glitch = False

            if is_ocr_glitch(mana_txt):
                mana_glitch = True
            else:
                parsed = parse_pair_flexible(mana_txt, good_mana[1])
                if parsed is None:
                    mana_glitch = True
                else:
                    cur, mx, strong = parsed
                    cur = repair_dropped_digit(cur, mx, good_mana[0], good_mana[1])

                    if strong and mx > 0 and cur > mx:
                        fixed = repair_extra_digit(cur, mx, good_mana[0])
                        if fixed > mx:
                            mana_glitch = True
                        else:
                            cur = fixed

                    if not mana_glitch:
                        ok, cur, mx, mana_max_candidate, mana_max_hits = accept_reading(
                            cur, mx, good_mana, strong, mana_max_candidate, mana_max_hits
                        )
                        if ok and mx > 0 and 0 <= cur <= mx:
                            good_mana[0] = cur
                            good_mana[1] = mx
                            mana_permille = int(cur * 1000 / mx)
                            mana_frame_ok = True
                        else:
                            mana_glitch = True

            if not mana_frame_ok:
                mana_lock_until = max(mana_lock_until, now + MANA_OCR_GLITCH_HOLD_SEC)
                if mana_bad_since is None:
                    mana_bad_since = now
            else:
                mana_bad_since = None

            mana_smooth = int((mana_smooth * 6 + mana_permille * 4) / 10)
            mana_smooth = quantize_permille(mana_smooth, COLOR_STEP_PERMILLE)

            mana_rgb = quantize_rgb(mana_gradient_from_permille(mana_smooth), RGB_STEP)

            mana_out = mana_rgb

            if now < mana_lock_until:
                mana_out = last_stable_mana_rgb if last_stable_mana_rgb is not None else FALLBACK_YELLOW_RGB

            if mana_bad_since is not None and (now - mana_bad_since) >= MANA_DEAD_DETECT_AFTER_SEC:
                mana_dead_now = True

            if mana_dead_now:
                mana_out = last_stable_mana_rgb if last_stable_mana_rgb is not None else FALLBACK_YELLOW_RGB

                if mana_frame_ok:
                    mana_alive_confirm += 1
                else:
                    mana_alive_confirm = 0

                if mana_alive_confirm >= MANA_ALIVE_FRAMES_REQUIRED:
                    mana_dead_now = False
                    mana_alive_confirm = 0
                    revive_blue_until = now + MANA_REVIVE_FORCE_BLUE_SEC

            if not mana_dead_now:
                last_stable_mana_rgb = mana_out

            if (not mana_dead_now) and (now < revive_blue_until):
                mana_out = quantize_rgb(MANA_GRADIENT_LOW_RGB, RGB_STEP)

            # ========= OUTPUT SELECTION =========
            base_rgb = mana_out if show_mana else hp_rgb
            mode_name = "MANA" if show_mana else "HP"

            # Objective pulse overrides base output
            if active_obj_name is not None and now < active_obj_until:
                if arduino is not None and (now - last_obj_send_time) >= SEND_EVERY_SEC:
                    send_objective(arduino, active_obj_team, active_obj_event_id)
                    last_obj_send_time = now

                if (now - last_obj_print_time) >= OBJECTIVE_PRINT_EVERY_SEC:
                    left = round(max(0.0, active_obj_until - now), 2)
                    print("OBJECTIVE PULSE:", active_obj_name, "leftSec", left)
                    last_obj_print_time = now

            elif active_obj_name is not None and now >= active_obj_until:
                print("OBJECTIVE END:", active_obj_name)
                active_obj_name = None
                active_obj_team = -1
                active_obj_event_id = -1
                last_sent_rgb = None

            # Normal send only when no objective pulse
            if active_obj_name is None:
                if arduino is not None and (now - last_send_time) >= SEND_EVERY_SEC:
                    if base_rgb != last_sent_rgb:
                        send_still_color(arduino, base_rgb)
                        last_sent_rgb = base_rgb
                        last_send_time = now

            # ========= DEBUG =========
            if PRINT_DEBUG and (now - last_debug_time) >= DEBUG_PRINT_EVERY_SEC:
                print(
                    f"MODE {mode_name} | "
                    f"HP [{hp_txt}] {good_hp[0]}/{good_hp[1]} perm {hp_permille} smooth {hp_smooth} rgb {hp_rgb} | "
                    f"MANA [{mana_txt}] {good_mana[0]}/{good_mana[1]} perm {mana_permille} smooth {mana_smooth} rgb {mana_out} | "
                    f"OUT {base_rgb}"
                )
                last_debug_time = now

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Events Debug", events_bgr)
                cv2.imshow("HP OCR Input", hp_bw)
                cv2.imshow("Mana OCR Input", mana_bw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
