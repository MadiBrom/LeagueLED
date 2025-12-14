import json
import re
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract

try:
    import serial
except Exception:
    serial = None

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None


print("RUNNING:", Path(__file__).resolve())

ENABLE_SERIAL = True

SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
AUTO_DETECT_SERIAL = True

CONFIG_PATH = Path("crops.json")

MONITOR_INDEX = 1

SHOW_DEBUG_WINDOWS = True
PRINT_OCR_DEBUG = True
DEBUG_PRINT_EVERY_SEC = 0.35

MANA_FOCUS_X = (0.40, 0.62)
MANA_FOCUS_Y = (0.32, 0.78)

EVENTS_CROP = {"top": 233, "left": 1550, "width": 160, "height": 292}
MANA_CROP = {"top": 996, "left": 775, "width": 280, "height": 27}

USE_THIN_DIGITS = True
THIN_ITERATIONS = 1

HSV_WHITE_LOW = (0, 0, 160)
HSV_WHITE_HIGH = (179, 80, 255)

SOLID_COLOR_OPCODE = 10

SWAP_GB_ON_SEND = True

OCR_GLITCH_HOLD_SEC = 0.40

DEAD_DETECT_AFTER_SEC = 1.20

REVIVE_FORCE_BLUE_SEC = 1.00
ALIVE_FRAMES_REQUIRED = 2

FALLBACK_YELLOW_RGB = (255, 255, 0)

COLOR_STEP_PERMILLE = 25
RGB_STEP = 5

MAX_CANDIDATE_FRAMES_REQUIRED = 6

SEND_EVERY_SEC = 0.08

MANA_GRADIENT_LOW_RGB = (20, 40, 255)
MANA_GRADIENT_HIGH_RGB = (170, 30, 200)

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

    devices = [p.device for p in ports]
    if preferred in devices:
        return preferred

    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return p.device

    return ports[0].device


def safe_serial_open():
    if not ENABLE_SERIAL:
        return None
    if serial is None:
        print("pyserial not installed, serial disabled")
        return None

    port = pick_serial_port(SERIAL_PORT)
    try:
        print("Serial trying:", port)
        ar = serial.Serial(port, SERIAL_BAUD, timeout=0)
        time.sleep(2.0)
        try:
            ar.reset_input_buffer()
        except Exception:
            pass
        return ar
    except Exception as exc:
        print("Serial open failed:", exc)
        return None


_last_serial_fail = 0.0


def safe_serial_write(arduino, text):
    global _last_serial_fail
    if not ENABLE_SERIAL:
        return
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


def is_ocr_glitch(txt):
    if not txt:
        return True
    if txt.count("/") != 1:
        return True
    if len(txt) > 10:
        return True
    return False


def parse_pair_flexible(txt, last_max):
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


def clamp_int(v, low, high):
    if v < low:
        return low
    if v > high:
        return high
    return v


def quantize_permille(permille, step):
    permille = clamp_int(int(permille), 0, 1000)
    return int(round(permille / float(step)) * step)


def clamp_rgb(rgb):
    return (
        clamp_int(int(rgb[0]), 0, 255),
        clamp_int(int(rgb[1]), 0, 255),
        clamp_int(int(rgb[2]), 0, 255),
    )


def quantize_rgb(rgb, step):
    if step <= 1:
        return clamp_rgb(rgb)
    return clamp_rgb(
        (
            int(round(rgb[0] / float(step)) * step),
            int(round(rgb[1] / float(step)) * step),
            int(round(rgb[2] / float(step)) * step),
        )
    )


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


def mana_gradient_from_permille(permille):
    t = 1.0 - (clamp_int(permille, 0, 1000) / 1000.0)

    r0, g0, b0 = MANA_GRADIENT_LOW_RGB
    r1, g1, b1 = MANA_GRADIENT_HIGH_RGB
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def load_crop_config():
    if not CONFIG_PATH.exists():
        return MONITOR_INDEX, MANA_CROP

    try:
        data = json.loads(CONFIG_PATH.read_text())
        mon = int(data.get("monitor_index", MONITOR_INDEX))
        mana = data.get("mana_crop", MANA_CROP)

        for key in ("top", "left", "width", "height"):
            if key not in mana:
                raise ValueError("mana_crop missing keys")

        return mon, mana
    except Exception as exc:
        print("Failed to load crop config, using defaults:", exc)
        return MONITOR_INDEX, MANA_CROP


def main():
    monitor_index, mana_crop = load_crop_config()
    arduino = safe_serial_open()

    good_mana = [0, 0]
    mana_permille = 500
    mana_smooth = 500

    mana_max_candidate = None
    mana_max_hits = 0

    lock_until = 0.0
    bad_since = None

    dead_now = False
    alive_confirm = 0
    revive_blue_until = 0.0

    last_stable_rgb = None
    last_sent_rgb = None
    last_send_time = 0.0

    last_debug_time = 0.0
    last_debug_tuple = None

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            monitor = sct.monitors[monitor_index]

            mana_region = build_region(monitor, mana_crop)
            mana_bgra = np.array(sct.grab(mana_region))
            mana_bw = prep_bar_for_ocr(mana_bgra, MANA_FOCUS_X, MANA_FOCUS_Y)
            mana_txt = ocr_bar_text(mana_bw)

            frame_ok = False
            glitch = False

            if is_ocr_glitch(mana_txt):
                glitch = True
            else:
                parsed = parse_pair_flexible(mana_txt, good_mana[1])
                if parsed is None:
                    glitch = True
                else:
                    cur, mx, strong = parsed

                    cur = repair_dropped_digit(cur, mx, good_mana[0], good_mana[1])

                    if strong and mx > 0 and cur > mx:
                        fixed = repair_extra_digit(cur, mx, good_mana[0])
                        if fixed > mx:
                            glitch = True
                        else:
                            cur = fixed

                    if not glitch:
                        ok, cur, mx, mana_max_candidate, mana_max_hits = accept_reading(
                            cur, mx, good_mana, strong, mana_max_candidate, mana_max_hits
                        )

                        if ok and mx > 0 and 0 <= cur <= mx:
                            good_mana[0] = cur
                            good_mana[1] = mx
                            mana_permille = int(cur * 1000 / mx)
                            frame_ok = True
                        else:
                            glitch = True

            if not frame_ok:
                lock_until = max(lock_until, now + OCR_GLITCH_HOLD_SEC)
                if bad_since is None:
                    bad_since = now
            else:
                bad_since = None

            mana_smooth = int((mana_smooth * 6 + mana_permille * 4) / 10)
            mana_smooth = quantize_permille(mana_smooth, COLOR_STEP_PERMILLE)

            mana_rgb = quantize_rgb(mana_gradient_from_permille(mana_smooth), RGB_STEP)

            out_rgb = mana_rgb
            reason = "mana"

            if now < lock_until:
                out_rgb = last_stable_rgb if last_stable_rgb is not None else FALLBACK_YELLOW_RGB
                reason = "lock"

            if bad_since is not None and (now - bad_since) >= DEAD_DETECT_AFTER_SEC:
                dead_now = True

            if dead_now:
                out_rgb = last_stable_rgb if last_stable_rgb is not None else FALLBACK_YELLOW_RGB
                reason = "dead_hold"

                if frame_ok:
                    alive_confirm += 1
                else:
                    alive_confirm = 0

                if alive_confirm >= ALIVE_FRAMES_REQUIRED:
                    dead_now = False
                    alive_confirm = 0
                    revive_blue_until = now + REVIVE_FORCE_BLUE_SEC

            if not dead_now and reason == "mana":
                last_stable_rgb = out_rgb

            if not dead_now and now < revive_blue_until:
                out_rgb = quantize_rgb(MANA_GRADIENT_LOW_RGB, RGB_STEP)
                reason = "revive_blue"

            if ENABLE_SERIAL and arduino is not None:
                if (now - last_send_time) >= SEND_EVERY_SEC:
                    if out_rgb != last_sent_rgb:
                        send_still_color(arduino, out_rgb)
                        last_sent_rgb = out_rgb
                        last_send_time = now

            if PRINT_OCR_DEBUG:
                lock_left = round(max(0.0, lock_until - now), 2)
                bad_for = round((now - bad_since), 2) if bad_since is not None else 0.0

                debug_tuple = (
                    mana_txt,
                    good_mana[0],
                    good_mana[1],
                    mana_permille,
                    mana_smooth,
                    mana_rgb,
                    glitch,
                    lock_left,
                    bad_for,
                    dead_now,
                    out_rgb,
                    reason,
                    mana_max_candidate,
                    mana_max_hits,
                )

                if (now - last_debug_time) >= DEBUG_PRINT_EVERY_SEC and debug_tuple != last_debug_tuple:
                    print(
                        "MANA:",
                        f"[{mana_txt}]",
                        f"{good_mana[0]}/{good_mana[1]}",
                        "perm",
                        mana_permille,
                        "smooth",
                        mana_smooth,
                        "rgb",
                        mana_rgb,
                        "glitch",
                        glitch,
                        "lock",
                        lock_left,
                        "badFor",
                        bad_for,
                        "deadLike",
                        dead_now,
                        "maxCandidate",
                        mana_max_candidate,
                        "hits",
                        mana_max_hits,
                        "| OUT:",
                        out_rgb,
                        "why",
                        reason,
                    )
                    last_debug_time = now
                    last_debug_tuple = debug_tuple

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Mana Crop RAW", bgra_to_bgr(mana_bgra))
                cv2.imshow("Mana OCR Input", mana_bw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
