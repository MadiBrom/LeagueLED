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

HEALTH_CROP = {"top": 984, "left": 775, "width": 280, "height": 27}
MANA_CROP = {"top": 996, "left": 775, "width": 280, "height": 27}
MANA_TOP_OFFSET = int(MANA_CROP["top"] - HEALTH_CROP["top"])

USE_THIN_DIGITS = True
THIN_ITERATIONS = 1

HSV_WHITE_LOW = (0, 0, 160)
HSV_WHITE_HIGH = (179, 80, 255)

SWAP_GB_ON_SEND = True

OCR_GLITCH_HOLD_SEC = 0.40
OCR_FALLBACK_YELLOW_AFTER_SEC = 1.20
FALLBACK_YELLOW_RGB = (255, 255, 0)

COLOR_STEP_PERMILLE = 25
RGB_STEP = 5

CMD_STILL = 10

MANA_GRADIENT_LOW_RGB = (20, 40, 255)
MANA_GRADIENT_HIGH_RGB = (170, 30, 200)

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not Path(TESSERACT_EXE).exists():
    raise RuntimeError("Tesseract exe not found at: " + TESSERACT_EXE)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
print("tesseract_cmd:", pytesseract.pytesseract.tesseract_cmd)


def derive_mana_crop(health_crop):
    return {
        "top": max(0, int(health_crop["top"]) + MANA_TOP_OFFSET),
        "left": int(health_crop["left"]),
        "width": int(health_crop["width"]),
        "height": int(health_crop["height"]),
    }


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
    if not ENABLE_SERIAL:
        return None
    if serial is None:
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


def parse_pair(txt, last_max):
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


def mana_gradient_from_permille(permille):
    t = clamp_int(permille, 0, 1000) / 1000.0
    r0, g0, b0 = MANA_GRADIENT_LOW_RGB
    r1, g1, b1 = MANA_GRADIENT_HIGH_RGB
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def load_crop_config():
    if not CONFIG_PATH.exists():
        return MONITOR_INDEX, HEALTH_CROP, MANA_CROP

    try:
        data = json.loads(CONFIG_PATH.read_text())
        mon = int(data.get("monitor_index", MONITOR_INDEX))
        health = data.get("health_crop", HEALTH_CROP)

        mana = data.get("mana_crop", None)
        if mana is None:
            mana = derive_mana_crop(health)

        for crop in (health, mana):
            for key in ("top", "left", "width", "height"):
                if key not in crop:
                    raise ValueError("crop missing keys")

        return mon, health, mana
    except Exception as exc:
        print("Failed to load crop config, using defaults:", exc)
        return MONITOR_INDEX, HEALTH_CROP, MANA_CROP


def main():
    monitor_index, _health_crop, mana_crop = load_crop_config()
    arduino = safe_serial_open()

    good_pair = [0, 0]
    good_permille = 500
    smooth_permille = 500

    lock_until = 0.0
    bad_since = None

    last_output_rgb = None
    last_sent_rgb = None

    last_debug_time = 0.0
    last_debug_tuple = None

    with mss.mss() as sct:
        while True:
            now = time.monotonic()
            monitor = sct.monitors[monitor_index]

            region = build_region(monitor, mana_crop)
            bgra = np.array(sct.grab(region))
            bw = prep_bar_for_ocr(bgra, MANA_FOCUS_X, MANA_FOCUS_Y)
            txt = ocr_bar_text(bw)

            frame_ok = False
            glitch = False

            if is_ocr_glitch(txt):
                glitch = True
            else:
                parsed = parse_pair(txt, good_pair[1])
                if parsed is None:
                    glitch = True
                else:
                    cur, mx, _strong = parsed
                    if mx > 0:
                        cur = min(cur, mx)
                        good_pair[0] = cur
                        good_pair[1] = mx
                        good_permille = int(cur * 1000 / mx)
                        frame_ok = True
                    else:
                        glitch = True

            if not frame_ok:
                lock_until = max(lock_until, now + OCR_GLITCH_HOLD_SEC)
                if bad_since is None:
                    bad_since = now
            else:
                bad_since = None

            smooth_permille = int((smooth_permille * 6 + good_permille * 4) / 10)
            smooth_permille = quantize_permille(smooth_permille, COLOR_STEP_PERMILLE)

            rgb = mana_gradient_from_permille(smooth_permille)
            rgb = quantize_rgb(rgb, RGB_STEP)

            any_lock = now < lock_until
            long_bad = (bad_since is not None) and ((now - bad_since) >= OCR_FALLBACK_YELLOW_AFTER_SEC)

            out_rgb = rgb
            reason = "mana"

            if any_lock:
                out_rgb = last_output_rgb if last_output_rgb is not None else FALLBACK_YELLOW_RGB
                reason = "lock"

            if long_bad:
                out_rgb = FALLBACK_YELLOW_RGB
                reason = "fallback"

            if reason == "mana":
                last_output_rgb = out_rgb

            if ENABLE_SERIAL and arduino is not None:
                if out_rgb != last_sent_rgb:
                    send_color_cmd(arduino, CMD_STILL, out_rgb)
                    last_sent_rgb = out_rgb

            if PRINT_OCR_DEBUG:
                lock_left = round(max(0.0, lock_until - now), 2)
                debug_tuple = (txt, good_pair[0], good_pair[1], good_permille, smooth_permille, rgb, glitch, lock_left, out_rgb, reason)

                if (now - last_debug_time) >= DEBUG_PRINT_EVERY_SEC and debug_tuple != last_debug_tuple:
                    print(
                        "MANA:",
                        f"[{txt}]",
                        f"{good_pair[0]}/{good_pair[1]}",
                        "perm",
                        good_permille,
                        "smooth",
                        smooth_permille,
                        "rgb",
                        rgb,
                        "glitch",
                        glitch,
                        "lock",
                        lock_left,
                        "| OUT:",
                        out_rgb,
                        "why",
                        reason,
                    )
                    last_debug_time = now
                    last_debug_tuple = debug_tuple

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Mana Crop RAW", bgra_to_bgr(bgra))
                cv2.imshow("Mana OCR Input", bw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
