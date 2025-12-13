import json
from pathlib import Path

import cv2
import mss
import numpy as np

MONITOR_INDEX = 1
PICK_EVENTS = False
CONFIG_PATH = Path("crops.json")


def grab_monitor(mon_idx):
    with mss.mss() as s:
        mon = s.monitors[mon_idx]
        img = np.array(s.grab(mon))[:, :, :3]
        return img, mon


def pick_box(win_name, img, prompt):
    pts = []
    clone = img.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal pts, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 2:
                cv2.rectangle(clone, pts[0], pts[1], (0, 255, 0), 2)
        if len(pts) <= 1:
            cv2.imshow(win_name, img)
        else:
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, on_mouse)
    print(prompt, "Click top-left then bottom-right, press any key when done.")
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)
    if len(pts) != 2:
        raise RuntimeError("Need two clicks")
    (x1, y1), (x2, y2) = pts
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    return {"top": top, "left": left, "width": width, "height": height}


def load_existing():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return None
    return None


def main():
    img, mon = grab_monitor(MONITOR_INDEX)
    print(f"Monitor {MONITOR_INDEX} bounds: {mon}")

    existing = load_existing() or {}
    events_crop = existing.get("events_crop")
    if PICK_EVENTS or events_crop is None:
        events_crop = pick_box("Pick Events Crop", img, "Select the objective banner area")

    h, w, _ = img.shape
    focus_w = int(w * 0.35)
    focus_h = int(h * 0.15)
    focus_left = int((w - focus_w) / 2)
    focus_top = h - focus_h - 10
    focus = img[focus_top : focus_top + focus_h, focus_left : focus_left + focus_w]
    health_crop_rel = pick_box("Pick Health Crop", focus, "Select your HP bar area (zoomed view)")
    health_crop = {
        "top": health_crop_rel["top"] + focus_top,
        "left": health_crop_rel["left"] + focus_left,
        "width": health_crop_rel["width"],
        "height": health_crop_rel["height"],
    }

    data = {
        "monitor_index": MONITOR_INDEX,
        "events_crop": events_crop,
        "health_crop": health_crop,
    }
    print("EVENTS_CROP =", events_crop)
    print("HEALTH_CROP =", health_crop)
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
        print(f"Saved to {CONFIG_PATH.resolve()}")
    except Exception as exc:
        print("Failed to save config:", exc)


if __name__ == "__main__":
    main()
