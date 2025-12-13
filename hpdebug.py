import cv2
import numpy as np
import mss
import pytesseract
import re

def make_tesseract_path():
    dash = chr(45)
    base = r"C:\Program Files\Tesseract"
    tail = r"OCR\tesseract.exe"
    return base + dash + tail


pytesseract.pytesseract.tesseract_cmd = make_tesseract_path()

healthCrop = {"top": 984, "left": 775, "width": 280, "height": 27}
HEALTH_FOCUS_X = (0.40, 0.62)
HEALTH_FOCUS_Y = (0.32, 0.78)
TEXT_ONLY_MAX_X = 0.65

def extract_hp_from_text(text):
    """
    Extracts HP numbers from OCR text like '1234/5678'
    Returns (current_hp, max_hp) or (None, None) if parsing fails.
    """
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))

with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        region = {
            "mon": 1,
            "top": monitor["top"] + healthCrop["top"],
            "left": monitor["left"] + healthCrop["left"],
            "width": healthCrop["width"],
            "height": healthCrop["height"]
        }

        img = np.array(sct.grab(region))
        h, w, _ = img.shape
        x1 = int(w * HEALTH_FOCUS_X[0])
        x2 = int(w * HEALTH_FOCUS_X[1])
        y1 = int(h * HEALTH_FOCUS_Y[0])
        y2 = int(h * HEALTH_FOCUS_Y[1])
        focus = img[y1:y2, x1:x2]
        fx = int(focus.shape[1] * TEXT_ONLY_MAX_X)
        focus = focus[:, :fx]

        gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789/"
        )

        current_hp, max_hp = extract_hp_from_text(text)

        if current_hp is not None and max_hp is not None and max_hp > 0:
            hp_percent = current_hp / max_hp
            print(f"HP: {current_hp}/{max_hp}  ({hp_percent*100:.1f}%)")
        else:
            print("OCR failed:", text.strip())

        cv2.imshow("Health Crop Debug", img)
        cv2.imshow("OCR Input", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
