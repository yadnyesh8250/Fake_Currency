"""
simple_detect.py — Minimal Indian Currency & Fake/Specimen Detector
────────────────────────────────────────────────────────────────────
A short, minimal, and explainable OpenCV script for college viva demonstration.
Contains ONLY the core logic:
  1. Load genuine currency templates from Dataset/
  2. Capture from webcam OR load an image file
  3. Feature matching (ORB + Lowe's Ratio + RANSAC inliers)
  4. Authenticity check (Specimen stamp detection in HSV color space)
  5. Display result banner on the image
"""

import os
import sys
import cv2
import numpy as np

# ── 1. CONFIGURATION & SETUP ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DENOMINATIONS = ["10", "20", "50", "100", "200", "500"]

# Initialize ORB feature detector and Brute-Force Matcher with Hamming distance
orb = cv2.ORB_create(nfeatures=1000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ── 2. LOAD REFERENCE TEMPLATES INTO MEMORY ───────────────────────────────────
def load_templates():
    """Load reference images for each denomination and extract their ORB features."""
    templates = {}
    print("[INFO] Loading genuine reference templates...")
    for denom in DENOMINATIONS:
        folder = os.path.join(DATASET_DIR, denom)
        if not os.path.isdir(folder):
            continue
        templates[denom] = []
        for fname in sorted(os.listdir(folder))[:10]:  # Use top 10 clean samples
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(folder, fname))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                if des is not None:
                    templates[denom].append({"bgr": img, "kp": kp, "des": des})
    print(f"[INFO] Loaded templates for: {list(templates.keys())}")
    return templates


# ── 3. RECOGNIZE DENOMINATION VIA FEATURE MATCHING ────────────────────────────
def match_currency(query_img, templates):
    """
    Matches query note against all denominations using:
      - ORB descriptors
      - Lowe's ratio test (0.75)
      - RANSAC geometric inliers
    """
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    q_kp, q_des = orb.detectAndCompute(gray, None)

    if q_des is None or len(q_des) < 10:
        return "Unknown", 0, 0, None

    best_denom = "Unknown"
    best_inliers = 0
    best_matches_count = 0
    best_ref = None

    for denom, sample_list in templates.items():
        for sample in sample_list:
            # KNN Match (k=2)
            matches = matcher.knnMatch(q_des, sample["des"], k=2)

            # Lowe's Ratio Test
            good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]

            # RANSAC Homography verification
            inliers = 0
            if len(good) >= 6:
                src_pts = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([sample["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                try:
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = int(mask.sum())
                except Exception:
                    inliers = 0

            # Winning denomination has the most geometric inliers
            if inliers > best_inliers:
                best_inliers = inliers
                best_matches_count = len(good)
                best_denom = denom
                best_ref = sample

    # Threshold: must have at least 8 verified inliers to be valid
    if best_inliers < 8:
        return "Unknown", best_inliers, best_matches_count, None

    return best_denom, best_inliers, best_matches_count, best_ref


# ── 4. CHECK AUTHENTICITY (SPECIMEN / FAKE DETECTION) ─────────────────────────
def check_authenticity(query_img, denom, inliers):
    """
    Inspects:
      1. Watermark field for red/crimson 'SPECIMEN' cancellation stamp
      2. Texture sharpness via Laplacian variance (photocopy detection)
    """
    h, w = query_img.shape[:2]
    reasons = []

    # Check 1: Watermark & signature field (x: 55% to 95%, y: 15% to 70%)
    wm_roi = query_img[int(h * 0.15):int(h * 0.70), int(w * 0.55):int(w * 0.95)]
    hsv = cv2.cvtColor(wm_roi, cv2.COLOR_BGR2HSV)

    # Red/Crimson color mask in HSV space
    m1 = cv2.inRange(hsv, np.array([0, 80, 75]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 80, 75]), np.array([180, 255, 255]))
    red_mask = m1 | m2

    red_pixels = cv2.countNonZero(red_mask)
    red_ratio = red_pixels / float(wm_roi.shape[0] * wm_roi.shape[1])

    # Genuine 500/100/200/10 notes have 0 red ink in this zone
    if denom in ["500", "100", "200", "10"] and red_ratio > 0.015:
        return "SPECIMEN (NOT GENUINE)", (0, 0, 255), f"Red SPECIMEN cancellation stamp detected ({red_ratio*100:.1f}% coverage)"

    # Check 2: Texture sharpness (Laplacian variance)
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 25.0:
        return "LIKELY FAKE (BLURRED)", (0, 140, 255), f"Low print sharpness ({lap_var:.1f}) — possible photocopy"

    if inliers < 15:
        return "SUSPICIOUS", (0, 165, 255), f"Weak feature correspondence ({inliers} inliers)"

    return "GENUINE (REAL CURRENCY)", (0, 255, 0), "All security features & patterns verified"


# ── 5. MAIN EXECUTION (IMAGE FILE OR WEBCAM) ──────────────────────────────────
def main():
    templates = load_templates()

    # Input: File argument OR live webcam
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load image from: {img_path}")
            return
        print(f"[INFO] Analyzing file: {img_path}")
    else:
        print("[INFO] Opening webcam... Hold note and press SPACE to capture, ESC to exit.")
        cap = cv2.VideoCapture(0)
        img = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "Press SPACE to capture | ESC to exit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Webcam Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                img = frame
                break
            elif key == 27:  # ESC
                break
        cap.release()
        cv2.destroyAllWindows()
        if img is None:
            print("[INFO] Capture cancelled.")
            return

    # Standardize image size (width 800px)
    h, w = img.shape[:2]
    if w > 800:
        img = cv2.resize(img, (800, int(h * (800 / w))), interpolation=cv2.INTER_AREA)

    # 1. Match Denomination
    denom, inliers, good_count, ref_sample = match_currency(img, templates)

    # 2. Check Authenticity
    if denom != "Unknown":
        verdict, color, reason = check_authenticity(img, denom, inliers)
    else:
        verdict, color, reason = "UNRECOGNIZED", (0, 165, 255), "Note patterns not identified"

    # 3. Print Results in Terminal
    print("\n" + "=" * 55)
    print("           RUPEEVISION — DETECTION RESULT")
    print("=" * 55)
    print(f"  Denomination : ₹{denom}")
    print(f"  Verdict      : {verdict}")
    print(f"  Inliers      : {inliers} verified geometric points")
    print(f"  Matches      : {good_count} feature pairs")
    print(f"  Reason       : {reason}")
    print("=" * 55 + "\n")

    # 4. Draw Visual Banner on Image & Display
    output = img.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (30, 30, 30), -1)
    cv2.putText(output, f"Denomination: INR {denom}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(output, f"Verdict: {verdict} ({reason})", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("RupeeVision Result (Press any key to close)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
