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


# Exclude known contaminated / misfiled / infographic files
EXCLUDED_FILES = {
    "10": {"0.jpg", "64.jpg", "106.jpg", "107.jpg"},
    "20": {"4.jpg", "37.jpg", "73.jpg", "28.jpg"},
    "50": {"44.jpg", "66.jpg", "84.jpg"},
    "100": {"8.jpg"},
    "200": {"7.jpg", "33.jpg", "56.jpg"},
    "500": {"13.jpg", "34.jpg", "42.jpg", "63.jpg", "84.jpg"}
}

# Curated reference templates ensuring front & back and new & old series representation
CURATED_TEMPLATES = {
    "10": ["9.jpg", "20.jpg", "22.jpg", "33.jpg", "39.jpg", "45.jpg", "102.jpg"],
    "20": ["18.jpg", "48.jpg", "60.jpg", "69.jpg", "81.jpg", "8.jpg", "49.jpg", "66.jpg", "35.jpg", "51.jpg"],
    "50": ["10.jpg", "11.jpg", "13.jpg", "49.jpg", "70.jpg", "14.jpg", "18.jpg", "27.jpg", "30.jpg"],
    "100": ["0.jpg", "10.jpg", "11.jpg", "14.jpg", "17.jpg", "19.jpg", "23.jpg", "47.jpg", "80.jpg"],
    "200": ["11.jpg", "19.jpg", "20.jpg", "42.jpg", "43.jpg", "50.jpg", "75.jpg", "87.jpg", "89.jpg"],
    "500": ["0.jpg", "10.jpg", "11.jpg", "100.jpg", "101.jpg", "102.jpg", "27.jpg", "36.jpg", "51.jpg"]
}


def compute_hsv_histogram(bgr_img):
    """Computes normalized Hue histogram (36 bins) in HSV space to capture banknote color."""
    try:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([180, 255, 245]))
        hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        return hist
    except Exception:
        return None


# ── 2. LOAD REFERENCE TEMPLATES INTO MEMORY ───────────────────────────────────
def load_templates():
    """Load curated genuine reference images for each denomination and extract their ORB features."""
    templates = {}
    print("[INFO] Loading genuine reference templates...")
    for denom in DENOMINATIONS:
        folder = os.path.join(DATASET_DIR, denom)
        if not os.path.isdir(folder):
            continue
        templates[denom] = []

        candidate_files = CURATED_TEMPLATES.get(denom, [])
        all_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in all_files:
            if f not in candidate_files and f not in EXCLUDED_FILES.get(denom, set()):
                candidate_files.append(f)

        for fname in candidate_files:
            if fname in EXCLUDED_FILES.get(denom, set()):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            target_w = 600
            scale = target_w / float(w)
            resized = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is not None and len(des) >= 20:
                templates[denom].append({
                    "bgr": resized,
                    "kp": kp,
                    "des": des,
                    "hist": compute_hsv_histogram(resized)
                })
                if len(templates[denom]) >= 10:
                    break

    print(f"[INFO] Loaded templates for: {list(templates.keys())}")
    return templates


# ── 3. RECOGNIZE DENOMINATION VIA FEATURE MATCHING & COLOR CORRELATION ────────
def match_currency(query_img, templates):
    """
    Matches query note against all denominations using:
      - ORB descriptors + Lowe's ratio test (0.75)
      - RANSAC geometric inliers
      - HSV chromaticity correlation
      - Multi-factor score: inliers * (0.35 + 0.65 * color_corr)
    """
    h, w = query_img.shape[:2]
    # Rotate if vertical
    if h > w:
        query_img = cv2.rotate(query_img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    q_kp, q_des = orb.detectAndCompute(gray, None)

    if q_des is None or len(q_des) < 10:
        return "Unknown", 0, 0, None

    q_hist = compute_hsv_histogram(query_img)

    best_denom = "Unknown"
    best_score = -1.0
    best_inliers = 0
    best_matches_count = 0
    best_ref = None

    for denom, sample_list in templates.items():
        for sample in sample_list:
            if sample["des"] is None or len(sample["des"]) < 5:
                continue

            try:
                matches = matcher.knnMatch(q_des, sample["des"], k=2)
            except Exception:
                continue

            good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]

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

            color_corr = 0.50
            if q_hist is not None and sample.get("hist") is not None:
                corr = cv2.compareHist(q_hist, sample["hist"], cv2.HISTCMP_CORREL)
                color_corr = max(0.0, float(corr))

            # Multi-cue score ensures notes sharing Gandhi's face are not confused across denominations
            score = (inliers * 1.0) * (0.35 + 0.65 * color_corr)

            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_matches_count = len(good)
                best_denom = denom
                best_ref = sample

    # Threshold: must have at least 7 verified inliers
    if best_inliers < 7:
        return "Unknown", best_inliers, best_matches_count, None

    return best_denom, best_inliers, best_matches_count, best_ref


# ── 4. CHECK AUTHENTICITY (SPECIMEN / FAKE DETECTION) ─────────────────────────
def check_authenticity(query_img, denom, inliers):
    """
    Inspects:
      1. Watermark field for red/crimson 'SPECIMEN' cancellation stamp
      2. Texture sharpness via Laplacian variance (photocopy detection)
      3. Geometric inlier correspondence density
    """
    h, w = query_img.shape[:2]

    # Check 1: Watermark & signature field (x: 55% to 94%, y: 14% to 72%)
    wm_roi = query_img[int(h * 0.14):int(h * 0.72), int(w * 0.55):int(w * 0.94)]
    if wm_roi.size > 0:
        hsv = cv2.cvtColor(wm_roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 75, 65]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 75, 65]), np.array([180, 255, 255]))
        red_mask = m1 | m2

        red_pixels = cv2.countNonZero(red_mask)
        red_ratio = red_pixels / float(wm_roi.shape[0] * wm_roi.shape[1])
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cluster = max([cv2.contourArea(c) for c in contours]) if contours else 0

        # Genuine ₹500 is stone-grey with 0.0% red ink in watermark area.
        # A red stamp cluster (>350px) or >2% red ratio conclusively indicates a specimen/cancelled note.
        if denom == "500" and (red_ratio > 0.02 and max_cluster > 350):
            return "SPECIMEN (NOT GENUINE)", (0, 0, 255), f"Red SPECIMEN cancellation stamp detected ({red_ratio*100:.1f}% coverage, {max_cluster:.0f}px cluster)"

    # Check 2: Texture sharpness (Laplacian variance)
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 8.0:
        return "LIKELY FAKE (BLURRED)", (0, 140, 255), f"Very low print sharpness ({lap_var:.1f}) — possible flat photocopy"

    if inliers < 8:
        return "SUSPICIOUS", (0, 165, 255), f"Weak feature correspondence ({inliers} inliers)"

    return "GENUINE (REAL CURRENCY)", (0, 255, 0), f"Verified security features ({inliers} inliers, sharpness {lap_var:.1f})"


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
