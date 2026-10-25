"""
demo_detect.py — RupeeVision: Indian Currency Recognition & IVP Verification
==============================================================================
A clean, minimal, and explainable Computer Vision script designed for college
Image & Video Processing (IVP) project demonstrations and viva examinations.

CORE TECHNIQUES USED:
  1. Image Preprocessing: Grayscale, Gaussian Blur (5x5), CLAHE Contrast Enhancement
  2. Feature Extraction: ORB (Oriented FAST & Rotated BRIEF) Invariant Keypoints
  3. Feature Correspondence: Brute-Force Matcher + Lowe's Ratio Test (0.75)
  4. Geometric Verification: RANSAC Homography Matrix Estimation (Inlier Counting)
  5. Chromaticity Validation: HSV Hue-Channel Histogram Correlation
  6. Authenticity Inspection:
     - Print sharpness via Laplacian variance (Photocopy / Blur detection)
     - Watermark security zone inspection (Red SPECIMEN / Cancellation stamp detection)

USAGE:
  1. Analyze an image file:
       ../venv/bin/python demo_detect.py Dataset/20/18.jpg
  2. Live webcam capture:
       ../venv/bin/python demo_detect.py
       (Hold note in view, press SPACE to analyze, ESC to exit)
"""

import os
import sys
import time
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. DIRECTORY CONFIGURATION & SUPPORTED DENOMINATIONS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DENOMINATIONS = ["10", "20", "50", "100", "200", "500"]

# Known corrupted, misfiled, or infographic files in dataset to exclude
EXCLUDED_FILES = {
    "10": {"0.jpg", "64.jpg", "106.jpg", "107.jpg"},
    "20": {"4.jpg", "37.jpg", "73.jpg", "28.jpg"},
    "50": {"44.jpg", "66.jpg", "84.jpg"},
    "100": {"8.jpg"},
    "200": {"7.jpg", "33.jpg", "56.jpg"},
    "500": {"13.jpg", "34.jpg", "42.jpg", "63.jpg", "84.jpg"}
}

# Curated reference templates (Front + Back, New & Old series)
CURATED_TEMPLATES = {
    "10": ["9.jpg", "20.jpg", "22.jpg", "33.jpg", "39.jpg", "45.jpg", "102.jpg"],
    "20": ["18.jpg", "48.jpg", "60.jpg", "69.jpg", "81.jpg", "8.jpg", "49.jpg", "66.jpg", "35.jpg", "51.jpg"],
    "50": ["10.jpg", "11.jpg", "13.jpg", "49.jpg", "70.jpg", "14.jpg", "18.jpg", "27.jpg", "30.jpg"],
    "100": ["0.jpg", "10.jpg", "11.jpg", "14.jpg", "17.jpg", "19.jpg", "23.jpg", "47.jpg", "80.jpg"],
    "200": ["11.jpg", "19.jpg", "20.jpg", "42.jpg", "43.jpg", "50.jpg", "75.jpg", "87.jpg", "89.jpg"],
    "500": ["0.jpg", "10.jpg", "11.jpg", "100.jpg", "101.jpg", "102.jpg", "27.jpg", "36.jpg", "51.jpg"]
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE DETECTOR & MATCHER INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────
# ORB: 1000 features, scaleFactor 1.2, 8 pyramid levels for scale invariance
orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

# Brute-Force Matcher with Hamming distance (fast bitwise popcount for binary descriptors)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ──────────────────────────────────────────────────────────────────────────────
# 3. COLOR & HISTOGRAM UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def compute_hsv_histogram(bgr_img):
    """
    Computes normalized Hue histogram (36 bins) in HSV color space.
    Masks out extreme bright highlights (glare) and dark shadows/backgrounds.
    """
    try:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([180, 255, 245]))
        hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        return hist
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4. REFERENCE TEMPLATE CACHE LOADER
# ──────────────────────────────────────────────────────────────────────────────
def load_reference_templates():
    """
    Loads verified genuine currency templates from the dataset into RAM.
    Standardizes each template to 600px width and pre-computes ORB features
    and HSV color histograms for instantaneous sub-second matching.
    """
    print("[INFO] Loading genuine reference currency templates...")
    templates = {}
    total_loaded = 0

    for denom in DENOMINATIONS:
        templates[denom] = []
        folder = os.path.join(DATASET_DIR, denom)
        if not os.path.isdir(folder):
            continue

        # Priority 1: Use curated list of verified genuine templates
        candidate_files = CURATED_TEMPLATES.get(denom, [])
        # Priority 2: Fall back to other valid images in folder if curated missing
        all_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in all_files:
            if f not in candidate_files and f not in EXCLUDED_FILES.get(denom, set()):
                candidate_files.append(f)

        for fname in candidate_files:
            if fname in EXCLUDED_FILES.get(denom, set()):
                continue
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            h, w = img.shape[:2]
            ar = max(w, h) / float(min(w, h) + 1e-5)
            # Accept horizontal banknote geometry
            if (1.5 <= ar <= 2.8 and min(w, h) >= 120) or (fname in CURATED_TEMPLATES.get(denom, [])):
                # Standardize reference width to 600px
                target_w = 600
                scale = target_w / float(w)
                resized = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)

                if des is not None and len(des) >= 20:
                    templates[denom].append({
                        "filename": fname,
                        "bgr": resized,
                        "gray": gray,
                        "kp": kp,
                        "des": des,
                        "hist": compute_hsv_histogram(resized)
                    })
                    total_loaded += 1
                    # Limit to top 12 clean templates per denomination
                    if len(templates[denom]) >= 12:
                        break

    print(f"[INFO] Successfully loaded {total_loaded} genuine reference templates across {len(templates)} denominations.\n")
    return templates


# ──────────────────────────────────────────────────────────────────────────────
# 5. IMAGE PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_input(img):
    """
    Classical IVP Preprocessing:
      1. Bounded resizing (max width 800px)
      2. Orientation rectification (horizontal alignment)
      3. Grayscale conversion
      4. Gaussian smoothing (5x5 kernel)
      5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    h, w = img.shape[:2]

    # Step 1: Rotate to horizontal if photographed in portrait orientation
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    # Step 2: Scale down large camera frames to max 800px width
    TARGET_WIDTH = 800
    if w > TARGET_WIDTH:
        scale = TARGET_WIDTH / float(w)
        resized_bgr = cv2.resize(img, (TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized_bgr = img.copy()

    # Step 3: Grayscale conversion (eliminates illumination bias)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # Step 4: Gaussian smoothing to suppress sensor noise & grain
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: CLAHE local contrast enhancement (amplifies watermark & micro-print)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(blurred)

    return resized_bgr, enhanced_gray


# ──────────────────────────────────────────────────────────────────────────────
# 6. DUAL-FACTOR DENOMINATION RECOGNITION (ORB + RANSAC + HSV COLOR)
# ──────────────────────────────────────────────────────────────────────────────
def recognize_denomination(query_bgr, query_gray, templates):
    """
    Multi-modal identification:
      1. Extracts ORB features from query note.
      2. Matches descriptors against reference templates via k-NN (k=2).
      3. Filters matches using Lowe's ratio test (m.distance < 0.75 * n.distance).
      4. Verifies geometric consistency with RANSAC homography estimation.
      5. Computes HSV color correlation against reference note.
      6. Combines spatial inliers and color correlation into a unified confidence score:
           Score = inliers * (0.35 + 0.65 * color_correlation)
    """
    q_kp, q_des = orb.detectAndCompute(query_gray, None)
    if q_des is None or len(q_des) < 10:
        return "Unknown", 0, 0, 0.0, None, [], None

    q_hist = compute_hsv_histogram(query_bgr)

    best_denom = "Unknown"
    best_score = -1.0
    best_inliers = 0
    best_good_matches = 0
    best_color_corr = 0.0
    best_ref_sample = None
    best_matches = []
    denom_scores = {}

    for denom, sample_list in templates.items():
        top_denom_inliers = 0
        top_denom_good = 0
        top_denom_score = -1.0
        top_denom_sample = None
        top_denom_matches = []
        top_denom_color = 0.0

        for sample in sample_list:
            if sample["des"] is None or len(sample["des"]) < 5:
                continue

            try:
                # k-NN match with k=2
                knn_matches = matcher.knnMatch(q_des, sample["des"], k=2)
            except Exception:
                continue

            # Lowe's Ratio Test
            good = [m for m, n in knn_matches if len(knn_matches) > 0 and m.distance < 0.75 * n.distance]

            # Geometric RANSAC verification
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

            # Chromaticity correlation in HSV
            color_corr = 0.50
            if q_hist is not None and sample["hist"] is not None:
                corr = cv2.compareHist(q_hist, sample["hist"], cv2.HISTCMP_CORREL)
                color_corr = max(0.0, float(corr))

            # Multi-cue score: spatial inliers weighted by color agreement
            score = (inliers * 1.0) * (0.35 + 0.65 * color_corr)

            if score > top_denom_score or (score == top_denom_score and inliers > top_denom_inliers):
                top_denom_score = score
                top_denom_inliers = inliers
                top_denom_good = len(good)
                top_denom_sample = sample
                top_denom_matches = good
                top_denom_color = color_corr

        denom_scores[denom] = {
            "score": round(top_denom_score, 1),
            "inliers": top_denom_inliers,
            "good_matches": top_denom_good,
            "color_corr": round(top_denom_color, 2)
        }

        if top_denom_score > best_score:
            best_score = top_denom_score
            best_denom = denom
            best_inliers = top_denom_inliers
            best_good_matches = top_denom_good
            best_color_corr = top_denom_color
            best_ref_sample = top_denom_sample
            best_matches = top_denom_matches

    # Threshold for positive identification: minimum 7 RANSAC inliers
    if best_inliers < 7:
        best_denom = "Unknown"

    return best_denom, best_inliers, best_good_matches, best_color_corr, best_ref_sample, best_matches, denom_scores


# ──────────────────────────────────────────────────────────────────────────────
# 7. MULTI-FACTOR AUTHENTICITY INSPECTION
# ──────────────────────────────────────────────────────────────────────────────
def inspect_authenticity(query_bgr, query_gray, denom, inliers, color_corr):
    """
    Evaluates physical banknote security characteristics:
      1. Saturated Red / Crimson SPECIMEN / CANCELLED stamp detection in watermark zone
      2. Intaglio print texture sharpness via Laplacian variance (detects smooth photocopies)
      3. Geometric correspondence density
    """
    h, w = query_bgr.shape[:2]
    anomalies = []
    is_specimen = False

    # Check 1: Watermark & Signature Security Zone (Right 55% - 94% width, 14% - 72% height)
    rx1, rx2 = int(w * 0.55), int(w * 0.94)
    ry1, ry2 = int(h * 0.14), int(h * 0.72)
    wm_roi = query_bgr[ry1:ry2, rx1:rx2]

    if wm_roi.size > 0:
        hsv_wm = cv2.cvtColor(wm_roi, cv2.COLOR_BGR2HSV)
        # Saturated crimson red mask (covers specimen / cancelled ink stamps)
        m1 = cv2.inRange(hsv_wm, np.array([0, 75, 65]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv_wm, np.array([160, 75, 65]), np.array([180, 255, 255]))
        stamp_mask = m1 | m2
        red_count = cv2.countNonZero(stamp_mask)
        red_ratio = red_count / float(wm_roi.shape[0] * wm_roi.shape[1])
        contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cluster = max([cv2.contourArea(c) for c in contours]) if contours else 0

        # Genuine ₹500 is stone-grey: saturated red ink in watermark indicates specimen/cancelled
        if denom == "500":
            if red_ratio > 0.02 and max_cluster > 300:
                is_specimen = True
                anomalies.append(f"Red SPECIMEN / cancellation stamp detected ({red_ratio*100:.1f}% coverage)")
        else:
            # On colored notes, require heavy crimson cluster to distinguish from background edge
            if red_ratio > 0.08 and max_cluster > 1200:
                is_specimen = True
                anomalies.append(f"Defacement / stamp detected ({red_ratio*100:.1f}% coverage)")

    # Check 2: Texture Sharpness via Laplacian Variance
    lap_var = float(cv2.Laplacian(query_gray, cv2.CV_64F).var())

    if is_specimen:
        verdict = "SPECIMEN (NOT LEGAL TENDER)"
        color = (0, 0, 255)  # Red
        reason = "Defaced with official SPECIMEN / CANCELLED cancellation stamp"
    elif lap_var < 10.0:
        verdict = "LIKELY FAKE / BLURRY"
        color = (0, 140, 255)  # Orange
        reason = f"Print sharpness abnormally low ({lap_var:.1f}) — possible flat photocopy"
    elif inliers >= 15:
        verdict = "GENUINE (REAL CURRENCY)"
        color = (0, 255, 0)  # Green
        reason = f"High structural correspondence ({inliers} inliers, sharpness {lap_var:.1f})"
    elif inliers >= 7:
        verdict = "GENUINE (VERIFIED)"
        color = (0, 255, 0)  # Green
        reason = f"Verified feature patterns ({inliers} inliers, color match {color_corr:.2f})"
    else:
        verdict = "SUSPICIOUS / UNVERIFIED"
        color = (0, 165, 255)  # Orange
        reason = f"Weak keypoint correspondences ({inliers} inliers)"

    return verdict, color, reason, lap_var


# ──────────────────────────────────────────────────────────────────────────────
# 8. VISUAL DEMONSTRATION & BANNER OVERLAY
# ──────────────────────────────────────────────────────────────────────────────
def create_demo_display(query_bgr, ref_sample, query_kp, best_matches, denom, verdict, color, reason, inliers, lap_var):
    """
    Constructs a dual-pane engineering demonstration window:
      - Left pane: Query banknote
      - Right pane: Matched genuine reference banknote
      - Top: Unified inspection header with Denomination, Verdict, Inliers, and Detail
    """
    qh, qw = query_bgr.shape[:2]

    # Draw feature match correspondences side-by-side if reference match exists
    if ref_sample is not None and len(best_matches) > 0:
        draw_count = min(len(best_matches), 35)
        sorted_matches = sorted(best_matches, key=lambda m: m.distance)[:draw_count]

        base_vis = cv2.drawMatches(
            query_bgr, query_kp,
            ref_sample["bgr"], ref_sample["kp"],
            sorted_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(100, 100, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    else:
        base_vis = query_bgr.copy()

    # Prepend an 80px high top banner bar above the image
    vw = base_vis.shape[1]
    banner = np.zeros((80, vw, 3), dtype=np.uint8)
    banner[:] = (20, 22, 26)  # Dark charcoal header background

    # Left side: Denomination & Verdict
    cv2.putText(banner, f"RUPEEVISION: INR {denom}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, f"Verdict: {verdict}", (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

    # Right side: Inliers, Sharpness, and Verification detail
    stats_text = f"RANSAC Inliers: {inliers} pts  |  Print Sharpness: {lap_var:.1f}  |  {reason}"
    cv2.putText(banner, stats_text, (max(16, vw - 750), 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 200, 210), 1, cv2.LINE_AA)

    # Accent color separator line
    cv2.line(banner, (0, 78), (vw, 78), color, 2)

    # Vertically stack banner above visual matches
    combined = np.vstack([banner, base_vis])
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN DEMONSTRATION WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  RupeeVision — Indian Banknote Recognition & IVP Inspection")
    print("  College Mini-Project Demonstration Console")
    print("=" * 65 + "\n")

    # Load genuine templates
    templates = load_reference_templates()

    # Determine input mode: File path argument OR Live Webcam
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            print(f"[ERROR] Could not load image from path: {img_path}")
            return
        print(f"[INFO] Processing image file: {img_path}")
    else:
        print("[INFO] No file specified. Opening webcam...")
        print("[INSTRUCTION] Hold banknote in front of camera.")
        print("[INSTRUCTION] Press SPACE to capture & analyze | Press ESC to exit.\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not access system webcam.")
            return

        raw_bgr = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw viewfinder overlay guide
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (int(w * 0.15), int(h * 0.20)), (int(w * 0.85), int(h * 0.80)), (0, 255, 120), 2)
            cv2.putText(frame, "HOLD BANKNOTE INSIDE BOX — PRESS SPACE TO CAPTURE", (int(w * 0.15), int(h * 0.16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

            cv2.imshow("RupeeVision — Webcam Viewfinder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                raw_bgr = frame
                break
            elif key == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

        if raw_bgr is None:
            print("[INFO] Demonstration closed.")
            return

    # Measure execution latency
    t_start = time.time()

    # Step 1: Preprocess input image
    processed_bgr, enhanced_gray = preprocess_input(raw_bgr)

    # Step 2: Extract query keypoints
    q_kp, _ = orb.detectAndCompute(enhanced_gray, None)

    # Step 3: Match Denomination
    denom, inliers, good_count, color_corr, ref_sample, best_matches, scores = recognize_denomination(
        processed_bgr, enhanced_gray, templates
    )

    # Step 4: Inspect Authenticity
    if denom != "Unknown":
        verdict, color, reason, lap_var = inspect_authenticity(
            processed_bgr, enhanced_gray, denom, inliers, color_corr
        )
    else:
        verdict, color, reason, lap_var = "UNRECOGNIZED", (0, 165, 255), "No distinct banknote patterns found", 0.0

    t_elapsed = (time.time() - t_start) * 1000.0

    # Step 5: Formatted Terminal Viva Report
    print("-" * 65)
    print("                    ANALYSIS & VIVA REPORT")
    print("-" * 65)
    print(f"  Identified Denomination : INR {denom}")
    print(f"  Authenticity Verdict    : {verdict}")
    print(f"  Verification Detail     : {reason}")
    print(f"  RANSAC Inliers          : {inliers} points")
    print(f"  Lowe's Matches          : {good_count} pairs")
    print(f"  Color Profile Corr      : {color_corr:.2f}")
    print(f"  Texture Sharpness       : {lap_var:.1f}")
    print(f"  Pipeline Latency        : {t_elapsed:.1f} ms")
    print("-" * 65)
    for d, s in scores.items():
        marker = " <== [WINNER]" if d == denom else ""
        print(f"    INR {d:3s} -> Inliers: {s['inliers']:3d} | Color: {s['color_corr']:.2f} | Score: {s['score']:5.1f}{marker}")
    print("=" * 65 + "\n")

    # Step 6: Render visual result window
    display_img = create_demo_display(
        processed_bgr, ref_sample, q_kp, best_matches,
        denom, verdict, color, reason, inliers, lap_var
    )

    # Save output demo image to artifact directory as well
    out_save_path = os.path.join(BASE_DIR, "demo_result_output.jpg")
    cv2.imwrite(out_save_path, display_img)
    print(f"[INFO] Saved demonstration result snapshot to: {out_save_path}")

    # Display interactive OpenCV window (if GUI display environment available)
    try:
        cv2.imshow("RupeeVision — Inspection Demo (Press any key to close)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("[INFO] GUI window not available; check demo_result_output.jpg for visualization.")


if __name__ == "__main__":
    main()
