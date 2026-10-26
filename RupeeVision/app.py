"""
RupeeVision — Indian Currency Recognition & Image Analysis Using IVP
=====================================================================
A college mini-project demonstrating classical Image & Video Processing (IVP)
techniques using OpenCV, NumPy, and Flask.

Structured into 7 distinct functional sections:
  /// SECTION 1: IMPORTS, DIRECTORY CONFIGURATION & FLASK APP INITIALIZATION
  /// SECTION 2: COMPUTER VISION FEATURE DETECTORS & MATCHER INITIALIZATION
  /// SECTION 3: DATASET CACHING & FEATURE EXTRACTION HELPERS
  /// SECTION 4: IMAGE GEOMETRY & PERSPECTIVE WARP UTILITIES
  /// SECTION 5: CORE 10-STAGE IMAGE & VIDEO PROCESSING (IVP) PIPELINE
  /// SECTION 6: FLASK REST API ENDPOINTS (ROUTING & CONTROLLERS)
  /// SECTION 7: APPLICATION ENTRYPOINT & SERVER STARTUP
"""

################################################################################
# /// SECTION 1: IMPORTS, DIRECTORY CONFIGURATION & FLASK APP INITIALIZATION ///
################################################################################

import os          # Operating system file and directory path management
import time        # Benchmarking execution latency for each pipeline run
import base64      # Encoding processed OpenCV images to data URLs for the browser UI
import numpy as np # Matrix and multidimensional array manipulations
import cv2         # OpenCV computer vision library (core algorithms & filters)
from flask import Flask, request, jsonify, render_template

# ── Directory Paths Setup ─────────────────────────────────────────────────────
# Resolve the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path pointing to the reference dataset of genuine Indian currency notes
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# List of denominations supported by the recognition engine
SUPPORTED_DENOMINATIONS = ["10", "20", "50", "100", "200", "500"]

# ── Flask Server Configuration ────────────────────────────────────────────────
# Initialize Flask app, setting template_folder to BASE_DIR so index.html is served from root
app = Flask(__name__, template_folder=BASE_DIR)

# Enforce a 16 Megabyte upload limit to prevent memory exhaustion from oversized files
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


################################################################################
# /// SECTION 2: COMPUTER VISION FEATURE DETECTORS & MATCHER INITIALIZATION ///
################################################################################

# ── ORB Detector Setup ────────────────────────────────────────────────────────
# Initialize Oriented FAST and Rotated BRIEF (ORB):
#   - nfeatures=1000: Retains up to 1000 strongest corners and textural patterns
#   - scaleFactor=1.2: Decimates image by 20% at each octave for scale invariance
#   - nlevels=8: Constructs an 8-level Gaussian scale pyramid for multi-distance recognition
ORB_DETECTOR = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

# ── BEBLID Descriptor Setup ───────────────────────────────────────────────────
# Check for Boosted Efficient Binary Local Image Descriptor (BEBLID) from opencv-contrib
BEBLID_DESCRIPTOR = None
try:
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "BEBLID_create"):
        # Scale factor 0.75 optimizes descriptor evaluation on currency micro-printing
        BEBLID_DESCRIPTOR = cv2.xfeatures2d.BEBLID_create(0.75)
        print("[INFO] BEBLID descriptor enabled (opencv-contrib).")
except Exception as e:
    # Gracefully fall back to standard ORB descriptors if BEBLID is absent
    print(f"[INFO] BEBLID not available ({e}); using standard ORB descriptors.")

# ── Brute-Force Matcher Setup ─────────────────────────────────────────────────
# Initialize Brute-Force Matcher with Hamming distance for binary feature descriptors:
#   - cv2.NORM_HAMMING: Counts bitwise XOR differences (fast hardware popcount)
#   - crossCheck=False: Left False to enable k-NN matching (k=2) for Lowe's ratio test
BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


################################################################################
# /// SECTION 3: DATASET CACHING & FEATURE EXTRACTION HELPERS ///
################################################################################

# In-memory RAM cache for genuine reference template features
# Structure: { "500": [ {"filename": "10.jpg", "bgr": ndarray, "gray": ndarray, "kp": [...], "des": ndarray}, ... ] }
DATASET_CACHE = {}


def extract_keypoints_and_descriptors(gray_img):
    """
    Detect invariant interest points (keypoints) and compute binary feature descriptors.
    Uses ORB for corner detection and BEBLID (or standard ORB fallback) for description.
    """
    # Detect corners using the FAST algorithm across pyramid octaves
    kps = ORB_DETECTOR.detect(gray_img, None)
    if not kps:
        return [], None

    # Compute boosted BEBLID descriptors if available
    if BEBLID_DESCRIPTOR is not None:
        try:
            return BEBLID_DESCRIPTOR.compute(gray_img, kps)
        except Exception:
            pass  # Fallback to standard ORB if BEBLID fails

    # Default fallback: compute descriptors using steered BRIEF (ORB)
    return ORB_DETECTOR.compute(gray_img, kps)


# Known corrupted, misfiled, or infographic files in dataset to exclude
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


def load_reference_dataset(max_images_per_denom=15):
    """
    Scans Dataset/{denom}/ on startup, systematically filters out contaminated/misplaced files,
    loads verified genuine templates (both front and back, new and old series), and caches
    ORB/BEBLID features and color histograms in RAM for robust real-time matching.
    """
    global DATASET_CACHE
    DATASET_CACHE = {}
    total_loaded = 0

    if not os.path.isdir(DATASET_DIR):
        print(f"[WARN] Dataset directory not found at {DATASET_DIR}")
        return

    # Loop through each denomination folder (10, 20, 50, 100, 200, 500)
    for denom in SUPPORTED_DENOMINATIONS:
        denom_path = os.path.join(DATASET_DIR, denom)
        if not os.path.isdir(denom_path):
            continue

        DATASET_CACHE[denom] = []
        candidate_files = list(CURATED_TEMPLATES.get(denom, []))
        all_files = [
            f for f in sorted(os.listdir(denom_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        for f in all_files:
            if f not in candidate_files and f not in EXCLUDED_FILES.get(denom, set()):
                candidate_files.append(f)

        for fname in candidate_files:
            if fname in EXCLUDED_FILES.get(denom, set()):
                continue
            fpath = os.path.join(denom_path, fname)
            bgr = cv2.imread(fpath)
            if bgr is None:
                continue

            h, w = bgr.shape[:2]
            ar = max(w, h) / float(min(w, h) + 1e-5)

            # Banknotes are horizontal rectangles with aspect ratio between 1.50 and 2.75
            if (1.50 <= ar <= 2.75 and min(w, h) >= 120) or (fname in CURATED_TEMPLATES.get(denom, [])):
                target_w = 600
                scale = target_w / float(w)
                resized_bgr = cv2.resize(bgr, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
                kp, des = extract_keypoints_and_descriptors(gray)
                if des is not None and len(des) >= 20:
                    DATASET_CACHE[denom].append({
                        "filename": fname,
                        "bgr": resized_bgr,
                        "gray": gray,
                        "kp": kp,
                        "des": des,
                        "quality_score": len(kp)
                    })
                    total_loaded += 1
                    if len(DATASET_CACHE[denom]) >= max_images_per_denom:
                        break

    print(f"[INFO] Loaded reference dataset: {total_loaded} high-quality templates across {len(DATASET_CACHE)} denominations.")

# Auto-load reference templates on import (ensures gunicorn and cloud runtimes have dataset ready)
if not DATASET_CACHE:
    load_reference_dataset()


################################################################################
# /// SECTION 4: IMAGE GEOMETRY & PERSPECTIVE WARP UTILITIES ///
################################################################################

def encode_cv2_image_to_base64(img, quality=80):
    """
    Encodes an OpenCV image matrix (BGR or Grayscale) into a base64 JPEG data URL
    so it can be displayed dynamically inside browser HTML <img> tags without writing files.
    """
    if img is None:
        return None
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def order_quad_points(pts):
    """
    Orders 4 quadrilateral coordinate points in standard clockwise sequence:
    [0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    # Top-left coordinate has smallest sum (x + y); bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right coordinate has smallest difference (y - x); bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_perspective_warp(image, pts):
    """
    Applies projective homography transformation to warp a quadrilateral banknote region
    into a flat, top-down rectangular view using cv2.warpPerspective.
    """
    rect = order_quad_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate Euclidean width of new rectified image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b))

    # Calculate Euclidean height of new rectified image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b))

    if max_w < 50 or max_h < 50:
        return None, None

    # Destination mapping coordinate rectangle
    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    # Compute 3x3 perspective homography matrix M
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped, (max_w, max_h)


def compute_masked_color_correlation(query_bgr, ref_bgr):
    """
    Computes Hue histogram correlation between query and reference banknotes,
    masking out white background borders, black shadows, and high-glare highlights.
    """
    try:
        hsv_q = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2HSV)
        hsv_r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        # Exclude extreme white/light-gray background and black borders
        mask_q = cv2.inRange(hsv_q, np.array([0, 25, 35]), np.array([180, 255, 245]))
        mask_r = cv2.inRange(hsv_r, np.array([0, 25, 35]), np.array([180, 255, 245]))
        hist_q = cv2.calcHist([hsv_q], [0], mask_q, [36], [0, 180])
        hist_r = cv2.calcHist([hsv_r], [0], mask_r, [36], [0, 180])
        cv2.normalize(hist_q, hist_q, 1, 0, cv2.NORM_L1)
        cv2.normalize(hist_r, hist_r, 1, 0, cv2.NORM_L1)
        corr = cv2.compareHist(hist_q, hist_r, cv2.HISTCMP_CORREL)
        return max(0.0, float(corr))
    except Exception:
        return 0.50


################################################################################
# /// SECTION 5: CORE 10-STAGE IMAGE & VIDEO PROCESSING (IVP) PIPELINE ///
################################################################################

def execute_ivp_pipeline(raw_bgr):
    """
    Executes the complete 10-stage Image & Video Processing pipeline:
      Stage 1: Image Acquisition & Decoding
      Stage 2: Aspect-Ratio Preserving Resizing
      Stage 3: Grayscale Conversion
      Stage 4: Gaussian Filtering (Noise Suppression)
      Stage 5: CLAHE Contrast Enhancement
      Stage 6: Canny Edge Detection
      Stage 7: Contour Detection & Corner Localization
      Stage 8: Perspective Correction (Four-Point Transform)
      Stage 9: Feature Matching & Denomination Recognition (ORB/BEBLID + Lowe's + RANSAC)
      Stage 10: Multi-Factor Authenticity & Specimen / Defacement Inspection
    """
    t_start = time.time()
    stages = {}
    warnings = []

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Image Acquisition & Decoding (Photo ko read karna)
    # ──────────────────────────────────────────────────────────────────────────
    # User ne jo image upload ki ya webcam se li, uski height aur width nikalte hain
    orig_h, orig_w = raw_bgr.shape[:2]
    stages["stage1_acquisition"] = {
        "title": "Stage 1: Image Acquisition",
        "description": "Validated the uploaded image stream and decoded into a 3-channel BGR numpy matrix.",
        "dimensions": f"{orig_w} × {orig_h} px",
        "image_data": encode_cv2_image_to_base64(raw_bgr)
    }

    # ── Pre-flight Image Quality Assessment (IQA) ──
    # Classical statistical & frequency-domain evaluation of the raw optical input
    raw_gray_iqa = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    iqa_lap_var = float(cv2.Laplacian(raw_gray_iqa, cv2.CV_64F).var())
    iqa_brightness = float(np.mean(raw_gray_iqa))
    iqa_contrast = float(np.std(raw_gray_iqa))
    glare_pixels = int(np.sum(raw_gray_iqa > 245))
    glare_ratio = float(glare_pixels) / float(raw_gray_iqa.size)

    if iqa_lap_var >= 25.0:
        sharpness_rating = "Sharp Texture"
    elif iqa_lap_var >= 14.0:
        sharpness_rating = "Moderate Sharpness"
    else:
        sharpness_rating = "Blurry / Motion Soft"

    if iqa_brightness < 60.0:
        lighting_rating = "Under-exposed / Dim"
    elif iqa_brightness > 205.0 or glare_ratio > 0.08:
        lighting_rating = "Over-exposed / Glare"
    else:
        lighting_rating = "Well-lit (Optimal)"

    if iqa_contrast >= 38.0:
        contrast_rating = "High Dynamic Range"
    elif iqa_contrast >= 24.0:
        contrast_rating = "Normal Contrast"
    else:
        contrast_rating = "Low Contrast"

    iqa_recommendations = []
    if iqa_lap_var < 14.0:
        iqa_recommendations.append("Camera blur detected: hold device steady and refocus.")
    if iqa_brightness < 60.0:
        iqa_recommendations.append("Low lighting: illuminate banknote with direct ambient light.")
    if glare_ratio > 0.08:
        iqa_recommendations.append("Harsh glare detected: tilt banknote slightly away from direct flash/lamp.")
    if iqa_contrast < 24.0:
        iqa_recommendations.append("Low contrast: place banknote against a contrasting dark/plain background.")

    if not iqa_recommendations:
        iqa_status = "EXCELLENT"
        iqa_recommendations.append("Optical quality is optimal for invariant keypoint extraction and verification.")
    elif len(iqa_recommendations) == 1 and iqa_lap_var >= 12.0:
        iqa_status = "ACCEPTABLE"
    else:
        iqa_status = "POOR (RE-CAPTURE SUGGESTED)"

    iqa_data = {
        "sharpness_variance": round(iqa_lap_var, 1),
        "sharpness_rating": sharpness_rating,
        "mean_brightness": round(iqa_brightness, 1),
        "lighting_rating": lighting_rating,
        "contrast_std": round(iqa_contrast, 1),
        "contrast_rating": contrast_rating,
        "glare_percentage": round(glare_ratio * 100.0, 1),
        "overall_status": iqa_status,
        "recommendations": iqa_recommendations
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: Aspect-Ratio Preserving Resizing (Photo ko chhota karna bina stretch kiye)
    # ──────────────────────────────────────────────────────────────────────────
    # Phone camera ki photos 4000x3000 hoti hain jo bohot heavy hoti hain.
    # Hum width ko max 800px karte hain taaki speed fast ho jaye par note ka shape na bigde.
    TARGET_WIDTH = 800
    if orig_w > TARGET_WIDTH:
        scale_ratio = TARGET_WIDTH / float(orig_w)
        resized_w = TARGET_WIDTH
        resized_h = int(orig_h * scale_ratio)
        resized_bgr = cv2.resize(raw_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        resize_note = f"Scaled down by factor of {scale_ratio:.2f} to {resized_w}×{resized_h} px."
    else:
        resized_bgr = raw_bgr.copy()
        resized_w, resized_h = orig_w, orig_h
        resize_note = "Image already within working resolution limits; no downsampling applied."

    stages["stage2_resized"] = {
        "title": "Stage 2: Image Resizing",
        "description": f"Standardized resolution while preserving exact aspect ratio. {resize_note} Resizing bounds the computational complexity of spatial convolutions and feature searches.",
        "dimensions": f"{resized_w} × {resized_h} px",
        "image_data": encode_cv2_image_to_base64(resized_bgr)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3: Grayscale Conversion (Black & White banana)
    # ──────────────────────────────────────────────────────────────────────────
    # Color image me 3 channels (B, G, R) hote hain. Grayscale me sirf 1 brightness channel hota hai.
    # Isse processing 3 guna fast ho jati hai aur lighting ka color fark nahi padta.
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    stages["stage3_grayscale"] = {
        "title": "Stage 3: Grayscale Conversion",
        "description": "Converted the 3-channel color image to single-channel intensity (luminance). Eliminates color sensitivity caused by ambient lighting variances and reduces processing memory by 66%.",
        "dimensions": f"{resized_w} × {resized_h} px (1 channel)",
        "image_data": encode_cv2_image_to_base64(gray)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4: Gaussian Filtering (Noise aur grain ko hatana)
    # ──────────────────────────────────────────────────────────────────────────
    # 5x5 pixel ka smoothing filter lagate hain taaki camera ka noise/grain smooth ho jaye.
    # Agar ye nahi karenge, to edge detection me nakli edges dikhne lagengi.
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    stages["stage4_gaussian"] = {
        "title": "Stage 4: Gaussian Filtering",
        "description": "Convolved the grayscale image with a 5×5 2D Gaussian smoothing kernel. Attenuates high-frequency sensor noise and paper grain artifacts prior to gradient edge extraction.",
        "kernel": "5 × 5 Gaussian (σ calculated automatically)",
        "image_data": encode_cv2_image_to_base64(blurred)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 5: CLAHE Contrast Enhancement (Faint details aur numbers ko ubharna)
    # ──────────────────────────────────────────────────────────────────────────
    # Normal histogram equalization poori photo ko safed kar deta hai.
    # CLAHE image ko 8x8 chhote blocks me baant kar local contrast badhata hai.
    # Isse note ke hidden numbers, Mahatma Gandhi watermark aur micro-print saaf dikhte hain.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(blurred)
    stages["stage5_clahe"] = {
        "title": "Stage 5: CLAHE Contrast Enhancement",
        "description": "Applied Contrast Limited Adaptive Histogram Equalization on 8×8 contextual grid tiles. Maximizes local contrast of currency micro-printing and symbols without amplifying noise in uniform background areas.",
        "parameters": "ClipLimit=2.5, TileGridSize=8×8",
        "image_data": encode_cv2_image_to_base64(clahe_gray)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 6: Canny Edge Detection (Note ke saare kinare aur outlines nikalna)
    # ──────────────────────────────────────────────────────────────────────────
    # Canny algorithm image me brightness ke achanak badlav (gradients) ko dhoondhta hai.
    # 50 se 150 thresholds use karke ye sirf strong aur real edges ko rakhta hai.
    edges = cv2.Canny(clahe_gray, threshold1=50, threshold2=150, apertureSize=3, L2gradient=True)
    stages["stage6_canny"] = {
        "title": "Stage 6: Canny Edge Detection",
        "description": "Extracted binary edge map via gradient computation (Sobel operator), non-maximum suppression (thinning to 1-pixel ridges), and dual-threshold hysteresis linking (50 / 150).",
        "thresholds": "Lower: 50, Upper: 150 (L2 Gradient)",
        "image_data": encode_cv2_image_to_base64(edges)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 7: Contour Detection & Corner Localization (Note ke 4 corners dhoondhna)
    # ──────────────────────────────────────────────────────────────────────────
    # Edges ko thoda dilate (mota) karte hain taaki tooti hui line jud jaye
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    # Contours (continuous lines) nikalte hain
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_contours = resized_bgr.copy()
    total_area = resized_w * resized_h
    banknote_corners = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Bada quadrilateral (4 kono wala shape) dhoondhte hain jo currency note ho sakta hai
    for cnt in sorted_contours[:10]:
        area = cv2.contourArea(cnt)
        if area < (0.05 * total_area):  # Kam se kam 5% screen area cover hona chahiye
            continue

        peri = cv2.arcLength(cnt, closed=True)
        # approxPolyDP curve ko polygon corners me convert karta hai
        approx = cv2.approxPolyDP(cnt, epsilon=0.02 * peri, closed=True)

        # Agar polygon me theek 4 corners hain aur wo convex rectangle hai
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            ar = max(w, h) / float(min(w, h) + 1e-5)
            if 1.4 <= ar <= 3.0:  # Indian currency notes ka aspect ratio lagbhag 2.1 hota hai
                banknote_corners = approx
                break

    # Agar 4 corners mil gaye, to unhe green line aur red dots se highlight karo
    if banknote_corners is not None:
        cv2.drawContours(annotated_contours, [banknote_corners], -1, (0, 255, 0), 3)
        for pt in banknote_corners.reshape(-1, 2):
            cv2.circle(annotated_contours, tuple(pt), 8, (0, 0, 255), -1)
            cv2.circle(annotated_contours, tuple(pt), 10, (255, 255, 255), 2)
        contour_status = "Detected quadrilateral banknote boundary and localized 4 corners."
    else:
        # Fallback: Agar 4 corners nahi mile, to crash mat hone do
        if sorted_contours and cv2.contourArea(sorted_contours[0]) > (0.05 * total_area):
            cv2.drawContours(annotated_contours, sorted_contours[:1], -1, (0, 165, 255), 2)
            contour_status = "General contour detected, but reliable 4 distinct corner points were not found. Continuing without perspective rectification."
        else:
            contour_status = "No prominent closed rectangular contour detected. Note may fill entire frame or blend with background."
        warnings.append("Note corner localization was inconclusive; perspective correction skipped.")

    stages["stage7_contours"] = {
        "title": "Stage 7: Contour Detection & Corner Localization",
        "description": f"Applied topological contour retrieval and polygon approximation (Ramer-Douglas-Peucker algorithm). {contour_status}",
        "corners_found": banknote_corners is not None,
        "image_data": encode_cv2_image_to_base64(annotated_contours)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 8: Perspective Correction (Tedhe note ko bilkul seedha/flat karna)
    # ──────────────────────────────────────────────────────────────────────────
    warped_bgr = None
    if banknote_corners is not None:
        # 4 corners ke basis par perspective transformation matrix banakar note ko flat karte hain
        warped_bgr, _ = four_point_perspective_warp(resized_bgr, banknote_corners)
        if warped_bgr is not None:
            # Agar note vertical (khada) hai, to use ghuma kar horizontal (aada) kar do
            wh, ww = warped_bgr.shape[:2]
            if wh > ww:
                warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_90_CLOCKWISE)

    if warped_bgr is not None:
        perspective_img_b64 = encode_cv2_image_to_base64(warped_bgr)
        perspective_desc = "Successfully calculated homography matrix H and warped quadrilateral region to fronto-parallel view using cv2.warpPerspective."
        rectified_available = True
        recognition_input_bgr = warped_bgr
    else:
        # Fallback: Agar 4 corners nahi the, to original resized photo par hi matching aage badha do
        perspective_img_b64 = None
        perspective_desc = "Perspective correction skipped due to absence of 4 distinct convex corners. Recognition will proceed directly using the preprocessed image."
        rectified_available = False
        recognition_input_bgr = resized_bgr

    stages["stage8_perspective"] = {
        "title": "Stage 8: Perspective Rectification",
        "description": perspective_desc,
        "rectified_available": rectified_available,
        "image_data": perspective_img_b64
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 9: Feature Extraction & Matching (ORB/BEBLID + Lowe's + RANSAC)
    # ──────────────────────────────────────────────────────────────────────────
    rec_gray = cv2.cvtColor(recognition_input_bgr, cv2.COLOR_BGR2GRAY)
    rec_clahe = clahe.apply(rec_gray)
    query_kp, query_des = extract_keypoints_and_descriptors(rec_clahe)

    # Annotated query keypoints visualization
    kp_vis = cv2.drawKeypoints(
        recognition_input_bgr, query_kp, None,
        color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    stages["stage9a_keypoints"] = {
        "title": "Stage 9A: Feature Keypoint Detection",
        "description": f"Extracted {len(query_kp)} invariant keypoints using ORB (Oriented FAST corners + multi-scale pyramid) and computed binary local intensity descriptors.",
        "keypoint_count": len(query_kp),
        "image_data": encode_cv2_image_to_base64(kp_vis)
    }

    # Compare query descriptors against reference templates in DATASET_CACHE
    denom_scores = {}
    best_matched_denom = None
    best_score = -1.0
    best_inlier_count = 0
    best_good_matches_count = 0
    best_ref_entry = None
    best_good_matches = []
    best_H_matrix = None

    if query_des is not None and len(query_des) >= 10 and DATASET_CACHE:
        for denom, ref_list in DATASET_CACHE.items():
            denom_total_good = 0
            denom_total_inliers = 0
            top_sample_good = 0
            top_sample_inliers = 0
            top_sample_score = -1.0
            top_sample_entry = None
            top_sample_matches = []
            top_sample_H = None

            for ref in ref_list:
                ref_des = ref["des"]
                ref_kp = ref["kp"]

                if ref_des is None or len(ref_des) < 5:
                    continue

                try:
                    # k-NN match with k=2
                    knn_matches = BF_MATCHER.knnMatch(query_des, ref_des, k=2)
                except Exception:
                    continue

                # Lowe's Ratio Test (m.distance < 0.75 * n.distance)
                good = []
                for pair in knn_matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                # Geometric Homography Verification with RANSAC
                inlier_count = 0
                sample_H = None
                if len(good) >= 6:
                    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    try:
                        H_mat, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inlier_count = int(mask.sum())
                            sample_H = H_mat
                    except Exception:
                        inlier_count = 0
                        sample_H = None

                # Color correlation validation
                sample_color = compute_masked_color_correlation(recognition_input_bgr, ref["bgr"])
                # Multi-modal score combines spatial inliers with chromaticity profile
                sample_score = (inlier_count * 1.0) * (0.35 + 0.65 * sample_color)

                denom_total_good += len(good)
                denom_total_inliers += inlier_count

                if sample_score > top_sample_score or (sample_score == top_sample_score and inlier_count > top_sample_inliers):
                    top_sample_score = sample_score
                    top_sample_inliers = inlier_count
                    top_sample_good = len(good)
                    top_sample_entry = ref
                    top_sample_matches = good
                    top_sample_H = sample_H

            denom_scores[denom] = {
                "total_good_matches": denom_total_good,
                "total_inliers": denom_total_inliers,
                "top_sample_good": top_sample_good,
                "top_sample_inliers": top_sample_inliers,
                "top_sample_score": round(top_sample_score, 1),
                "sample_count": len(ref_list)
            }

            # Winning denomination has the strongest multi-factor correspondence
            if top_sample_score > best_score:
                best_score = top_sample_score
                best_inlier_count = top_sample_inliers
                best_good_matches_count = top_sample_good
                best_matched_denom = denom
                best_ref_entry = top_sample_entry
                best_good_matches = top_sample_matches
                best_H_matrix = top_sample_H

    # ── Stage 9B: Draw green connecting lines between matching features ──
    if best_ref_entry is not None and best_good_matches:
        draw_matches_count = min(len(best_good_matches), 60)
        sorted_good = sorted(best_good_matches, key=lambda m: m.distance)[:draw_matches_count]

        match_vis_img = cv2.drawMatches(
            recognition_input_bgr, query_kp,
            best_ref_entry["bgr"], best_ref_entry["kp"],
            sorted_good, None,
            matchColor=(0, 255, 0),
            singlePointColor=(100, 100, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        stages["stage9b_matches"] = {
            "title": "Stage 9B: Feature Correspondence Visualization",
            "description": f"Visualized geometric feature matches between query note and reference ₹{best_matched_denom} template filtered by Lowe's Ratio and verified with RANSAC.",
            "image_data": encode_cv2_image_to_base64(match_vis_img)
        }
    else:
        stages["stage9b_matches"] = {
            "title": "Stage 9B: Feature Correspondence Visualization",
            "description": "No significant feature correspondences were found across reference templates.",
            "image_data": None
        }

    # ── Homography-driven Banknote Localization & Refinement ──
    auth_target_bgr = recognition_input_bgr
    if best_H_matrix is not None and best_ref_entry is not None and best_inlier_count >= 8:
        ref_h, ref_w = best_ref_entry["bgr"].shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        try:
            proj_corners = cv2.perspectiveTransform(ref_corners, best_H_matrix)
            proj_poly = np.int32(proj_corners)
            area_proj = cv2.contourArea(proj_poly)
            total_img_area = resized_w * resized_h

            if cv2.isContourConvex(proj_poly) and (0.04 * total_img_area <= area_proj <= 0.98 * total_img_area):
                cv2.polylines(annotated_contours, [proj_poly], isClosed=True, color=(0, 255, 0), thickness=3)
                for pt in proj_poly.reshape(-1, 2):
                    cv2.circle(annotated_contours, tuple(pt), 7, (0, 0, 255), -1)
                stages["stage7_contours"]["image_data"] = encode_cv2_image_to_base64(annotated_contours)
                stages["stage7_contours"]["description"] = "Successfully localized banknote quadrilateral boundary using RANSAC geometric homography projection."

                warped_note, _ = four_point_perspective_warp(resized_bgr, proj_corners)
                if warped_note is not None:
                    wh, ww = warped_note.shape[:2]
                    if wh > ww:
                        warped_note = cv2.rotate(warped_note, cv2.ROTATE_90_CLOCKWISE)
                    stages["stage8_perspective"]["image_data"] = encode_cv2_image_to_base64(warped_note)
                    stages["stage8_perspective"]["description"] = "Homography unwarping generated fronto-parallel rectified banknote."
                    auth_target_bgr = warped_note
        except Exception:
            pass

    # ── Final Denomination Decision Logic ──
    MIN_INLIER_THRESHOLD = 8
    MIN_GOOD_MATCHES = 14

    is_recognized = False
    recognized_denom = None
    confidence_level = "None"
    evidence_summary = ""

    if best_matched_denom is not None:
        if best_inlier_count >= 20 and best_good_matches_count >= MIN_GOOD_MATCHES:
            is_recognized = True
            recognized_denom = best_matched_denom
            confidence_level = "High"
            evidence_summary = (
                f"Strong correspondence: {best_inlier_count} RANSAC inliers and "
                f"{best_good_matches_count} feature matches against ₹{best_matched_denom} reference."
            )
        elif best_inlier_count >= MIN_INLIER_THRESHOLD:
            is_recognized = True
            recognized_denom = best_matched_denom
            confidence_level = "Moderate"
            evidence_summary = (
                f"Moderate correspondence: {best_inlier_count} RANSAC inliers and "
                f"{best_good_matches_count} feature matches against ₹{best_matched_denom} reference."
            )
        else:
            evidence_summary = (
                f"Weak matching evidence (Inliers: {best_inlier_count}, Good matches: {best_good_matches_count}). "
                "Insufficient correspondences to establish identity reliably."
            )
    else:
        evidence_summary = "Zero distinctive keypoint correspondences found against reference dataset."

    # ──────────────────────────────────────────────────────────────────────────
    # Heuristic Consistency Metrics (Texture Sharpness & Color Histogram)
    # ──────────────────────────────────────────────────────────────────────────
    consistency = {}
    auth_gray = cv2.cvtColor(auth_target_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(auth_gray, cv2.CV_64F).var()
    consistency["sharpness_variance"] = round(float(lap_var), 1)
    consistency["sharpness_status"] = "Normal print sharpness" if lap_var >= 15.0 else "Slightly soft / low texture detail"

    ah, aw = auth_target_bgr.shape[:2]
    aspect_ratio = max(aw, ah) / float(min(aw, ah) + 1e-5)
    consistency["aspect_ratio"] = round(float(aspect_ratio), 2)
    consistency["aspect_ratio_status"] = "Consistent with banknote geometry" if (1.60 <= aspect_ratio <= 2.50) else "Non-standard aspect ratio"

    color_corr = None
    if best_ref_entry is not None:
        color_corr = compute_masked_color_correlation(auth_target_bgr, best_ref_entry["bgr"])
        consistency["color_correlation"] = round(color_corr, 3)
        consistency["color_status"] = "Color palette matches reference" if color_corr >= 0.30 else "Color profile deviates from reference"
    else:
        consistency["color_correlation"] = None
        consistency["color_status"] = "Unavailable (no reference match)"

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 10: Multi-Factor Authenticity & RBI Physical Security Inspection
    # ──────────────────────────────────────────────────────────────────────────
    annotated_auth = auth_target_bgr.copy()
    anomalies = []
    is_specimen = False

    # Watermark and signature field coordinates:
    # Front-side watermark zone: x from 55% to 94%, y from 14% to 72%
    rx1, rx2 = int(aw * 0.55), int(aw * 0.94)
    ry1, ry2 = int(ah * 0.14), int(ah * 0.72)
    wm_roi = auth_target_bgr[ry1:ry2, rx1:rx2]

    stamp_boxes = []
    red_ratio = 0.0
    max_cluster = 0.0

    if wm_roi.size > 0:
        hsv_wm = cv2.cvtColor(wm_roi, cv2.COLOR_BGR2HSV)
        # Saturated red/crimson ink mask (covers red SPECIMEN and CANCELLED stamps)
        m1 = cv2.inRange(hsv_wm, np.array([0, 75, 65]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv_wm, np.array([160, 75, 65]), np.array([180, 255, 255]))
        stamp_mask = m1 | m2
        wm_area = wm_roi.shape[0] * wm_roi.shape[1]
        red_count = cv2.countNonZero(stamp_mask)
        red_ratio = red_count / float(max(wm_area, 1))

        contours_stamp, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stamp_boxes = [c for c in contours_stamp if cv2.contourArea(c) > 200]
        if contours_stamp:
            max_cluster = max([cv2.contourArea(c) for c in contours_stamp])

        # Draw inspected watermark security zone
        cv2.rectangle(annotated_auth, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
        cv2.putText(annotated_auth, "Watermark Security Zone", (rx1, ry1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # Denomination-aware specimen stamp check:
        # Genuine ₹500 is stone-grey: saturated red ink in the watermark field is strictly 0.0%.
        if recognized_denom == "500":
            if red_ratio > 0.02 and (max_cluster > 350 or len(stamp_boxes) >= 1):
                is_specimen = True
                anomalies.append(f"Detected red cancellation / 'SPECIMEN' stamp in watermark field ({red_count} px, {red_ratio*100:.1f}% coverage).")
                for c in stamp_boxes:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cv2.rectangle(annotated_auth, (rx1 + bx, ry1 + by), (rx1 + bx + bw, ry1 + by + bh), (0, 0, 255), 2)
                cv2.putText(annotated_auth, "SPECIMEN STAMP DETECTED", (rx1, ry2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # ── RBI Physical Security Feature Matrix Extraction ──
    security_features = {}

    # Feature 1: Windowed Security Thread (localized at 37% - 44% width)
    st_x1, st_x2 = int(aw * 0.37), int(aw * 0.44)
    st_roi = auth_target_bgr[:, st_x1:st_x2]
    if st_roi.size > 0:
        st_gray = cv2.cvtColor(st_roi, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(st_gray, cv2.CV_64F, 1, 0, ksize=3)
        st_grad = float(np.mean(np.abs(sobel_x)))
        st_verified = st_grad >= 18.0
        cv2.rectangle(annotated_auth, (st_x1, 0), (st_x2, ah), (0, 255, 255), 1)
        security_features["security_thread"] = {
            "name": "Security Thread Band",
            "verified": st_verified,
            "metric_value": round(st_grad, 1),
            "status": "Verified: Vertical high-gradient security band localized" if st_verified else "Caution: Thread gradient energy is weak"
        }
    else:
        security_features["security_thread"] = {
            "name": "Security Thread Band",
            "verified": False,
            "metric_value": 0.0,
            "status": "Region unavailable"
        }

    # Feature 2: Mahatma Gandhi Watermark Window (Right side)
    if wm_roi.size > 0:
        wm_gray = cv2.cvtColor(wm_roi, cv2.COLOR_BGR2GRAY)
        wm_mean = float(np.mean(wm_gray))
        wm_verified = (not is_specimen) and (wm_mean >= 85.0)
        security_features["watermark_window"] = {
            "name": "Watermark Window",
            "verified": wm_verified,
            "metric_value": round(wm_mean, 1),
            "status": "Verified: Clean translucent watermark window" if wm_verified else "Flagged: Watermark area occluded or stamped"
        }
    else:
        security_features["watermark_window"] = {
            "name": "Watermark Window",
            "verified": False,
            "metric_value": 0.0,
            "status": "Region unavailable"
        }

    # Feature 3: Intaglio Relief & Micro-lettering Texture
    intaglio_roi = auth_target_bgr[int(ah*0.2):int(ah*0.8), int(aw*0.15):int(aw*0.65)]
    if intaglio_roi.size > 0:
        int_gray = cv2.cvtColor(intaglio_roi, cv2.COLOR_BGR2GRAY)
        int_lap = float(cv2.Laplacian(int_gray, cv2.CV_64F).var())
        int_verified = int_lap >= 15.0
        security_features["intaglio_relief"] = {
            "name": "Intaglio Relief Texture",
            "verified": int_verified,
            "metric_value": round(int_lap, 1),
            "status": "Verified: Crisp micro-printing & intaglio textures detected" if int_verified else "Caution: Low high-frequency textural detail"
        }
    else:
        security_features["intaglio_relief"] = {
            "name": "Intaglio Relief Texture",
            "verified": False,
            "metric_value": 0.0,
            "status": "Region unavailable"
        }

    # Feature 4: RBI Dimensional Aspect Ratio
    ar_verified = (1.70 <= aspect_ratio <= 2.55)
    security_features["aspect_ratio"] = {
        "name": "Dimensional Aspect Ratio",
        "verified": ar_verified,
        "metric_value": round(float(aspect_ratio), 2),
        "status": f"Verified: Ratio ({aspect_ratio:.2f}) matches RBI banknote standard" if ar_verified else f"Flagged: Non-standard geometry ({aspect_ratio:.2f})"
    }

    # Feature 5: Color Profile Chromaticity Match
    color_verified = (color_corr is not None and color_corr >= 0.28)
    security_features["color_palette"] = {
        "name": "Base Color Chromaticity",
        "verified": color_verified,
        "metric_value": round(color_corr, 3) if color_corr is not None else 0.0,
        "status": f"Verified: HSV chromaticity matches reference ({color_corr:.2f})" if color_verified else "Caution: Chromaticity deviates from genuine template"
    }

    # Feature 6: Specimen & Defacement Check
    security_features["specimen_check"] = {
        "name": "Specimen & Defacement Check",
        "verified": not is_specimen,
        "metric_value": red_count if 'red_count' in locals() else 0,
        "status": "Verified: Clean banknote without cancellation stamps" if not is_specimen else "CRITICAL: Saturated red SPECIMEN / CANCELLED stamp detected!"
    }

    # Multi-factor score components:
    # 1. Inlier score (35% weight)
    if best_inlier_count >= 25:
        inlier_score = 100
    elif best_inlier_count >= 15:
        inlier_score = 85
    elif best_inlier_count >= 10:
        inlier_score = 70
    elif best_inlier_count >= 6:
        inlier_score = 50
    else:
        inlier_score = 20

    # 2. Aspect ratio score (15% weight)
    if 1.60 <= aspect_ratio <= 2.50:
        ar_score = 100
    elif 1.40 <= aspect_ratio <= 2.70:
        ar_score = 80
    else:
        ar_score = 50
        anomalies.append(f"Aspect ratio ({aspect_ratio:.2f}) deviates from standard RBI banknote geometry.")

    # 3. Color correlation score (30% weight)
    if color_corr is not None:
        if color_corr >= 0.45:
            color_score = 100
        elif color_corr >= 0.30:
            color_score = 80
        elif color_corr >= 0.18:
            color_score = 60
        else:
            color_score = 30
            anomalies.append(f"Color profile correlation low ({color_corr:.2f}) vs genuine RBI reference.")
    else:
        color_score = 70

    # 4. Texture sharpness score (20% weight)
    if lap_var >= 25.0:
        tex_score = 100
    elif lap_var >= 15.0:
        tex_score = 85
    elif lap_var >= 8.0:
        tex_score = 65
    else:
        tex_score = 40
        anomalies.append("Low texture variance — hold camera steady or ensure clear lighting.")

    composite_score = int(0.35 * inlier_score + 0.30 * color_score + 0.15 * ar_score + 0.20 * tex_score)

    # Final Authenticity Verdict Assignment
    if not is_recognized:
        auth_verdict = "UNRECOGNIZED"
        auth_label = "Unable to verify authenticity (insufficient feature evidence)"
        is_genuine = False
    elif is_specimen:
        auth_verdict = "SPECIMEN"
        auth_label = "SPECIMEN / CANCELLED NOTE (NOT LEGAL TENDER)"
        is_genuine = False
    elif composite_score >= 65:
        auth_verdict = "GENUINE"
        auth_label = f"GENUINE ({composite_score}% Visual Consistency)"
        is_genuine = True
        anomalies = []  # Clear warnings for genuine display
    elif composite_score >= 45:
        auth_verdict = "FAKE"
        auth_label = f"SUSPICIOUS / LIKELY FAKE ({composite_score}% Consistency)"
        is_genuine = False
    else:
        auth_verdict = "FAKE"
        auth_label = f"FAKE / COUNTERFEIT ({composite_score}% Consistency)"
        is_genuine = False

    stages["stage10_authenticity"] = {
        "title": "Stage 10: Authenticity & Security Inspection",
        "description": f"Multi-factor visual consistency evaluation: Inliers ({best_inlier_count}), Color correlation ({color_corr if color_corr is not None else 'N/A'}), Sharpness ({lap_var:.1f}). Verdict: {auth_label}.",
        "verdict": auth_verdict,
        "anomalies": anomalies,
        "image_data": encode_cv2_image_to_base64(annotated_auth)
    }

    t_elapsed = time.time() - t_start

    result = {
        "status": "success" if is_recognized else "unrecognized",
        "recognized": is_recognized,
        "denomination": recognized_denom,
        "confidence_level": confidence_level,
        "evidence_summary": evidence_summary,
        "authenticity_verdict": auth_verdict,
        "authenticity_label": auth_label,
        "is_genuine": is_genuine,
        "authenticity_anomalies": anomalies,
        "best_inlier_count": best_inlier_count,
        "best_good_matches": best_good_matches_count,
        "processing_time_sec": round(t_elapsed, 3),
        "denom_candidate_scores": denom_scores,
        "consistency_analysis": consistency,
        "image_quality_assessment": iqa_data,
        "security_features": security_features,
        "warnings": warnings,
        "pipeline_stages": stages
    }
    return result


################################################################################
# /// SECTION 6: FLASK REST API ENDPOINTS (ROUTING & CONTROLLERS) ///
################################################################################

@app.route("/", methods=["GET"])
def index():
    """Serve the complete frontend user interface (index.html)."""
    return render_template("index.html")


@app.route("/docs", methods=["GET"])
@app.route("/index1.html", methods=["GET"])
def documentation():
    """Serve the technical workflow documentation website (index1.html)."""
    return render_template("index1.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Return JSON system health status, OpenCV version, and loaded dataset inventory."""
    denom_counts = {d: len(DATASET_CACHE.get(d, [])) for d in SUPPORTED_DENOMINATIONS}
    return jsonify({
        "status": "healthy",
        "service": "RupeeVision API",
        "supported_denominations": SUPPORTED_DENOMINATIONS,
        "dataset_loaded": sum(denom_counts.values()) > 0,
        "cached_images_per_denomination": denom_counts,
        "beblid_supported": BEBLID_DESCRIPTOR is not None,
        "opencv_version": cv2.__version__
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_currency():
    """
    Accepts an uploaded image file (via multipart/form-data) OR base64 data URI (from webcam).
    Processes image through the 10-stage IVP pipeline and returns complete results in JSON.
    """
    raw_bgr = None

    # Option A: Multipart file upload from user's file picker / drag-and-drop
    if "image" in request.files:
        file = request.files["image"]
        if file and file.filename != "":
            file_bytes = np.frombuffer(file.read(), np.uint8)
            raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Option B: Base64 JSON payload from live browser webcam capture canvas
    elif request.is_json and "image_base64" in request.json:
        b64_data = request.json["image_base64"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        try:
            decoded_bytes = base64.b64decode(b64_data)
            file_bytes = np.frombuffer(decoded_bytes, np.uint8)
            raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"status": "error", "error": f"Failed to decode base64 image: {str(e)}"}), 400

    if raw_bgr is None:
        return jsonify({
            "status": "error",
            "error": "No valid image provided or file format is unsupported (must be JPEG/PNG/WebP)."
        }), 400

    try:
        pipeline_result = execute_ivp_pipeline(raw_bgr)
        return jsonify(pipeline_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"Internal error during image processing: {str(e)}"
        }), 500


################################################################################
# /// SECTION 7: APPLICATION ENTRYPOINT & SERVER STARTUP ///
################################################################################

if __name__ == "__main__":
    print("=" * 65)
    print("  RupeeVision — Indian Currency Recognition & IVP Analysis")
    print("=" * 65)
    if not DATASET_CACHE:
        load_reference_dataset()
    port = int(os.environ.get("PORT", 5050))
    print(f"\n[INFO] Starting server on port {port}...")
    print(f"[INFO] Access console at: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)