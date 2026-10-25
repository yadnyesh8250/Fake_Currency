"""
RupeeVision — Core IVP Demonstration
Core concepts: preprocessing, ORB features, descriptor matching,
RANSAC verification, HSV colour comparison, and result visualisation.

Dataset layout:
Dataset/
    10/   20/   50/   100/   200/   500/
Place reference banknote images in the matching denomination folders.

Run:
    python demo_detect_core.py Dataset/500/example.jpg
or:
    python demo_detect_core.py
"""

import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DENOMINATIONS = ["10", "20", "50", "100", "200", "500"]

# ORB finds distinctive image points; BFMatcher compares their binary descriptors.
orb = cv2.ORB_create(nfeatures=1000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def preprocess(image):
    """Resize, convert to grayscale, reduce noise, and improve local contrast."""
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    h, w = image.shape[:2]
    if w > 800:
        image = cv2.resize(image, (800, int(h * 800 / w)),
                           interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return image, enhanced_gray


def hue_histogram(image):
    """Create a normalized Hue histogram for a coarse colour comparison."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([180, 255, 245]))
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def load_templates():
    """Load reference images and calculate their features once."""
    templates = []

    for denomination in DENOMINATIONS:
        folder = os.path.join(DATASET_DIR, denomination)
        if not os.path.isdir(folder):
            continue

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image = cv2.imread(os.path.join(folder, filename))
            if image is None:
                continue

            image, gray = preprocess(image)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is not None and len(keypoints) >= 10:
                templates.append({
                    "denomination": denomination,
                    "filename": filename,
                    "image": image,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "histogram": hue_histogram(image)
                })

    return templates


def recognize(query_image, query_gray, templates):
    """Rank reference images using ORB matches, RANSAC inliers, and colour."""
    query_kp, query_des = orb.detectAndCompute(query_gray, None)
    if query_des is None or len(query_des) < 10:
        return "Unknown", None, [], 0, 0.0, query_kp

    query_hist = hue_histogram(query_image)
    best = None
    best_matches = []
    best_inliers = 0
    best_colour = 0.0
    best_score = -1.0

    for template in templates:
        pairs = matcher.knnMatch(query_des, template["descriptors"], k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]

        inliers = 0
        if len(good) >= 6:
            src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([template["keypoints"][m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())

        colour = max(0.0, float(cv2.compareHist(
            query_hist, template["histogram"], cv2.HISTCMP_CORREL)))
        score = inliers * (0.35 + 0.65 * colour)

        if score > best_score:
            best_score = score
            best = template
            best_matches = good
            best_inliers = inliers
            best_colour = colour

    denomination = best["denomination"] if best and best_inliers >= 7 else "Unknown"
    return denomination, best, best_matches, best_inliers, best_colour, query_kp


def main():
    templates = load_templates()
    if not templates:
        print("No reference templates found. Check the Dataset folders.")
        return

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
    else:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Could not open webcam.")
            return

        image = None
        print("Press SPACE to capture, or ESC to exit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.imshow("RupeeVision Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                image = frame.copy()
                break
            if key == 27:
                break
        camera.release()
        cv2.destroyAllWindows()

    if image is None:
        print("Could not read an image.")
        return

    processed, gray = preprocess(image)
    denomination, reference, matches, inliers, colour, query_kp = recognize(
        processed, gray, templates
    )

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Predicted denomination: {denomination}")
    print(f"RANSAC inliers: {inliers}")
    print(f"Colour correlation: {colour:.2f}")
    print(f"Image sharpness (Laplacian variance): {sharpness:.1f}")
    print("Note: these measurements do not prove that a banknote is genuine.")

    if reference is not None and matches:
        shown = cv2.drawMatches(
            processed, query_kp, reference["image"], reference["keypoints"],
            sorted(matches, key=lambda m: m.distance)[:35], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    else:
        shown = processed.copy()

    cv2.imshow("RupeeVision — Input and Reference Matches", shown)
    cv2.imwrite(os.path.join(BASE_DIR, "demo_result_output.jpg"), shown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
