# RupeeVision: Indian Currency Recognition & Image Analysis Using IVP

An educational Image & Video Processing (IVP) college mini-project demonstrating classical computer vision techniques for Indian banknote denomination recognition, geometric rectification, and multi-stage image analysis using OpenCV, NumPy, and Flask.

---

## 1. Project Overview & Context

This project satisfies university-level requirements for Image & Video Processing (IVP). Rather than relying on black-box deep learning models, RupeeVision showcases an **explainable, deterministic 10-stage classical computer vision pipeline** implemented with OpenCV.

### Core Architecture
- **Simplified Two-File Architecture**:
  - `app.py`: Complete backend containing all 10 IVP algorithmic stages and Flask REST endpoints.
  - `index.html`: Complete, standalone frontend UI with drag-and-drop upload, webcam capture, speech synthesis, and an interactive IVP stage visualizer.
  - `Dataset/`: Reference templates for ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 denominations.

---

## 2. Theoretical Image-Processing Pipeline

RupeeVision executes a 10-stage processing sequence on each query image:

```
[1. Acquisition] ➔ [2. Resize] ➔ [3. Grayscale] ➔ [4. Gaussian Blur] ➔ [5. CLAHE]
       ➔ [6. Canny Edges] ➔ [7. Contours & Corners] ➔ [8. Perspective Warp]
       ➔ [9. ORB/BEBLID Feature Matching & RANSAC] ➔ [10. Visual Consistency]
```

### Stage 1: Image Acquisition & Decoding
- Decodes JPEG/PNG/WebP byte stream into a 3-channel BGR NumPy array using `cv2.imdecode`.
- Validates matrix dimensions and handles corrupt files gracefully.

### Stage 2: Aspect-Ratio Preserving Resizing
- Standardizes maximum dimension (e.g. width 800px) while maintaining the original aspect ratio:
  $$\text{Scale} = \frac{W_{\text{target}}}{W_{\text{orig}}}, \quad H_{\text{new}} = H_{\text{orig}} \times \text{Scale}$$
- **Why it matters**: Bounds computational overhead of spatial convolutions, morphological filters, and feature searches without distorting banknote proportions.

### Stage 3: Grayscale Conversion
- Transforms 3-channel color image into a single-channel luminance map using the ITU-R BT.601 standard:
  $$Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$$
- **Why it matters**: Removes color sensitivity under varying ambient lighting conditions and cuts memory footprint by 66%.

### Stage 4: Gaussian Filtering (Noise Suppression)
- Convolves the image with a $5 \times 5$ Gaussian kernel:
  $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
- **Why it matters**: Low-pass filtering attenuates high-frequency sensor noise, salt-and-pepper artifacts, and paper grain before computing spatial derivatives.

### Stage 5: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- Divides the image into $8 \times 8$ contextual tiles, calculates local histograms, applies contrast clipping (limit = 2.5), and redistributes clipped pixels before bilinear interpolation.
- **Why it matters**: Enhances subtle security micro-printing, numerals, and emblems without amplifying noise in uniform background areas.

### Stage 6: Canny Edge Detection
- Multi-stage optimal edge detection:
  1. Computes directional gradients using Sobel operators ($G_x, G_y$).
  2. Calculates gradient magnitude $|G| = \sqrt{G_x^2 + G_y^2}$ and angle $\theta = \arctan(G_y / G_x)$.
  3. Non-Maximum Suppression (NMS) thins edges to 1-pixel ridges along the gradient direction.
  4. Hysteresis thresholding ($T_{\text{lower}} = 50, T_{\text{upper}} = 150$) links weak edges to strong edges.

### Stage 7: Contour Detection & Corner Localization
- Detects external topological contours via border following (`cv2.findContours`).
- Filters candidates by minimum area (> 5% of total image area) and simplifies polygon geometry using the Ramer-Douglas-Peucker algorithm (`cv2.approxPolyDP` with $\epsilon = 0.02 \times \text{perimeter}$).
- Locates 4 convex corners forming a quadrilateral boundary.

### Stage 8: Perspective Correction (Four-Point Transform)
- Solves for the $3 \times 3$ homography transformation matrix $\mathbf{H}$ using `cv2.getPerspectiveTransform`:
  $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- Warps the quadrilateral region to a fronto-parallel view using `cv2.warpPerspective`.
- **Fallback**: If 4 distinct convex corners are not detected (e.g. note cropped or filling the entire frame), perspective correction is skipped safely and processing continues on the preprocessed frame.

### Stage 9: Feature Extraction & Matching
- Extracts multi-scale Oriented FAST corners and Rotated BRIEF (ORB) keypoints, enriched with Boosted Efficient Binary Local Image Descriptors (BEBLID) if available.
- Matches descriptors against pre-cached reference images using `cv2.BFMatcher(cv2.NORM_HAMMING)`.
- Applies **Lowe's Ratio Test**: accepts match $m$ only if $d(m) < 0.75 \times d(n)$ against the second-nearest neighbor $n$.
- Performs **Geometric Verification with RANSAC**: computes candidate homography matrix and counts inliers matching rigid planar geometry.
- Selects the denomination with the strongest inlier evidence; requires at least 8 RANSAC inliers or 15 filtered matches.

### Stage 10: Authenticity & Specimen / Defacement Inspection
- **Specimen / Cancellation Stamp Detection**: Performs color-space segmentation in HSV space over the watermark and RBI signature field ($x \in [55\%, 96\%], y \in [14\%, 72\%]$). Genuine ₹500, ₹100, ₹200, and ₹10 notes feature no saturated red ink in this region. If a red stamp contour cluster or diagonal box (such as a red **"SPECIMEN"** or **"CANCELLED"** stamp) is detected ($> 1.2\%$ coverage or connected stamp components), the note is flagged as `SPECIMEN NOTE (NOT LEGAL TENDER)`.
- **Texture Sharpness (Photocopy Detection)**: Computes Laplacian variance $\sigma^2(\nabla^2 I)$. Blurry or photocopied notes exhibit low sharpness ($< 28.0$) compared to genuine intaglio-printed notes.
- **Color Profile Correlation**: Computes 2D Hue-Saturation histogram correlation ($r$) against the genuine reference template.
- **RANSAC Inlier Density**: Validates that planar feature correspondences conform to rigid homography.
- **Final Verdicts**:
  - 🟢 **GENUINE (AUTHENTIC CURRENCY)**: Passes all security checks; zero specimen markings.
  - ⚠️ **SPECIMEN / CANCELLED NOTE**: Features specimen/void stamp or dummy markers; not valid legal tender.
  - ❌ **SUSPICIOUS / LIKELY FAKE**: Fails texture, color correlation, or geometric inlier thresholds.

---

## 3. Directory Structure

```
RupeeVision/
├── app.py                # Complete Flask backend with all 10 IVP stages
├── index.html            # Complete user interface (Upload, Camera, Speech, Visualizer)
├── requirements.txt      # Python dependencies
├── README.md             # Technical documentation and viva preparation guide
└── Dataset/              # Reference dataset directory
    ├── 10/               # Reference images for ₹10
    ├── 20/               # Reference images for ₹20
    ├── 50/               # Reference images for ₹50
    ├── 100/              # Reference images for ₹100
    ├── 200/              # Reference images for ₹200
    └── 500/              # Reference images for ₹500
```

---

## 4. Setup and Installation

### Prerequisites
- Python 3.8+ (Python 3.9 recommended)
- macOS, Linux, or Windows

### Step 1: Create and Activate Virtual Environment
```bash
# Navigate to project directory
cd /Users/yadnyesh8250/Desktop/MINI_Project_sem5/IVP_MINI_PROJECT/RupeeVision

# Create virtual environment (if not already created)
python3 -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# (Or on Windows:)
# venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the Application
```bash
python app.py
```
Output:
```
=================================================================
  RupeeVision — Indian Currency Recognition & IVP Analysis
=================================================================
[INFO] Loaded reference dataset: 90 images across 6 denominations.
[INFO] Starting local Flask development server...
[INFO] Open your web browser at: http://127.0.0.1:5000
```

### Step 4: Open in Web Browser
Open your browser at `http://127.0.0.1:5000`.

---

## 5. How to Use the Application

1. **Upload Mode**: Click the upload box or drag-and-drop any photo of an Indian banknote (₹10, ₹20, ₹50, ₹100, ₹200, ₹500).
2. **Webcam Capture Mode**: Switch to the "Webcam Capture" tab, grant camera access, position your banknote, and click **Capture Still Frame**.
3. **Analyze Currency**: Click **Analyze Currency**.
4. **Inspect Results**:
   - The recognized denomination badge will illuminate (e.g. ₹500).
   - If audio announcement is enabled, the browser will announce the denomination aloud via Speech Synthesis.
   - The **IVP Processing Pipeline Visualizer** displays all intermediate OpenCV stages with technical explanations.

---

## 6. How Reference Images Work

The `Dataset/` folder contains clean, flat reference images for each denomination:
- Each folder (`Dataset/10/`, `Dataset/500/`, etc.) holds representative samples.
- During initialization, `app.py` loads these references, standardizes their resolution, extracts keypoints/descriptors, and caches them in memory.
- When a query note is analyzed, its descriptors are matched against these cached templates in milliseconds.

---

## 7. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web interface (`index.html`) |
| `/api/health` | `GET` | Returns JSON status, OpenCV version, and loaded dataset count |
| `/api/analyze` | `POST` | Accepts multipart image upload or base64 JSON payload; returns complete 10-stage IVP results |

---

## 8. College Viva / Exam Q&A Preparation

**Q1: Why did you use classical computer vision instead of a deep-learning CNN model?**
> *Answer*: Deep learning acts as a black box that predicts a class label without exposing intermediate representations. For an Image & Video Processing (IVP) course, our objective is to demonstrate core spatial filters, gradient operators, histogram transformations, topological contours, projective geometry, and feature descriptors at every step of the pipeline.

**Q2: What is the purpose of Gaussian Filtering before Canny Edge Detection?**
> *Answer*: Edge detection relies on gradient derivatives. Numerical differentiation amplifies high-frequency noise. Applying a Gaussian low-pass filter suppresses noise before gradient calculation, preventing spurious edge artifacts.

**Q3: How does Lowe's Ratio Test filter feature matches?**
> *Answer*: For each query descriptor, Lowe's test compares the Euclidean/Hamming distance of the closest match ($m$) to that of the second closest match ($n$). If $d(m) < 0.75 \times d(n)$, the match is accepted as distinctive; ambiguous or repetitive matches are rejected.

**Q4: How does RANSAC eliminate false matches?**
> *Answer*: RANSAC (Random Sample Consensus) iteratively selects a minimal subset of 4 point correspondences to estimate a prospective homography matrix, then calculates how many remaining matches agree with this planar transformation within a 5-pixel error threshold. Points that do not conform are rejected as outliers.
