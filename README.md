***

# Cylindrical Bottle Label Stitching

This project provides a robust Python pipeline to stitch multiple images of a rotating cylindrical object (like a bottle) into a single, flat panoramic image.

It is specifically engineered to handle common computer vision challenges associated with bottle inspection, including **low-light environments** and **conical distortion** (tapered bottles).

## 📂 Project Structure

*   `cropping_script.py`: A GUI tool to manually select the label region to exclude background noise. Generates a JSON config.
*   `stitching_script.py`: The main logic that straightens the bottle geometry, enhances contrast, matches features (SIFT), and stitches the images.
*   **Dataset 1** (`1.jpg` - `4.jpg`): A dataset (4 images).
*   **Dataset 2** (`im1.jpg` - `im5.jpg`): Another dataset (5 images).

## ✨ Key Features

1.  **Robust Feature Matching (CLAHE):** Includes Contrast Limited Adaptive Histogram Equalization to detect SIFT features even in extremely dark or low-contrast images (Fixed the issue with Dataset 2).
2.  **Conical Taper Correction:** Solves the "Banana Effect" or "Ghosting" at the bottom of the label. It mathematically unrolls tapered bottles where the bottom radius is smaller than the top radius.
3.  **Affine Stitching:** Uses Affine transformations rather than standard Homography to prevent severe distortion on rotating cylinders.

## 🛠️ Prerequisites

You need Python installed along with OpenCV and NumPy.

```bash
pip install opencv-python numpy
```

---

## 🚀 How to Use

### Step 1: Configure the Image Set
Open `cropping_script.py` and `stitching_script.py` in your text editor. You need to tell the scripts which dataset you want to process.

**For Dataset 1 (Bright):**
```python
IMAGE_FILES = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
OUTPUT_JSON = "crop_config.json"
```

**For Dataset 2 (Dark):**
```python
IMAGE_FILES = ["im1.jpg", "im2.jpg", "im3.jpg", "im4.jpg", "im5.jpg"]
OUTPUT_JSON = "crop_config3.json"
```

### Step 2: Generate Crop Configuration
Run the cropping tool to isolate the bottle from the background (conveyor belt, walls, etc.).

```bash
python cropping_script.py
```

1.  A window will open for the first image.
2.  **Draw a box** tightly around the bottle label (exclude the background!).
3.  Press **SPACE** or **ENTER** to confirm the crop.
4.  Repeat for all images.
5.  A JSON file (e.g., `crop_config.json`) will be saved.

### Step 3: Stitch the Images
Run the stitching script.

```bash
python stitching_script.py
```

The script will:
1.  Load the images and the JSON crop configuration.
2.  Apply **Taper Correction** (Straighten the cone).
3.  Apply **CLAHE** (Enhance text visibility).
4.  Stitch the images sequentially.
5.  Save the result (e.g., `stitch_img1.jpg`).

---

## ⚙️ Tuning for the Bottle (The "Taper" Fix)

If the bottle is not a perfect cylinder (i.e., it is wider at the top than the bottom), standard stitching will fail at the bottom.

Inside `stitching_script.py`, adjust the `TAPER_AMOUNT` variable:

```python
# 0.0  = Perfectly Cylindrical
# 0.10 = Bottom is 10% narrower than top
# 0.15 = Bottom is 15% narrower than top
TAPER_AMOUNT = 0.132 
```

*   **Shadows/Ghosting at the bottom?** The bottom is moving too slowly. **Increase** `TAPER_AMOUNT` (e.g., try `0.15`).
*   **Jittering/Misalignment at the bottom?** The bottom is moving too fast. **Decrease** `TAPER_AMOUNT` (e.g., try `0.10`).


## 📝 Troubleshooting

*   **"Warning: Low matches for image X":**
    *   Ensure your crop is tight (don't include background).
    *   Ensure the images have enough overlap (at least 30-40%).
*   **Stitched image looks curved:**
    *   This is a limitation of 2D stitching on 3D objects, but `TAPER_AMOUNT` usually fixes the alignment issues.