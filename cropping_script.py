import cv2
import json
import os

# Define your image list
IMAGE_FILES = ["im1.jpg", "im2.jpg", "im3.jpg", "im4.jpg", "im5.jpg"]
OUTPUT_JSON = "crop_config3.json"

def generate_crops():
    crop_data = {}

    print("--- CROP CONFIGURATION TOOL ---")
    print("1. A window will open for each image.")
    print("2. Draw a box tightly around the bottle.")
    print("3. Press SPACE or ENTER to confirm the selection.")
    print("4. Press 'c' to cancel/skip an image.")
    print("-------------------------------")

    for filename in IMAGE_FILES:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue

        img = cv2.imread(filename)
        h_img, w_img = img.shape[:2]

        # Resize for display if image is too large for screen (optional)
        # We use a scale factor to ensure coordinates map back correctly
        scale = 1.0
        display_img = img
        if h_img > 1000:
            scale = 1000 / h_img
            display_img = cv2.resize(img, (0,0), fx=scale, fy=scale)

        print(f"Select ROI for {filename}...")
        
        # Open ROI Selector
        # Note: If you resized, the ROI returned is scaled
        roi = cv2.selectROI(f"Crop: {filename}", display_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(f"Crop: {filename}")

        # roi is (x, y, w, h)
        x, y, w, h = roi

        # If user pressed 'c' or selected nothing
        if w == 0 or h == 0:
            print(f"   Skipped {filename}")
            continue

        # Map back to original size if we scaled
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        # Convert to Top/Bottom/Left/Right format
        # This format is safer for images of slightly varying sizes
        crop_top = y
        crop_bottom = h_img - (y + h)
        crop_left = x
        crop_right = w_img - (x + w)

        crop_data[filename] = {
            "top": crop_top,
            "bottom": crop_bottom,
            "left": crop_left,
            "right": crop_right
        }
        print(f"   Saved config for {filename}")

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(crop_data, f, indent=4)
    
    print(f"\nSUCCESS: Configuration saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_crops()