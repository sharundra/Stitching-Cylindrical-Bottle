import json
import math
import cv2
import numpy as np
import os

# --- TUNABLE PARAMETER ---
# How much narrower is the bottom compared to the top?
# 0.0 = Perfectly Cylindrical (No change)
# 0.1 = Bottom is 10% narrower than top
# 0.2 = Bottom is 20% narrower than top (Try this for tapered bottles)
TAPER_AMOUNT = 0.132

# ---------------------------
# 1. Straighten the Taper (The Fix)
# ---------------------------
def straighten_cone(img, taper_percent):
    """
    Stretches the bottom of the image to match the width of the top.
    Assumes the crop is centered on the bottle.
    """
    if taper_percent <= 0:
        return img
    
    h, w = img.shape[:2]
    
    # Calculate how many pixels to "push in" the bottom corners in the Source
    # to map them to the full width in the Destination.
    # Actually, we want to take the NARROW bottom and stretch it to FULL width.
    
    # Source Points: The trapezoid shape of the bottle in the image
    # TL: (0,0)
    # TR: (w,0)
    # BL: (pixels_in, h)
    # BR: (w - pixels_in, h)
    
    delta_w = int(w * taper_percent / 2.0)
    
    src_pts = np.float32([
        [0, 0],                 # Top Left
        [w, 0],                 # Top Right
        [delta_w, h],           # Bottom Left (The actual bottle edge is inside)
        [w - delta_w, h]        # Bottom Right
    ])
    
    # Destination Points: A perfect rectangle
    dst_pts = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp
    # We maintain the same output size, effectively stretching the bottom pixels out
    straightened = cv2.warpPerspective(img, M, (w, h))
    return straightened

# ---------------------------
# Utility: cropping via JSON
# ---------------------------
def apply_json_crop(img, filename, config):
    if filename not in config:
        return img
    c = config[filename]
    h, w = img.shape[:2]
    top, bottom, left, right = c['top'], c['bottom'], c['left'], c['right']
    if top + bottom >= h or left + right >= w:
        return img
    return img[top:h-bottom, left:w-right]

# ---------------------------
# SIFT + Matching (With CLAHE Fix)
# ---------------------------
def detect_and_compute_sift(img, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for dark images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc

def match_descs(desc1, desc2, ratio_thresh=0.75): 
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    good = []
    for m_n in matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good

def find_affine_from_matches(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    A, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if A is None: return None
    return np.vstack([A, [0,0,1]])

# ---------------------------
# Pipeline logic
# ---------------------------
def compute_pairwise_homographies(images):
    n = len(images)
    sift = cv2.SIFT_create()
    kps_desc = [detect_and_compute_sift(im, sift) for im in images]
    pair_H = [None] * n 
    
    for i in range(1, n):
        kp2, desc2 = kps_desc[i]
        kp1, desc1 = kps_desc[i-1]
        good = match_descs(desc2, desc1)
        print(f"   Matches {i}->{i-1}: {len(good)}")
        if len(good) < 8:
            print(f"  Warning: Low matches for image {i}")
            pair_H[i] = None
            continue
        pair_H[i] = find_affine_from_matches(kp2, kp1, good)
    return pair_H

def compose_to_base(pair_H):
    n = len(pair_H)
    H_to_base = [None] * n
    H_to_base[0] = np.eye(3, dtype=np.float64)
    for i in range(1, n):
        if pair_H[i] is None or H_to_base[i-1] is None:
            H_to_base[i] = None
        else:
            H_to_base[i] = H_to_base[i-1] @ pair_H[i]
    return H_to_base

def stitch_images(images):
    # 1. Compute Alignment
    pair_H = compute_pairwise_homographies(images)
    H_to_base = compose_to_base(pair_H)
    
    # 2. Calculate Canvas Size
    all_corners = []
    valid_indices = []
    for i, img in enumerate(images):
        if H_to_base[i] is None: continue
        valid_indices.append(i)
        h, w = img.shape[:2]
        corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=np.float64).T
        corners_warp = H_to_base[i] @ corners
        corners_warp /= corners_warp[2:3,:]
        all_corners.append((corners_warp[0,:].min(), corners_warp[1,:].min(), 
                            corners_warp[0,:].max(), corners_warp[1,:].max()))
    
    if not all_corners: return None
    
    min_x = min([c[0] for c in all_corners])
    min_y = min([c[1] for c in all_corners])
    max_x = max([c[2] for c in all_corners])
    max_y = max([c[3] for c in all_corners])
    
    off_x = -min_x if min_x < 0 else 0
    off_y = -min_y if min_y < 0 else 0
    canvas_w = int(max_x + off_x)
    canvas_h = int(max_y + off_y)

    # 3. Warp and Blend
    acc_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    acc_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i in valid_indices:
        img = images[i]
        T = np.array([[1,0,off_x],[0,1,off_y],[0,0,1]], dtype=np.float64)
        H_final = T @ H_to_base[i]
        
        warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray_w > 0).astype(np.float32)
        
        # Simple blending weight (center usually better)
        # Using simple distance transform for blending mask
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        if dt.max() > 0: dt /= dt.max()
        
        for c in range(3):
            acc_image[:,:,c] += warped[:,:,c].astype(np.float32) * dt
        acc_weight += dt

    acc_weight[acc_weight == 0] = 1.0
    result = (acc_image / acc_weight[:,:,None]).astype(np.uint8)
    return result

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # IMAGE_FILES = ["im1.jpg", "im2.jpg", "im3.jpg", "im4.jpg", "im5.jpg"]
    IMAGE_FILES = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    JSON_FILE = "crop_config.json"
    OUTPUT_FILE = "stitch_img1.jpg"
    
    print("--- LOADING ---")
    images = []
    filenames = []
    for f in IMAGE_FILES:
        if os.path.exists(f):
            images.append(cv2.imread(f))
            filenames.append(f)
            
    # Load Crop Config
    config = {}
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            config = json.load(f)

    # PROCESS: Crop -> Straighten -> List
    processed_imgs = []
    for img, fname in zip(images, filenames):
        # 1. Crop Background
        cropped = apply_json_crop(img, fname, config)
        
        # 2. Straighten Cone (Fixes the bottom shadow issue)
        # Note: Only apply if image exists
        if cropped.size > 0:
            straightened = straighten_cone(cropped, TAPER_AMOUNT)
            processed_imgs.append(straightened)
            print(f"Processed {fname}: Cropped & Straightened (Taper={TAPER_AMOUNT})")

    print("\n--- STITCHING ---")
    result = stitch_images(processed_imgs)
    
    if result is not None:
        cv2.imwrite(OUTPUT_FILE, result)
        print(f"Saved: {OUTPUT_FILE}")
    else:
        print("Stitching failed.")