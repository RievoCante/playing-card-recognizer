# Visual inspection of extracted rank or suit patches from full card images
# NOT a part of main app, just for testing

import os
import cv2
from constants import RANK_BOX, SUIT_BOX, RANK_WIDTH, RANK_HEIGHT, SUIT_WIDTH, SUIT_HEIGHT

# PATCH_TYPE can be 'rank' or 'suit'
PATCH_TYPE = 'suit'  # Change to 'suit' to inspect suit patches

# Directory containing full card images
FULL_DIR = os.path.join(os.path.dirname(__file__), '../templates/full')
image_files = [f for f in os.listdir(FULL_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for idx, fname in enumerate(sorted(image_files)[:10]):  # Show first 10 images
    img_path = os.path.join(FULL_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {fname}")
        continue
    # Extract patch from full card image
    if PATCH_TYPE == 'rank':
        box = RANK_BOX
        width, height = RANK_WIDTH, RANK_HEIGHT
        patch_label = fname.split('_')[0].lower()  # e.g., '10' from '10_of_clubs.png'
        template_dir = os.path.join(os.path.dirname(__file__), '../templates/patch/ranks')
    else:
        box = SUIT_BOX
        width, height = SUIT_WIDTH, SUIT_HEIGHT
        patch_label = fname.split('_')[-1].split('.')[0].lower()  # e.g., 'clubs' from '10_of_clubs.png'
        template_dir = os.path.join(os.path.dirname(__file__), '../templates/patch/suits')
    patch_crop = img[box[1]:box[3], box[0]:box[2]]
    # --- Preprocess patch as in extract_patches_from_full.py ---
    patch_gray = cv2.cvtColor(patch_crop, cv2.COLOR_BGR2GRAY) if len(patch_crop.shape) == 3 else patch_crop
    patch_blur = cv2.GaussianBlur(patch_gray, (5,5), 0)
    _, patch_thresh = cv2.threshold(patch_blur, 155, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(patch_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(cnts[0])
        patch_roi = patch_thresh[y:y+h, x:x+w]
        patch_disp = cv2.resize(patch_roi, (width, height), interpolation=cv2.INTER_AREA)
    else:
        patch_disp = cv2.resize(patch_thresh, (width, height), interpolation=cv2.INTER_AREA)
    # Find the corresponding template patch
    template_patch = None
    for ext in ['.png', '.jpg', '.jpeg']:
        template_path = os.path.join(template_dir, f"{patch_label}{ext}")
        if os.path.exists(template_path):
            template_patch = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            break
    if template_patch is None:
        print(f"No template patch found for {patch_label}")
        continue
    # Resize to same height for side-by-side display
    h = max(patch_disp.shape[0], template_patch.shape[0])
    patch_disp = cv2.resize(patch_disp, (int(patch_disp.shape[1] * h / patch_disp.shape[0]), h), interpolation=cv2.INTER_AREA)
    template_patch_disp = cv2.resize(template_patch, (int(template_patch.shape[1] * h / template_patch.shape[0]), h), interpolation=cv2.INTER_AREA)
    # Concatenate side by side
    combined = cv2.hconcat([patch_disp, template_patch_disp])
    cv2.imshow(f"{PATCH_TYPE.capitalize()} Patch (left) vs Template (right): {fname}", combined)
    print(f"Showing {fname} (left, preprocessed) and template {patch_label} (right). Press any key for next, or 'q' to quit.")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord('q'):
        break
