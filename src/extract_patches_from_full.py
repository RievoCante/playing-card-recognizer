# Script to auto-extract rank and suit patches from full card images using the same pipeline as the recognizer.
# Saves patches to templates/patch/ranks/ and templates/patch/suits/ with ground truth labels from filenames.
import os
import cv2
import numpy as np
# Script to auto-extract rank and suit patches from full card images using fixed cropping.
# Crops top-left for rank, just below for suit, from 500x726 images. Adjust coordinates if needed.
import os
import cv2

FULL_DIR = os.path.join(os.path.dirname(__file__), '../templates/full')
RANK_OUT = os.path.join(os.path.dirname(__file__), '../templates/patch/ranks')
SUIT_OUT = os.path.join(os.path.dirname(__file__), '../templates/patch/suits')

os.makedirs(RANK_OUT, exist_ok=True)
os.makedirs(SUIT_OUT, exist_ok=True)

def parse_label(filename):
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    for sep in ['_of_', '-of-', ' of ']:
        if sep in name:
            rank, suit = name.split(sep)
            return rank, suit
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return name, ''

def main(preview=False):
    # These coordinates are for 500x726 images. Adjust as needed!
    RANK_BOX = (0, 5, 90, 85)   # (x1, y1, x2, y2)
    SUIT_BOX = (5, 86, 97, 180)  # (x1, y1, x2, y2)
    files = [f for f in os.listdir(FULL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    total, extracted = 0, 0
    for fname in sorted(files):
        img_path = os.path.join(FULL_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {fname}")
            continue
        gt_rank, gt_suit = parse_label(fname)
        total += 1
        rank_patch = img[RANK_BOX[1]:RANK_BOX[3], RANK_BOX[0]:RANK_BOX[2]]
        suit_patch = img[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]]
        if preview:
            # Show preview for debugging
            preview_img = img.copy()
            # cv2.rectangle(preview_img, (RANK_BOX[0], RANK_BOX[1]), (RANK_BOX[2], RANK_BOX[3]), (0,255,0), 2)
            cv2.rectangle(preview_img, (SUIT_BOX[0], SUIT_BOX[1]), (SUIT_BOX[2], SUIT_BOX[3]), (255,0,0), 2)
            cv2.imshow("Preview", preview_img)
            # cv2.imshow("Rank Patch", rank_patch)
            cv2.imshow("Suit Patch", suit_patch)
            print(f"Previewing {fname}. Press any key for next, or 'q' to quit.")
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            cv2.destroyAllWindows()
        # Save patches
        rank_path = os.path.join(RANK_OUT, f"{gt_rank}.png")
        suit_path = os.path.join(SUIT_OUT, f"{gt_suit}.png")
        cv2.imwrite(rank_path, rank_patch)
        cv2.imwrite(suit_path, suit_patch)
        print(f"Extracted: {fname} -> {rank_path}, {suit_path}")
        extracted += 1
    print(f"\nExtracted patches from {extracted}/{total} images.")

if __name__ == "__main__":
    # Set preview=True to visually check the crop on a few cards before extracting all
    main(preview=False)
