# Script to auto-extract rank and suit patches from full card images using the same pipeline as the recognizer.
# Saves patches to templates/patch/ranks/ and templates/patch/suits/ with ground truth labels from filenames.
import os
import cv2
import numpy as np

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
    from constants import RANK_BOX, SUIT_BOX
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
        from constants import RANK_WIDTH, RANK_HEIGHT, SUIT_WIDTH, SUIT_HEIGHT
        # Extract initial patches
        rank_crop = img[RANK_BOX[1]:RANK_BOX[3], RANK_BOX[0]:RANK_BOX[2]]
        suit_crop = img[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]]

        # --- Refine Rank Patch ---
        rank_gray = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY) if len(rank_crop.shape) == 3 else rank_crop
        rank_blur = cv2.GaussianBlur(rank_gray, (5,5), 0)
        _, rank_thresh = cv2.threshold(rank_blur, 155, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(rank_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(cnts[0])
            rank_roi = rank_thresh[y:y+h, x:x+w]
            rank_patch = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            rank_patch = cv2.resize(rank_thresh, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)

        # --- Refine Suit Patch ---
        suit_gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY) if len(suit_crop.shape) == 3 else suit_crop
        suit_blur = cv2.GaussianBlur(suit_gray, (5,5), 0)
        _, suit_thresh = cv2.threshold(suit_blur, 155, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(suit_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(cnts[0])
            suit_roi = suit_thresh[y:y+h, x:x+w]
            suit_patch = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            suit_patch = cv2.resize(suit_thresh, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)

        if preview:
            # Show preview for debugging
            preview_img = img.copy()
            cv2.rectangle(preview_img, (RANK_BOX[0], RANK_BOX[1]), (RANK_BOX[2], RANK_BOX[3]), (0,255,0), 2)
            cv2.rectangle(preview_img, (SUIT_BOX[0], SUIT_BOX[1]), (SUIT_BOX[2], SUIT_BOX[3]), (255,0,0), 2)
            cv2.imshow("Preview", preview_img)
            cv2.moveWindow("Preview", 100, 100)
            cv2.imshow("Refined Rank Patch", rank_patch)
            cv2.moveWindow("Refined Rank Patch", 650, 100)
            cv2.imshow("Refined Suit Patch", suit_patch)
            cv2.moveWindow("Refined Suit Patch", 1200, 100)
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
