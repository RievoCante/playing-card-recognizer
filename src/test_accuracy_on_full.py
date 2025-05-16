# Tests recognition accuracy on all full card images in templates/full.

import os
import cv2
from detect_card import CardDetector
from recognize_patch import PatchRecognizer

FULL_DIR = os.path.join(os.path.dirname(__file__), '../templates/full')
from constants import RANK_BOX, SUIT_BOX

# Preprocess patch to match template preprocessing
# Converts to grayscale (if needed) and applies adaptive thresholding
# Used for both rank and suit patches
def preprocess_patch(patch):
    gray = patch if len(patch.shape) == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

# Helper to parse ground truth from filename, e.g., 'ace_of_spades.png' -> ('ace', 'spades')
def parse_label(filename):
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    # Accepts 'ace_of_spades', '10_of_hearts', etc.
    for sep in ['_of_', '-of-', ' of ']:
        if sep in name:
            rank, suit = name.split(sep)
            return rank, suit
    # fallback: try splitting by first '_'
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return name, ''

def main():
    detector = CardDetector()
    rank_recognizer = PatchRecognizer('rank')
    suit_recognizer = PatchRecognizer('suit')

    files = [f for f in os.listdir(FULL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    total, correct = 0, 0
    
    for fname in sorted(files):
        img_path = os.path.join(FULL_DIR, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Could not read {fname}")
            continue
        gt_rank, gt_suit = parse_label(fname)
        
        # Preprocess the full image (grayscale + adaptive threshold)
        gray_full = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pre_full = cv2.adaptiveThreshold(gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        
        # Use same crop as template extraction
        rank_patch = frame[RANK_BOX[1]:RANK_BOX[3], RANK_BOX[0]:RANK_BOX[2]]
        suit_patch = frame[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]]
        rank_patch = preprocess_patch(rank_patch)
        suit_patch = preprocess_patch(suit_patch)
        
        # Save preprocessed patches for visual inspection in debug_images directory
        debug_dir = os.path.join(os.path.dirname(__file__), '../debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"debug_test_rank_{fname}"), rank_patch)
        cv2.imwrite(os.path.join(debug_dir, f"debug_test_suit_{fname}"), suit_patch)
        
        # Debug: Print number of ORB keypoints in each patch
        rank_kp, _ = rank_recognizer.orb.detectAndCompute(rank_patch, None)
        suit_kp, _ = suit_recognizer.orb.detectAndCompute(suit_patch, None)
        print(f"[DEBUG] {fname}: rank_kp={len(rank_kp)}, suit_kp={len(suit_kp)}")
        
        pred_rank, _, _ = rank_recognizer.recognize(rank_patch)
        pred_suit, _, _ = suit_recognizer.recognize(suit_patch)
        is_correct = (pred_rank == gt_rank and pred_suit == gt_suit)
        print(f"{fname:30} | GT: {gt_rank:8} {gt_suit:8} | Pred: {pred_rank or '?':8} {pred_suit or '?':8} | {'CORRECT' if is_correct else 'WRONG'}")
        total += 1
        if is_correct:
            correct += 1
    print(f"\nAccuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    main()