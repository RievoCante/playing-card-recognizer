# Preprocesses card template images to extract and save ORB features for fast recognition.

import os
import cv2
import numpy as np
from constants import RANK_BOX, SUIT_BOX

# Configuration
BASE_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '../templates')
RANKS_DIR = os.path.join(BASE_TEMPLATE_DIR, 'patch/ranks')
SUITS_DIR = os.path.join(BASE_TEMPLATE_DIR, 'patch/suits')
OUTPUT_DIR = os.path.join(BASE_TEMPLATE_DIR, 'precomputed')
RANK_SIZE = (RANK_BOX[2] - RANK_BOX[0], RANK_BOX[3] - RANK_BOX[1])  # (width, height)
SUIT_SIZE = (SUIT_BOX[2] - SUIT_BOX[0], SUIT_BOX[3] - SUIT_BOX[1])  # (width, height)

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']


def get_template_type_and_label(filename, ttype):
    """Infer label from filename and type from folder."""
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    return ttype, name




# Preprocess template image to match extract_patches_from_full.py pipeline
def process_image(image_path, ttype, save_png_dir=None, out_label=None):
    from constants import RANK_WIDTH, RANK_HEIGHT, SUIT_WIDTH, SUIT_HEIGHT
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    # Convert to grayscale if needed
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    # Fixed threshold
    _, img_thresh = cv2.threshold(img_blur, 155, 255, cv2.THRESH_BINARY_INV)
    # Contour detection
    cnts, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(cnts[0])
        roi = img_thresh[y:y+h, x:x+w]
        if ttype == 'rank':
            img_final = cv2.resize(roi, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            img_final = cv2.resize(roi, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        # Fallback: resize the thresholded image
        if ttype == 'rank':
            img_final = cv2.resize(img_thresh, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            img_final = cv2.resize(img_thresh, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)
    # Optionally save as PNG for visual inspection
    if save_png_dir is not None and out_label is not None:
        os.makedirs(save_png_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_png_dir, f"{out_label}.png"), img_final)
    return img_final

def compute_orb_features(img):
    """Compute ORB keypoints and descriptors for the image."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    # Keypoints must be converted to a serializable format (list of tuples)
    if keypoints is not None:
        keypoints_as_list = [kp.pt + (kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
    else:
        keypoints_as_list = []
    return keypoints_as_list, descriptors


def save_precomputed(label, ttype, img, keypoints, descriptors):
    """Save precomputed data as .npz file in OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{ttype}_{label}.npz")
    # Convert keypoints to numpy object array for safe saving
    keypoints_np = np.array(keypoints, dtype=object)
    np.savez_compressed(
        out_path,
        label=label,
        type=ttype,
        image=img,
        keypoints=keypoints_np,
        descriptors=descriptors
    )
    print(f"Saved: {out_path}")


def process_folder(folder, ttype, size):
    if not os.path.exists(folder):
        print(f"Warning: {folder} does not exist.")
        return
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
    if not files:
        print(f"No template images found in {folder}.")
        return
    for fname in files:
        try:
            label = os.path.splitext(fname)[0].lower()
            img = process_image(os.path.join(folder, fname), ttype)
            keypoints, descriptors = compute_orb_features(img)
            save_precomputed(label, ttype, img, keypoints, descriptors)
        except Exception as e:
            print(f"Error processing {fname} in {folder}: {e}")


def main():
    process_folder(RANKS_DIR, 'rank', RANK_SIZE)
    process_folder(SUITS_DIR, 'suit', SUIT_SIZE)


if __name__ == "__main__":
    main()
