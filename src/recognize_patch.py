# Recognises rank or suit patches by brute-force ORB matching to precomputed templates.

import os
import cv2
import numpy as np

# Path to precomputed templates
BASE_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '../templates/precomputed')

class PatchRecognizer:
    def __init__(self, template_type):
        # Loads all precomputed templates of the given type ('rank' or 'suit') into memory.
        self.templates = []
        self.template_type = template_type
        for fname in os.listdir(BASE_TEMPLATE_DIR):
            if fname.endswith('.npz') and fname.startswith(f'{template_type}_'):
                data = np.load(os.path.join(BASE_TEMPLATE_DIR, fname), allow_pickle=True)
                self.templates.append({
                    'label': data['label'].item(),
                    'image': data['image'],
                    'keypoints': data['keypoints'],
                    'descriptors': data['descriptors']
                })
        if not self.templates:
            raise ValueError(f"No templates of type '{template_type}' found in {BASE_TEMPLATE_DIR}")
        print(f"Loaded {template_type} templates:", [t['label'] for t in self.templates]) # Debug: print loaded template labels
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def recognize(self, patch_img, min_matches=2, ratio=0.9):
        # Recognise the patch by matching to all templates.
        # Returns (best_label, best_score, details_dict)
        # Compute descriptors for input patch
        kp2, des2 = self.orb.detectAndCompute(patch_img, None)
        if des2 is None or len(kp2) == 0:
            return None, 0, {'reason': 'No features found in patch'}
        best_label = None
        best_score = -1
        match_details = {}
        for template in self.templates:
            des1 = template['descriptors']
            if des1 is None or not isinstance(des1, np.ndarray) or des1.ndim != 2 or des1.shape[0] == 0:
                continue
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # Lowe's ratio test
            good = [m for m, n in matches if m.distance < ratio * n.distance] if len(matches[0]) == 2 else []
            score = len(good)
            if score > best_score:
                best_score = score
                best_label = template['label']
                match_details = {
                    'label': best_label,
                    'score': best_score,
                    'total_matches': len(matches),
                    'good_matches': score
                }
        # Optionally, require a minimum number of matches
        if best_score < min_matches:
            return None, best_score, {'reason': 'Not enough good matches', **match_details}
        return best_label, best_score, match_details

# Example usage:
# rank_recognizer = PatchRecognizer('rank')
# label, score, details = rank_recognizer.recognize(rank_patch)