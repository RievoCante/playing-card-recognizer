# Recognize rank or suit patches by brute-force ORB matching to precomputed templates.

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
        
        # Load templates
        for fname in os.listdir(BASE_TEMPLATE_DIR):
            if fname.endswith('.npz') and fname.startswith(f'{template_type}_'):
                data = np.load(os.path.join(BASE_TEMPLATE_DIR, fname), allow_pickle=True)
                self.templates.append({
                    'label': data['label'].item(),
                    'image': data['image'],
                    'keypoints': data['keypoints'],
                    'descriptors': data['descriptors']
                })
                
        # Check if any templates were loaded
        if not self.templates:
            raise ValueError(f"No templates of type '{template_type}' found in {BASE_TEMPLATE_DIR}")
        
        # Initialize ORB and Brute Force matcher
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) #cv2.NORM_HAMMING for ORB

    def recognize(self, patch_img, min_matches=2):
        # Recognise the patch by matching to all templates.
        # Returns (best_label, best_score, details_dict)
        
        # Compute descriptors for input patch
        kp2, des2 = self.orb.detectAndCompute(patch_img, None)
        
        # Check if any features were found
        if des2 is None or len(kp2) == 0:
            return None, 0, {'reason': 'No features found in patch'}
        
        # Initialize variables
        best_label = None
        best_score = -1
        match_details = {}
        
        # Match descriptors
        for template in self.templates:
            des1 = template['descriptors']
            
            # Check if template has valid descriptors
            if des1 is None or not isinstance(des1, np.ndarray) or des1.ndim != 2 or des1.shape[0] == 0:
                continue
            
            # Match descriptors using brute-force matching
            matches = self.matcher.match(des1, des2)
            
            # Sort matches by distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Count the number of matches and put it as Score
            score = len(matches)
            
            # Update best match if this template has more matches
            if score > best_score:
                best_score = score
                best_label = template['label']
                match_details = {
                    'label': best_label,
                    'score': best_score,
                    'total_matches': len(matches),
                    'good_matches': score
                }
                
        return best_label, best_score, match_details

# Example usage:
# rank_recognizer = PatchRecognizer('rank')
# label, score, details = rank_recognizer.recognize(rank_patch)