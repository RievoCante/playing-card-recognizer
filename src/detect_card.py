# Card detection module for Playing Card Recognizer.
# Entry point of this app

import cv2
import numpy as np
import argparse
import os
import sys
from recognize_patch import PatchRecognizer
from capture import Capture

from constants import RANK_BOX, SUIT_BOX

class CardDetector:
    """
    Detects playing cards in images and extracts standardized views.
    Provides methods to find card contours, apply perspective transforms,
    and extract rank and suit patches.
    """
    
    def __init__(self, card_width = 500, card_height = 726):
        # Initialize card detector with target dimensions.
        self.card_width = card_width
        self.card_height = card_height
        
        # Define the destination points for the perspective transform
        # These represent the corners of our standardized card size
        self.dst_points = np.array([
            [0, 0],                        # Top-left
            [card_width, 0],               # Top-right
            [card_width, card_height],     # Bottom-right
            [0, card_height]               # Bottom-left
        ], dtype=np.float32)
    
    def preprocess_image(self, frame):
        # Apply preprocessing steps to prepare for card detection.
        
        # Make a copy of the original frame for drawing results later
        color_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return gray, blurred, edges, color_frame
    
    def find_card_contour(self, edges, min_area=20000, max_area=200000):
        # Find the contour of a playing card in the edge-detected image.

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours were found
        if not contours:
            return None, None
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for a quadrilateral contour that might be a card
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If the contour has 4 corners, we consider it a card
            if len(approx) == 4:
                # Reorder points in [top-left, top-right, bottom-right, bottom-left]
                corners = self.reorder_corners(approx.reshape(4, 2))
                return contour, corners
        
        return None, None
    
    def reorder_corners(self, pts):
        # Reorder corners to be [top-left, top-right, bottom-right, bottom-left].
        # pts: Array of 4 corner points
        # numpy.ndarray: Reordered points
        
        # Create a new array for the reordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Calculate the sum and difference of x and y coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has the smallest sum of coordinates
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point has the largest sum of coordinates
        rect[2] = pts[np.argmax(s)]
        # Top-right point has the smallest difference between coordinates
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left point has the largest difference between coordinates
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def warp_card(self, frame, corners):
        # Apply perspective transform to get a standardized view of the card.
        # frame: Input image
        # numpy.ndarray: Warped card image (standardized size)
        
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, self.dst_points)
        
        # Apply the transform
        warped = cv2.warpPerspective(frame, M, (self.card_width, self.card_height))
        
        return warped
    
    def extract_patches(self, warped_card):
        # Extract and refine rank and suit patches using the robust pipeline
        from constants import RANK_BOX, SUIT_BOX, RANK_WIDTH, RANK_HEIGHT, SUIT_WIDTH, SUIT_HEIGHT
        # --- Rank Patch ---
        rank_crop = warped_card[RANK_BOX[1]:RANK_BOX[3], RANK_BOX[0]:RANK_BOX[2]]
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
        # --- Suit Patch ---
        suit_crop = warped_card[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]]
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
        return rank_patch, suit_patch
    
    def detect_card(self, frame):
        # Detect a card in the frame, with optional tracking.
        
        gray, blurred, edges, color_frame = self.preprocess_image(frame)
        card_contour, corners = self.find_card_contour(edges)
        
        if card_contour is None:
            return False, None, None, None, None, color_frame
        
        # Draw contour on color frame
        cv2.drawContours(color_frame, [card_contour], 0, (0, 255, 0), 2)
        
        # Warp the card to a standardized view
        warped_card = self.warp_card(frame, corners)
        
        # Extract rank and suit patches
        rank_patch, suit_patch = self.extract_patches(warped_card)
        
        return True, warped_card, rank_patch, suit_patch, corners, color_frame


def main():
    # Simple test function for the CardDetector class. 
    
    parser = argparse.ArgumentParser(description='Test the CardDetector module')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # Initialize capture from webcam or folder
        if args.folder:
            capture = Capture(folder_path=args.folder)
        else:
            capture = Capture(source=args.camera)
            
        # Initialize card detector
        detector = CardDetector()
        
        # Initialize patch recognizers (once)
        rank_recognizer = PatchRecognizer('rank')
        suit_recognizer = PatchRecognizer('suit')

        while True:
            # Get frame
            frame, success = capture.get_frame()
            if not success:
                print("Failed to capture frame")
                break

            # Detect card
            card_success, warped_card, rank_patch, suit_patch, corners, debug_frame = \
                detector.detect_card(frame)

            # Show FPS
            fps = capture.get_fps()
            cv2.putText(debug_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show status
            status = "Card Detected" if card_success else "No Card Detected"
            cv2.putText(debug_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Recognition and overlay results
            rank_label, suit_label = None, None
            
            if card_success:
                
                # Preprocess patches to match template preprocessing
                def preprocess_patch(patch):
                    # Match the robust pipeline: grayscale, blur, fixed threshold, contour, crop, resize
                    from constants import RANK_WIDTH, RANK_HEIGHT, SUIT_WIDTH, SUIT_HEIGHT
                    is_rank = patch.shape[1] == RANK_WIDTH or patch.shape[0] == RANK_HEIGHT
                    gray = patch if len(patch.shape) == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    _, thresh = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY_INV)
                    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                        x, y, w, h = cv2.boundingRect(cnts[0])
                        roi = thresh[y:y+h, x:x+w]
                        if is_rank:
                            out = cv2.resize(roi, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)
                        else:
                            out = cv2.resize(roi, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)
                    else:
                        if is_rank:
                            out = cv2.resize(thresh, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_AREA)
                        else:
                            out = cv2.resize(thresh, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_AREA)
                    return out
                
                # Preprocessed patches
                rank_patch_proc = preprocess_patch(rank_patch)
                suit_patch_proc = preprocess_patch(suit_patch)
                
                # Recognize patches
                rank_label, rank_score, _ = rank_recognizer.recognize(rank_patch_proc)
                suit_label, suit_score, _ = suit_recognizer.recognize(suit_patch_proc)
                display_text = f"{rank_label or '?'} of {suit_label or '?'}"
                
                # Overlay at the top-left corner of the detected card
                if corners is not None:
                    pt = tuple(np.int32(corners[0]))  # top-left corner
                    # Offset to avoid drawing over the edge
                    pt = (pt[0] + 5, pt[1] + 30)
                    cv2.putText(debug_frame, display_text, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    # Fallback: overlay at fixed position
                    cv2.putText(debug_frame, display_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Only show main camera window
            cv2.imshow("Card Detection", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if 'capture' in locals():
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
