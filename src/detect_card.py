"""
Card detection module for Playing Card Recognizer.
Detects cards in images, applies perspective transform, and extracts patches.
"""

import cv2
import numpy as np
from recognize_patch import PatchRecognizer


class CardDetector:
    """
    Detects playing cards in images and extracts standardized views.
    
    Provides methods to find card contours, apply perspective transforms,
    and extract rank and suit patches.
    """
    
    def __init__(self, card_width=500, card_height=726):
        """
        Initialize card detector with target dimensions.
        
        Args:
            card_width: Width for standardized card (default: 200)
            card_height: Height for standardized card (default: 300)
        """
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
        
        # Initialize tracking variables
        self.track_window = None
        self.roi_hist = None
        self.tracking_active = False
    
    def preprocess_image(self, frame):
        """
        Apply preprocessing steps to prepare for card detection.
        
        Args:
            frame: Input color image
            
        Returns:
            tuple: (gray, blurred, edges, color_frame) processed images
        """
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
        """
        Find the contour of a playing card in the edge-detected image.
        
        Args:
            edges: Edge-detected image
            min_area: Minimum contour area to consider (default: 20000)
            max_area: Maximum contour area to consider (default: 200000)
            
        Returns:
            tuple: (contour, corners) of the detected card, or (None, None) if not found
        """
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
        """
        Reorder corners to be [top-left, top-right, bottom-right, bottom-left].
        
        Args:
            pts: Array of 4 corner points
            
        Returns:
            numpy.ndarray: Reordered points
        """
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
        """
        Apply perspective transform to get a standardized view of the card.
        
        Args:
            frame: Input image
            corners: Four corner points of the card
            
        Returns:
            numpy.ndarray: Warped card image (standardized size)
        """
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, self.dst_points)
        
        # Apply the transform
        warped = cv2.warpPerspective(frame, M, (self.card_width, self.card_height))
        
        return warped
    
    def extract_patches(self, warped_card):
        """
        Extract rank and suit patches from the standardized card (500x726).
        Returns: (rank_patch, suit_patch)
        """
        # Use the same coordinates as in template extraction
        RANK_BOX = (0, 5, 90, 85)   # (x1, y1, x2, y2)
        SUIT_BOX = (5, 86, 97, 180) # (x1, y1, x2, y2)
        rank_patch = warped_card[RANK_BOX[1]:RANK_BOX[3], RANK_BOX[0]:RANK_BOX[2]]
        suit_patch = warped_card[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]]
        return rank_patch, suit_patch
    
    def setup_tracking(self, frame, card_contour):
        """
        Set up CamShift tracking for the detected card.
        
        Args:
            frame: Input frame
            card_contour: Contour of the detected card
            
        Returns:
            bool: True if tracking was set up successfully
        """
        # Check if we have a valid contour
        if card_contour is None:
            self.tracking_active = False
            return False
        
        # Create a mask of the card
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [card_contour], 0, 255, -1)
        
        # Calculate initial tracking window (ROI)
        x, y, w, h = cv2.boundingRect(card_contour)
        self.track_window = (x, y, w, h)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram of the card region for tracking
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        self.roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.tracking_active = True
        return True
    
    def track_card(self, frame):
        """
        Track the card using CamShift algorithm.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (success, track_box, track_window)
        """
        if not self.tracking_active or self.track_window is None:
            return False, None, None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate back projection
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # Apply CamShift tracking
        ret, self.track_window = cv2.CamShift(dst, self.track_window, 
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        
        # Check if tracking is still good
        x, y, w, h = self.track_window
        if w <= 0 or h <= 0:
            self.tracking_active = False
            return False, None, None
        
        # Get the rotated rectangle from CamShift
        track_box = cv2.boxPoints(ret).astype(np.int32)
        
        return True, track_box, self.track_window
    
    def detect_card(self, frame, use_tracking=False):
        """
        Detect a card in the frame, with optional tracking.
        
        Args:
            frame: Input image
            use_tracking: Whether to use tracking for previously detected cards
            
        Returns:
            tuple: (success, warped_card, rank_patch, suit_patch, corners, color_frame)
        """
        # Try tracking if active
        if use_tracking and self.tracking_active:
            track_success, track_box, _ = self.track_card(frame)
            
            if track_success:
                color_frame = frame.copy()
                # Draw tracking result
                cv2.polylines(color_frame, [track_box], True, (0, 255, 0), 2)
                
                # Use tracking box corners for perspective transform
                corners = self.reorder_corners(track_box.astype(np.float32))
                warped_card = self.warp_card(frame, corners)
                rank_patch, suit_patch = self.extract_patches(warped_card)
                
                return True, warped_card, rank_patch, suit_patch, corners, color_frame
        
        # Full detection if tracking failed or not used
        gray, blurred, edges, color_frame = self.preprocess_image(frame)
        card_contour, corners = self.find_card_contour(edges)
        
        if card_contour is None:
            return False, None, None, None, None, color_frame
        
        # If tracking is enabled, set it up with the new detection
        if use_tracking:
            self.setup_tracking(frame, card_contour)
        
        # Draw contour on color frame
        cv2.drawContours(color_frame, [card_contour], 0, (0, 255, 0), 2)
        
        # Warp the card to a standardized view
        warped_card = self.warp_card(frame, corners)
        
        # Extract rank and suit patches
        rank_patch, suit_patch = self.extract_patches(warped_card)
        
        return True, warped_card, rank_patch, suit_patch, corners, color_frame


def main():
    """Simple test function for the CardDetector class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the CardDetector module')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--track', action='store_true', help='Enable card tracking')
    args = parser.parse_args()
    
    # Import from the capture module
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.capture import Capture
    
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
            frame, success = capture.get_frame()
            if not success:
                print("Failed to capture frame")
                break

            card_success, warped_card, rank_patch, suit_patch, corners, debug_frame = \
                detector.detect_card(frame, use_tracking=args.track)

            fps = capture.get_fps()
            cv2.putText(debug_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            status = "Card Detected" if card_success else "No Card Detected"
            cv2.putText(debug_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Recognition and overlay results
            rank_label, suit_label = None, None
            if card_success:
                # Preprocess patches to match template preprocessing
                def preprocess_patch(patch):
                    gray = patch if len(patch.shape) == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, 2)
                rank_patch_proc = preprocess_patch(rank_patch)
                suit_patch_proc = preprocess_patch(suit_patch)
                rank_label, rank_score, _ = rank_recognizer.recognize(rank_patch_proc)
                suit_label, suit_score, _ = suit_recognizer.recognize(suit_patch_proc)
                # DEBUG: Save the extracted patches for visual inspection
                cv2.imwrite("debug_live_rank_patch.png", rank_patch_proc)
                cv2.imwrite("debug_live_suit_patch.png", suit_patch_proc)
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
