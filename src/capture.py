# This handles video capture from webcam or image folder

import os
import time
import cv2 as cv
import numpy as np

class Capture:
    def __init__(self, source=0, folder_path=None) -> None:
        self.source = source
        self.folder_path = folder_path
        self.image_files = []
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.current_image_index = 0
        
        
        if folder_path and os.path.isdir(folder_path):
            # Get list of image files in the folder
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            self.image_files.sort()
            if not self.image_files:
                raise ValueError(f"No valid image files found in {folder_path}")
            print(f"Found {len(self.image_files)} images in {folder_path}")
        else:
            # Initialize webcam
            self.cap = cv.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source {source}")
            print(f"Webcam initialized with index {source}")
        
    # Get the next frame from the video source or image folder.
    def get_frame(self):
        self.frame_count += 1
        
        # Calculate FPS every 10 frames
        if self.frame_count % 10 == 0:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
        
        # If use images from folder
        if self.image_files:
            
            # Read from image files
            if self.current_image_index < len(self.image_files):
                image_path = self.image_files[self.current_image_index]
                frame = cv.imread(image_path)
                self.current_image_index += 1
                return frame, True # frame was successfully captured
            
            else:
                self.current_image_index = 0
                if self.current_image_index < len(self.image_files):
                    image_path = self.image_files[self.current_image_index]
                    frame = cv.imread(image_path)
                    self.current_image_index += 1
                    return frame, True
                return None, False
                
        # If use camera
        else:
            ret, frame = self.cap.read()
            return frame, ret
        
    def get_fps(self):
        return self.fps
    
    # Release resources    
    def release(self):
        if self.cap:
            self.cap.release()
        
        
# Test function for Capture class        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Capture module')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Display width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Display height (default: 480)')
    args = parser.parse_args()
    
    try:
        # Initialize capture from webcam or folder
        if args.folder:
            capture = Capture(folder_path=args.folder)
        else:
            capture = Capture(source=args.camera)
        
        while True:
            # Get frame
            frame, success = capture.get_frame()
            
            if not success:
                print('Failed to capture frame')
                break
            
            # Resize frame for display
            display_frame = cv.resize(frame, (args.width, args.height))
            
            # Display FPS
            fps = capture.get_fps()
            cv.putText(
                display_frame, f"FPS: {fps:.2f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Show frame
            cv.imshow("Capture Test", display_frame)
            
            # Break loop on 'q' key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # check if variable capture exists in local scope
        if 'capture' in locals():
            capture.release()
        cv.destroyAllWindows()
        

if __name__ == "__main__":
    main()
    