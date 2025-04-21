# Playing Card Recognizer: Step-by-Step Todo List

## 1. Project Setup

- [x] ~~_*Create project directory structure:*_~~ [2025-04-21]

  ```
  playing-card-recognizer/
  ├── src/
  │   ├── __init__.py
  │   ├── capture.py
  │   ├── detect_card.py
  │   ├── recognise_patch.py
  │   ├── dl_verify.py
  │   ├── ui_overlay.py
  │   └── run.py
  ├── templates/
  ├── tests/
  │   ├── __init__.py
  │   ├── test_capture.py
  │   ├── test_detect_card.py
  │   ├── test_recognise_patch.py
  │   └── test_dl_verify.py
  ├── results/
  └── README.md
  ```

- [x] ~~_Set up conda environment:_~~ [2025-04-21]

- [x] ~~_Install required packages:_~~ [2025-04-21]

  ```bash
  conda install -c conda-forge opencv numpy pytest matplotlib
  pip install pyinstaller
  ```

- [x] ~~_Create basic README.md with project overview and setup instructions_~~ [2025-04-21]

## 2. Data Collection and Preparation

- [ ] Collect reference playing card templates:

  - [ ] 13 rank templates (Ace through King)
  - [ ] 4 suit templates (Hearts, Diamonds, Clubs, Spades)

- [ ] Preprocess reference templates:

  - [ ] Crop to include only the rank/suit symbol
  - [ ] Resize to standardized dimensions
  - [ ] Save as PNGs in templates/ directory

- [ ] Prepare test dataset:
  - [ ] Capture or collect ~30 cropped card images for testing
  - [ ] Ensure variety in lighting conditions and angles

## 3. Core Implementation

### 3.1 Frame Capture Module (`capture.py`)

- [ ] Implement webcam frame capture functionality
- [ ] Add support for reading images from a folder
- [ ] Create a generic frame provider class with common interface

### 3.2 Card Detection Module (`detect_card.py`)

- [ ] Implement grayscale conversion
- [ ] Add Gaussian blur for noise reduction
- [ ] Implement Canny edge detection
- [ ] Add contour detection and filtering for quadrilaterals
- [ ] Implement perspective transform to standardize card size (200x300 pixels)
- [ ] Extract rank and suit patches from the top-left corner

### 3.3 Recognition Module (`recognise_patch.py`)

- [ ] Implement ORB keypoint detection for patches
- [ ] Create brute-force matcher with Lowe ratio test
- [ ] Add template matching against reference templates
- [ ] Calculate confidence scores for matches

### 3.4 Tracking Module (part of `detect_card.py`)

- [ ] Implement CamShift tracking for detected cards
- [ ] Add fallback to full detection when tracking fails
- [ ] Create a tracking toggle mechanism

### 3.5 Deep Learning Verification (Optional) (`dl_verify.py`)

- [ ] Set up MobileNetV2 model structure
- [ ] Configure Apple Silicon optimizations (MPS backend for PyTorch or Metal for TensorFlow)
- [ ] Implement inference function for verification
- [ ] Create a verification toggle mechanism

### 3.6 UI Overlay Module (`ui_overlay.py`)

- [ ] Implement overlay for detection results (rank, suit)
- [ ] Add confidence score visualization
- [ ] Create FPS counter and display
- [ ] Implement feedback for failed detections

### 3.7 Main Application (`run.py`)

- [ ] Set up command-line interface with argparse
- [ ] Implement main application loop
- [ ] Add logging functionality to CSV
- [ ] Create performance metrics tracking

## 4. Testing

- [ ] Write unit tests for each module:

  - [ ] `test_capture.py`
  - [ ] `test_detect_card.py`
  - [ ] `test_recognise_patch.py`
  - [ ] `test_dl_verify.py`

- [ ] Perform sanity testing with test dataset
- [ ] Generate confusion matrix for recognition accuracy
- [ ] Measure and log FPS performance

## 5. Optimization

- [ ] Profile code to identify bottlenecks
- [ ] Optimize for Apple Silicon using hardware acceleration
- [ ] Implement multi-threading where beneficial
- [ ] Fine-tune parameters for best performance/accuracy balance

## 6. Documentation and Finalization

- [ ] Complete inline code documentation
- [ ] Update README with detailed usage instructions
- [ ] Create example usage scripts
- [ ] Document known limitations and troubleshooting

## 7. Packaging and Deployment

- [ ] Use PyInstaller to create standalone executable:

  ```bash
  pyinstaller --onefile --add-data "templates:templates" src/run.py -n playing-card-recognizer
  ```

- [ ] Test executable on target hardware (MacBook Pro with M4)
- [ ] Create release package with executable and documentation
- [ ] Document deployment instructions in README

## 8. Project Presentation

- [ ] Prepare demonstration video
- [ ] Generate performance metrics and visualizations
- [ ] Compile test results into report
- [ ] Document potential improvements and extensions

## Advanced Extensions (Optional)

- [ ] Implement multiple card detection capability
- [ ] Add support for different card designs
- [ ] Create a graphical user interface
- [ ] Implement real-time statistics for poker hand evaluation
