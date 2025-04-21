# Playing Card Recognizer

A real-time playing card recognition system using classical computer vision methods with optional deep learning verification.

## Description

This application captures webcam frames in real-time, detects a single playing card, and identifies its rank and suit using classical computer vision techniques. It includes CamShift tracking for stability and an optional lightweight deep learning model for verification.

## Features

- Real-time webcam input or sequential processing of images
- Edge detection and contour analysis for card detection
- Perspective transform to standardize card images
- ORB keypoint matching for rank and suit recognition
- CamShift tracking for efficiency and stability
- Optional deep learning verification
- Live UI overlay with detection results
- Logging of detection results to CSV file

## Installation

### Prerequisites

- Python 3.11+
- Conda (Miniconda)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n playing-card-recognizer python=3.11
conda activate playing-card-recognizer

# Install dependencies
conda install -c conda-forge opencv numpy pytest matplotlib
pip install pyinstaller
```

## Usage

```bash
# Run with webcam input
python src/run.py --webcam

# Run with images from a folder
python src/run.py --folder path/to/images

# Run with deep learning verification
python src/run.py --webcam --dl_verify

# Run with tracking enabled
python src/run.py --webcam --track

# Save detection logs
python src/run.py --webcam --save_log results/detection_log.csv
```

## Project Structure

```
playing-card-recognizer/
├── src/                 # Source code
├── templates/           # Reference card templates
├── tests/               # Unit and integration tests
├── results/             # Logs and performance metrics
└── README.md            # Project documentation
```

## License

[MIT](LICENSE)
