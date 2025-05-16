# Playing Card Recognizer

A real-time playing card recognition system using classical computer vision methods.

## Description

This application captures webcam frames or processes images from a folder, detects a single playing card, and identifies its rank and suit using classical computer vision techniques. The pipeline includes card detection via edge and contour analysis, perspective transformation, patch extraction, and ORB-based feature matching. CamShift tracking is included for improved runtime efficiency.

## Features

- Real-time webcam input or sequential processing of images
- Card detection using edge detection and contour analysis
- Perspective transform to standardize card images
- Patch extraction for rank and suit
- ORB keypoint matching for rank and suit recognition
- CamShift tracking for stability (optional)
- Logging of detection results to CSV file

## Installation

### Prerequisites
- Python 3.11+
- Conda (Miniconda recommended)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n playing-card-recognizer python=3.11
conda activate playing-card-recognizer

# Install dependencies
conda install -c conda-forge opencv numpy pytest matplotlib
```

## Usage

All scripts are located in the `src/` folder.

### 1. Preprocess Templates
Generate and preprocess templates for rank and suit patches:
```bash
python src/preprocess_templates.py
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
