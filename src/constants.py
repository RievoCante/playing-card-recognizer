# Patch extraction box coordinates for 500x726 card images
# Used for cropping rank and suit patches throughout the pipeline

RANK_BOX = (5, 10, 90, 85)   # (x1, y1, x2, y2)
SUIT_BOX = (5, 85, 97, 180)  # (x1, y1, x2, y2)
RANK_WIDTH = 70
RANK_HEIGHT = 125
SUIT_WIDTH = 70
SUIT_HEIGHT = 100
