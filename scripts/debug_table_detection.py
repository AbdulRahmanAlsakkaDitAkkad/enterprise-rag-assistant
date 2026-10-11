
from pathlib import Path

import cv2


IMAGE_PATH = Path("debug_page_131.png")
OUTPUT_PATH = Path("debug_detected_tables.png")

if not IMAGE_PATH.exists():
    raise SystemExit(f"Image not found: {IMAGE_PATH}")

# Load the page image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise SystemExit("Could not read page image.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to binary image
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    10,
)

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (80, 1),
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel,
)

# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, 80),
)

vertical_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel,
)

# Combine table lines
table_mask = cv2.add(
    horizontal_lines,
    vertical_lines,
)

# Find candidate table regions
contours, _ = cv2.findContours(
    table_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

detected = image.copy()
regions = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w < 250 or h < 100:
        continue

    regions.append((x, y, w, h))

regions.sort(key=lambda region: (region[1], region[0]))

for index, (x, y, w, h) in enumerate(regions, start=1):
    cv2.rectangle(
        detected,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        8,
    )

    cv2.putText(
        detected,
        f"Table {index}",
        (x, max(y - 15, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        5,
    )

cv2.imwrite(str(OUTPUT_PATH), detected)

print("Image dimensions:", image.shape)
print("Detected table regions:", len(regions))

for index, region in enumerate(regions, start=1):
    print(f"Table {index}: {region}")

print("Saved:", OUTPUT_PATH.resolve())
print("No RAG files were modified.")
