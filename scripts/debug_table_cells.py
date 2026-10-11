
from pathlib import Path

import cv2
import pytesseract
from PIL import Image


IMAGE_PATH = Path("debug_page_131.png")

TABLE_X = 2653
TABLE_Y = 5347
TABLE_W = 2877
TABLE_H = 1284

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise SystemExit("Page image not found.")

table = image[
    TABLE_Y:TABLE_Y + TABLE_H,
    TABLE_X:TABLE_X + TABLE_W,
]

gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    10,
)

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (80, 1),
)

vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, 80),
)

horizontal = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel,
)

vertical = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel,
)


def group_positions(positions, max_gap=12):
    groups = []

    for position in positions:
        if not groups or position - groups[-1][-1] > max_gap:
            groups.append([position])
        else:
            groups[-1].append(position)

    return [
        round(sum(group) / len(group))
        for group in groups
    ]


# Detect internal grid lines
row_positions = [
    i
    for i, value in enumerate(horizontal.sum(axis=1))
    if value > 100000
]

column_positions = [
    i
    for i, value in enumerate(vertical.sum(axis=0))
    if value > 100000
]

row_lines = group_positions(row_positions)
column_lines = group_positions(column_positions)

# Include outer boundaries of the cropped table
row_lines = sorted(set([0, *row_lines, table.shape[0]]))
column_lines = sorted(set([0, *column_lines, table.shape[1]]))

print("Row boundaries:", row_lines)
print("Column boundaries:", column_lines)

print("\n========== EXTRACTED TABLE ==========\n")

for row_index in range(len(row_lines) - 1):
    cells = []

    for col_index in range(len(column_lines) - 1):
        x1 = column_lines[col_index] + 8
        x2 = column_lines[col_index + 1] - 8
        y1 = row_lines[row_index] + 8
        y2 = row_lines[row_index + 1] - 8

        if x2 <= x1 or y2 <= y1:
            cells.append("")
            continue

        cell = table[y1:y2, x1:x2]

        rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)

        text = pytesseract.image_to_string(
            Image.fromarray(rgb),
            lang="deu",
            config="--psm 6",
        ).strip()

        cells.append(" ".join(text.split()))

    print(" | ".join(cells))

print("\nFinished. No RAG files were modified.")
