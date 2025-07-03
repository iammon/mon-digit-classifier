import cv2
import os

# ======= CONFIGURATION =======
INPUT_IMAGE = "mon_sheets/Mon_zero.jpg"
OUTPUT_FOLDER = "mon_segmented_raw/၀"
RESIZE_TO = (28, 28)  # Match MNIST size
PADDING = 4           # Add white border around digits

# ======= Ensure output folder exists =======
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======= Load and preprocess image =======
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ======= Find contours (digit blobs) =======
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ======= Sort contours by top-to-bottom, left-to-right =======
def sort_contours(cnts):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    return [c for _, c in sorted(zip(bounding_boxes, cnts), key=lambda b: (b[0][1], b[0][0]))]

contours = sort_contours(contours)

# ======= Loop through each contour and save cropped digits =======
count = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        continue  # Skip small specks

    roi = thresh[y:y+h, x:x+w]

    # Add padding
    roi = cv2.copyMakeBorder(roi, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=0)

    # Resize to MNIST format
    digit = cv2.resize(roi, RESIZE_TO, interpolation=cv2.INTER_AREA)

    # Save image
    filename = os.path.join(OUTPUT_FOLDER, f"digit_{count:03}.png")
    cv2.imwrite(filename, digit)
    count += 1

print(f">> Done. {count} digit(s) saved to '{OUTPUT_FOLDER}'")
