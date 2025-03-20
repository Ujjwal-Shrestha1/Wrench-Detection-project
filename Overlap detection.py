from ultralytics import YOLO
import numpy as np
import cv2

# Paths
model_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/runs/segment/train12/weights/last.pt'
image_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/data/images/val/pic5_Color.png'
output_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/data/output_overlapping_contours.png'

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")
H, W, _ = img.shape
print(f"Image shape: {img.shape}")

# Load YOLO model
model = YOLO(model_path)

# Perform inference
results = model(img)

final_cont = []

# Process results
for result in results:
    if result.masks is not None:
        print(f"Total number of masks detected: {len(result.masks.data)}")
        for i, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = mask.astype(np.uint8)
            print(f"Mask {i} shape before resize: {mask.shape}")
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H))

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Mask {i} - Number of contours found: {len(contours)}")

            # Handle contours based on number
            if len(contours) > 1:  # Specifically handles 2 or more contours
                # Merge multiple contours into one
                merged_contour = np.vstack(contours)  # Combine all contour points
                merged_contour = cv2.convexHull(merged_contour)
                peri = cv2.arcLength(merged_contour, True)
                approx = cv2.approxPolyDP(merged_contour, 0.02 * peri, True)
                print(f"  Mask {i} - Merged {len(contours)} contours into one with {merged_contour.shape[0]} points (approx: {approx.shape[0]} points)")

                # Get bounding box for merged contour
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # Draw merged contour and bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.drawContours(img, [merged_contour], -1, color, 2)  # Draw merged contour
                #cv2.drawContours(img, [approx], -1, color, 2)  # Optional: Draw approximated contour
                #cv2.drawContours(img, [box], -1, (0, 0, 255), 2)  # Optional: Draw bounding box
                print(f"  Mask {i} - Merged bounding box: {box}")

                final_cont.append(merged_contour)

            elif len(contours) == 1:
                # Process single contour
                contour = contours[0]
                print(f"  Contour 0 has {contour.shape[0]} points")

                # Get bounding box for single contour
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # Draw contour and bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.drawContours(img, [contour], -1, color, 2)
                #cv2.drawContours(img, [box], -1, (0, 0, 255), 2)  # Optional: Draw bounding box
                print(f"  Mask {i} - Single contour bounding box: {box}")

                final_cont.append(contour)

            else:
                print(f"  Mask {i} - No contours found")

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1 = np.min(box1, axis=0)
    x2, y2 = np.max(box1, axis=0)
    x3, y3 = np.min(box2, axis=0)
    x4, y4 = np.max(box2, axis=0)

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# Check if any contours in final_cont overlap
overlapping_pairs = []
for i in range(len(final_cont)):
    for j in range(i + 1, len(final_cont)):
        # Compute bounding boxes for each contour
        rect1 = cv2.minAreaRect(final_cont[i])
        box1 = cv2.boxPoints(rect1)
        box1 = np.int32(box1)

        rect2 = cv2.minAreaRect(final_cont[j])
        box2 = cv2.boxPoints(rect2)
        box2 = np.int32(box2)

        # Calculate IoU
        iou = calculate_iou(box1, box2)
        print(f"IoU between Contour {i} and Contour {j}: {iou:.2f}")
        if iou > 0.01:  # Threshold for overlap
            overlapping_pairs.append((i, j))
            print(f"Contours {i} and {j} overlap with IoU: {iou:.2f}")

# Draw bounding boxes around overlapping contours
for pair in overlapping_pairs:
    i, j = pair
    # Merge the overlapping contours
    merged_contour = np.vstack([final_cont[i], final_cont[j]])
    merged_contour = cv2.convexHull(merged_contour)

    # Draw a single bounding box around the merged contour
    rect = cv2.minAreaRect(merged_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(img, [box], -1, (0, 0, 255), 2)  # Red bounding box
    print(f"Merged bounding box for Contours {i} and {j}: {box}")

# Save the output image
cv2.imwrite(output_path, img)

# Display the result
cv2.imshow('Overlapping Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print final_cont for verification
print("\nFinal Contour Points:")
for idx, cont in enumerate(final_cont):
    print(f"Contour {idx}: {cont.reshape(-1, 2).tolist()}")