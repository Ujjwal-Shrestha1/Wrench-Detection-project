from ultralytics import YOLO
import numpy as np
import cv2
import math
import itertools

# Paths
model_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/runs/segment/train12/weights/last.pt'
image_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/data/images/val/pic13_Color.png'
#image_path = r"C:\Users\shres\OneDrive\Pictures\Camera Roll\WIN_20250313_14_56_08_Pro.jpg"
output_path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/data/output_contour_PCA.png'

# Load image
img = cv2.imread(image_path)
H, W, _ = img.shape

# Load YOLO model
model = YOLO(model_path)

# Perform inference
results = model(img)



finalbbox = []
ratio = 19 / 529

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def find_rectangle_length_width(points):
    """Find the length (longer side) and width (shorter side) of a rotated rectangle."""
    distances = [(euclidean_distance(p1, p2), p1, p2) for p1, p2 in itertools.combinations(points, 2)]
    distances.sort(key=lambda x: x[0])
    unique_lengths = sorted(set(d[0] for d in distances[:4]))
    if len(unique_lengths) < 2:
        raise ValueError("Points do not form a proper rectangle")
    width, length = unique_lengths[0], unique_lengths[-1]  # -1 for longest side
    return length, width, unique_lengths

def get_orientation(contour):
    """Calculate the orientation of a contour using OpenCV PCA."""
    data_pts = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    center = mean.flatten()
    major_axis = eigenvectors[0]
    angle = math.degrees(math.atan2(major_axis[1], major_axis[0]))
    if angle < 0:
        angle += 180
    return angle, center, major_axis

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
            if len(contours) > 1:
                # Merge multiple contours into one
                merged_contour = np.vstack(contours)  # Stack all contour points vertically
                print(f"  Mask {i} - Merged {len(contours)} contours into one with {merged_contour.shape[0]} points")
                
                # Process merged contour
                rect = cv2.minAreaRect(merged_contour)
                bbox = cv2.boxPoints(rect)
                bbox = np.int32(bbox)
                print(f"  Mask {i} merged bounding box: {bbox}")
                finalbbox.append([bbox])

                # Draw contour and bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.drawContours(img, contours, -1, color, 2)  # Draw original contours
                cv2.drawContours(img, [bbox], -1, (255, 0, 0), 2)  # Draw merged bbox

                # Calculate length and width from merged bbox
                box = bbox
                length, width, _ = find_rectangle_length_width(box)
                print(f"  Mask {i} merged length: {length:.2f}")
                print(f"  Mask {i} merged width: {width:.2f}")

                # Add length text at top-left of bounding box
                cv2.putText(img, str(int(length * ratio)), tuple(box[0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # PCA for orientation using merged contour
                angle, center, major_axis = get_orientation(merged_contour)
                print(f"  Mask {i} merged orientation: {angle:.2f} degrees")
                
                # Draw the major axis from the center
                scale = length / 2
                end_point = (int(center[0] + scale * major_axis[0]), 
                             int(center[1] + scale * major_axis[1]))
               

            elif len(contours) == 1:
                # Process single contour as before
                contour = contours[0]
                print(f"  Contour 0 has {contour.shape[0]} points")
                
                rect = cv2.minAreaRect(contour)
                bbox = cv2.boxPoints(rect)
                bbox = np.int32(bbox)
                print(f"  Contour 0 bounding box: {bbox}")
                finalbbox.append([bbox])

                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.drawContours(img, [contour], -1, color, 2)
                cv2.drawContours(img, [bbox], -1, (255, 0, 0), 2)

                box = bbox
                length, width, _ = find_rectangle_length_width(box)
                print(f"  Contour 0 length: {length:.2f}")
                print(f"  Contour 0 width: {width:.2f}")

                cv2.putText(img, str(int(length * ratio)), tuple(box[0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                angle, center, major_axis = get_orientation(contour)
                print(f"  Contour 0 orientation: {angle:.2f} degrees")
                
                scale = length / 2
                end_point = (int(center[0] + scale * major_axis[0]), 
                             int(center[1] + scale * major_axis[1]))
                

            else:
                print(f"  Mask {i} - No contours found")

# Save the output image
cv2.imwrite(output_path, img)

# Display the result
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()