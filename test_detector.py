# test_detector.py
import cv2
import numpy as np
from detector import VehicleDetector

# Load test image
image_path = 'test_images/.jpg'  # Make sure this image exists
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or could not be read.")
    exit()

# Initialize the detector
detector = VehicleDetector(
    cfg_path='yolov7.cfg',
    weights_path='yolov7.weights',
    names_path='coco.names'
)

# Perform detection
detected_classes = detector.detect(image)
print("Detected Vehicles:", detected_classes)

# Check for ambulance
if detector.contains_ambulance(detected_classes):
    print("🚑 Ambulance detected!")
else:
    print("No ambulance in the image.")

# Optionally: Draw labels and save result
for label in detected_classes:
    cv2.putText(image, label, (30, 30 + 30 * detected_classes.index(label)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

output_path = "output_detection.jpg"
cv2.imwrite(output_path, image)
print(f"✅ Output saved as {output_path}. Open it to see the result.")
