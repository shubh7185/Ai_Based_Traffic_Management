import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, cfg_path='yolov7.cfg', weights_path='yolov7.weights', names_path='coco.names'):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        with open(names_path, "r") as f:
            self.classes = f.read().strip().split("\n")
        self.vehicle_labels = ['car', 'bus', 'truck']

    def detect(self, image_np, draw_boxes=False, output_path=None):
        height, width, _ = image_np.shape
        blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        output_layers = self.net.getUnconnectedOutLayersNames()
        detections = self.net.forward(output_layers)

        detected_labels = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    label = self.classes[class_id]
                    if label in self.vehicle_labels:
                        detected_labels.append(label)
                        if draw_boxes:
                            center_x = int(obj[0] * width)
                            center_y = int(obj[1] * height)
                            w = int(obj[2] * width)
                            h = int(obj[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        if draw_boxes and output_path:
            cv2.imwrite(output_path, image_np)

        return detected_labels
