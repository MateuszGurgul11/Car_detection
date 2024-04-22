from ultralytics import YOLO
import cv2
import math 
import cvzone
from sort import *

cap = cv2.VideoCapture('cars.mp4')

model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [360, 300, 1000, 320]

while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    results = model(img_region, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            current_class = classNames[cls]

            if current_class == 'car' or current_class == 'truck' or current_class == 'bus' or current_class == 'motorbike' and conf > 0.3: 
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5)
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=8)

    cv2.imshow("img", img)
    cv2.imshow("img_region", img_region)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()