import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"

IMAGE_PATH = "datasets/test/images/frame1125_jpg.rf.30fa3d580be0daf1150d04fddabaa8a8.jpg"

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH)

counts = {}
for result in results:
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1

annotated_image = results[0].plot()

y_offset = 30
for label, count in counts.items():
    text = f"{label}: {count}"
    cv2.putText(annotated_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_offset += 30

cv2.imshow("Vehicle Detection & Counting", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()