import cv2
import imutils
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
cap = cv2.VideoCapture("2.mp4")
frame_count = 0

# Loading images
img_red = cv2.imread("1.jpg")
img_green = cv2.imread("3.jpg")

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    # Process every 10th frame
    frame_count += 1
    if frame_count % 10 != 0:
        continue
    
    img = imutils.resize(img, width=600)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    vehicles = 0
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Count number of vehicles
                label = str(classes[class_id])
                if label in ["car", "bus", "truck", "bicycle"]:
                    vehicles += 1

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Show appropriate traffic light image
    if vehicles > 7:
        cv2.imshow("GREEN", img_green)
        cv2.waitKey(2000)
        cv2.destroyWindow("GREEN")
    else:
        cv2.imshow("RED", img_red)
        cv2.waitKey(2000)
        cv2.destroyWindow("RED")

    # show the output frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
