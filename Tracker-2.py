import cv2
from tracker import *
import random

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("C:/Users/mmabo/Downloads/Media10-10.AVI")

object_detector = cv2.createBackgroundSubtractorMOG2(history=900, varThreshold=80)
# Create a dictionary to store colors for each tracker ID
colors = {
    0: (255, 0, 0),  # Magenta
    1: (0, 0, 255),  # Red
    2: (255, 0, 0),  # Magenta
    3: (255, 0, 0),  # Red
    4: (255, 0, 0),  # Magenta
    5: (0, 0, 255),  # Magenta
    6: (0, 0, 255),  # Magenta
    7: (0, 0, 255),  # Magenta
    8: (0, 0, 255),  # Magenta
    9: (0, 0, 255),  # Magenta
    10: (0, 0, 255),  # Magenta
    11: (0, 0, 255),  # Magenta
    12: (0, 0, 255),  # Magenta
    13: (0, 0, 255),  # Magenta
    14: (0, 255, 0),  # Magenta
    15: (0, 0, 255),  # Magenta
    16: (0, 0, 255),  # Magenta
    17: (0, 0, 255),  # Magenta
    18: (0, 0, 255),  # Magenta
    19: (255, 0, 0),  # Magenta
    20: (255, 0, 0),  # Magenta
    21: (255, 0, 0),  # Magenta
    22: (255, 0, 0),  # Magenta
    23: (255, 0, 0),  # Magenta
    24: (255, 0, 0),  # Magenta
    25: (255, 0, 0),  # Magenta
    26: (255, 0, 0),  # Magenta
    27: (255, 0, 0),  # Magenta
    28: (255, 0, 0),  # Magenta
    29: (255, 0, 0),  # Magenta
    30: (255, 0, 0),  # Magenta
    31: (255, 0, 0),  # Magenta
    32: (255, 0, 0),  # Magenta
    33: (255, 0, 0),  # Magenta
    34: (255, 0, 0),  # Magenta
    35: (255, 0, 0),  # Magenta
    36: (255, 0, 0),  # Magenta
    37: (255, 0, 0),  # Magenta
    38: (255, 0, 0),  # Magenta
    39: (255, 0, 0),  # Magenta
    40: (255, 0, 0),  # Magenta
    41: (0, 0, 255),  # Magenta
    42: (0, 0, 255),  # Magenta
    43: (0, 0, 255),  # Magenta
    44: (255, 0, 0),  # Magenta
    45: (255, 0, 0),  # Magenta
    46: (255, 0, 0),  # Magenta
    47: (255, 0, 0),  # Magenta
    48: (255, 0, 0),  # Magenta
    49: (255, 0, 0),  # Magenta
    50: (255, 0, 0),  # Magenta
    # Add more colors as needed 
}
i = 0 # for image saving
skip = 1 # skipping for plotting the images
count = 0
center_points = {}
center_points_cur_frame = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    count += 1

    roi = frame[0:1000,0:1000]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:
            # cv2.drawContours(roi, [cnt], 0, (0, 255, 255), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        # If the tracker ID is not in the colors dictionary, add it with a random color
        if id not in colors:
            colors[id] = (int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))

        # Only add the point to the list if the current frame is a multiple of skip
        if count % skip == 0:
            if id not in center_points:
                center_points[id] = []
            center_points[id].append((cx, cy))

    # Only draw the points if the current frame is a multiple of 5
    if count % 1 == 0:
        for id, points in center_points.items():
            for pt in points:
                cv2.circle(roi, pt, 6, colors[id], 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    i += 1
    #cv2.imwrite('C:/Users/mmabo/Downloads/Track-net5/' + str(i) + '.png', roi)
    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()