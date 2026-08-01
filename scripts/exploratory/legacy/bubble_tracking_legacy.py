from __future__ import print_function
import cv2

# Initialize video capture
cap = cv2.VideoCapture("C:/Users/mmabo/Downloads/mask.avi")

# Skip the first two frames
for _ in range(10):
    success, frame = cap.read()
    if not success:
        print("Failed to read the video")
        sys.exit(1)

# Convert the frame to grayscale
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur
# blurred = cv2.GaussianBlur(gray, (1, 1), 0)

# # Apply threshold
# _, thresh = cv2.threshold(blurred, 100, 200, cv2.THRESH_BINARY)

# Let the user draw the first bounding box
bbox1 = cv2.selectROI('Select Object 1', frame, False)
cv2.destroyWindow('Select Object 1')

# Let the user draw the second bounding box
bbox2 = cv2.selectROI('Select Object 2', frame, False)
cv2.destroyWindow('Select Object 2')

# Initialize MultiTracker
multiTracker = cv2.MultiTracker_create()

# Add bounding boxes to the multi-tracker
multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox1)
multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox2)
# Initialize lists to store points
points1 = []
points2 = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Update tracker
    success, boxes = multiTracker.update(frame)

    # Draw tracked objects
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        center = (int(x + w / 2), int(y + h / 2))

        if i == 0:
            points1.append(center)
            color = (0, 255, 0)  # Green for the first tracker
        else:
            points2.append(center)
            color = (0, 0, 255)  # Red for the second tracker

        # Draw current point
        cv2.circle(frame, center, radius=5, color=color, thickness=-1)

        # Draw previous points
        if i == 0:
            for point in points1:
                cv2.circle(frame, point, radius=2, color=color, thickness=3)
        else:
            for point in points2:
                cv2.circle(frame, point, radius=2, color=color, thickness=3)

    cv2.imshow('MultiTracker', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()