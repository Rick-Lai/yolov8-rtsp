import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')

cap = cv2.VideoCapture('url')

# 建立視窗
cv2.namedWindow('CAM_display', cv2.WINDOW_FREERATIO)
cv2.resizeWindow('CAM_display', 800, 450)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start = time.time()

        # Run YOLOv8 inference on the frame
        results = model(frame, device=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        end = time.time()
        fps = str(1/(end - start))
        
        # Display fps on frame
        cv2.putText(annotated_frame, fps, (650, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the annotated frame
        cv2.imshow("CAM_display", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()