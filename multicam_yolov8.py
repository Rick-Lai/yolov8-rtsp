from ultralytics import YOLO
# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv

cams = [
    ['cam_name', 'cam_url'],
    ['cam2_name', 'cam2_url'],
    ['cam3_name', 'cam3_url']
]

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

def start(cams):
    urls = [cv.VideoCapture(c[1]) for c in cams]
    for i in range(len(urls)):
        urls[i].set(cv.CAP_PROP_BUFFERSIZE, 0)
    names = [c[0] for c in cams]
    for i in range(len(names)):
        cv.namedWindow(names[i], cv.WINDOW_FREERATIO)
        cv.resizeWindow(names[i], 540, 540)
    
    while True:
        for i in range(len(cams)):
        # 從 RTSP 串流讀取一張影像
            ret, frame = urls[i].read()
            if ret:     
                # Run YOLOv8 inference on the frame
                results = model(frame, device=0)   
                # Visualize the results on the frame
                annotated_frame = results[0].plot() 
                # 顯示影像
                cv.imshow(names[i], annotated_frame)
            else:
                print(names[i], 'no signal')
                break

        #cv.waitKey(1) & press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    #釋放cam資源
    for u in urls:
        u.release()    
    # 關閉所有 OpenCV 視窗
    cv.destroyAllWindows()
    print("stopped")

# def record():

if __name__ == '__main__':
    start(cams)