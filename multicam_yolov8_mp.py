from ultralytics import YOLO
from multiprocessing import Process
import cv2 as cv

cams = [
    ['cam_name', 'cam_url'],
    ['cam2_name', 'cam2_url'],
    ['cam3_name', 'cam3_url']
]

model = YOLO('yolov8s.pt')

def open_cam(name, url):
    cam = cv.VideoCapture(url)
    cam.set(cv.CAP_PROP_BUFFERSIZE, 2)
    print(name + " start")
    cv.namedWindow(name, cv.WINDOW_FREERATIO)
    cv.resizeWindow(name, 540, 540)
    display(cam, name)

def detect_objs(frame):
    results = model(frame, device=0)   
    annotated_frame = results[0].plot()  
    return(annotated_frame)

def display(cam, name):
    while True:
        ret, frame = cam.read()
        if ret:    
            annotated_frame = detect_objs(frame)  
            cv.imshow(name, annotated_frame)
        else:
            print(name, 'no signal')
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    close(cam, name)

def close(cam, name):
    cam.release()    
    if not cam.isOpened():
        print(name + " is closed")   
    cv.destroyWindow(name)


if __name__ == '__main__':
    
    threads = [Process(target = open_cam, args = (name, url), daemon = True) for name, url in cams]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("program stopped")