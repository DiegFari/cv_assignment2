import cv2
import numpy as np
import os


# PART 1: CREATING THE BACKGROUND MODEL
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  

for cam_id in range(1, 5):
    print(f"Processing camera {cam_id}")
    video_path = os.path.join(SCRIPT_DIR, "data", f"cam{cam_id}", "background.avi") 

    # sampling 10 frames

    video = cv2.VideoCapture(video_path)

    frames = []
    step = 20
    frame_idx = 0
    max_frames_used = 10


    while True:

        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frames.append(frame)

        frame_idx += 1

        if len(frames) >= max_frames_used:
            break

    video.release()
    print("Sampled frames:", len(frames))

    # getting the mean of them 

    mean = np.mean(frames, axis=0)          
    mean_uint8 = np.clip(mean, 0, 255).astype(np.uint8)

    cv2.imwrite(f"background_mean{cam_id}.png", mean_uint8)

    background = cv2.imread("background_mean.png") 

    while True:
        ret, frame = video.read()
        if not ret:
            break

        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # threshold: this is used to deicde which pixels are part of the background (clack) or of the foreground (white) based on a treshold 30
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # cleaning up noise trhough opencv functions (OPEN cleans up noise, CLOSE fills gaps)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    video.release()
    cv2.destroyAllWindows()


# PART 2: LOADING THE BACKGROUND MODEL AND ...

background_bgr = cv2.imread("background.png")
background_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
Hb, Sb, Vb = cv2.split(background_hsv)

for cam_id in range(1,5):

    vid = cv2.VideoCapture("video.avi")
    ret, frame_bgr = vid.read()

    

