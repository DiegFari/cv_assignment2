import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  

# checks different combinations of thresholds against ground truth mask and returns the ones that have the least differences/mistakes. Uses cam1.
def auto_set_thresholds(mask, frame):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # there are some gray pixels in the mask because of how foreground removal works in gimp, so it was easier to force all pixels to be black/white like this
    _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    background_bgr = cv2.imread("background_mean1.png")
    background_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
    Hb, Sb, Vb = cv2.split(background_hsv)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Hf, Sf, Vf = cv2.split(frame_hsv)

    dH = cv2.absdiff(Hf, Hb)
    dH = cv2.min(dH, 180 - dH)

    dS = cv2.absdiff(Sf, Sb)
    dV = cv2.absdiff(Vf, Vb)

    # score = amount of mistakes, so lower is better
    best_score = 99999999999
    best_H = 0
    best_S = 0
    best_V = 0

    for H in range(0, 30, 2):
        for S in range(0, 80, 5):
            for V in range(0, 80, 5):
                maskH = (dH > H).astype(np.uint8) * 255
                maskS = (dS > S).astype(np.uint8) * 255
                maskV = (dV > V).astype(np.uint8) * 255

                test_mask = cv2.bitwise_or(maskH, maskS)
                test_mask = cv2.bitwise_or(test_mask, maskV)

                #255 = wrong, 0 = correct
                diff = cv2.bitwise_xor(test_mask, mask_gray)    
                score = np.sum(diff) / 255

                if score < best_score:
                    best_score = score
                    best_H = H
                    best_S = S
                    best_V = V

    return best_H, best_S, best_V                    


# PART 1: CREATING THE BACKGROUND MODEL
def create_background_models():
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

        print("Sampled frames:", len(frames))

        # getting the mean of them 

        mean = np.mean(frames, axis=0)          
        mean_uint8 = np.clip(mean, 0, 255).astype(np.uint8)

        cv2.imwrite(f"background_mean{cam_id}.png", mean_uint8)

        background = cv2.imread(f"background_mean{cam_id}.png") 

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


# PART 2: LOADING THE BACKGROUND MODEL AND DO BACKGROUND SUBSTRACTION
def run_background_subtraction(threshold_H, threshold_S, threshold_V):

    for cam_id in range(1,5):

        # background different channels 
        background_bgr = cv2.imread(f"background_mean{cam_id}.png")
        background_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
        Hb, Sb, Vb = cv2.split(background_hsv)

        # frame's different channels
        video_path = os.path.join(SCRIPT_DIR, "data", f"cam{cam_id}", "video.avi")
        vid = cv2.VideoCapture(video_path)

        while True:
            ret, frame_bgr = vid.read()
            if not ret:
                break
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            Hf, Sf, Vf = cv2.split(frame_hsv)

            # compute the differences
            dH = cv2.absdiff(Hf, Hb)
            dH = cv2.min(dH, 180 - dH)  # this is becayse H is circular so we do need to adjust

            dS = cv2.absdiff(Sf, Sb)
            dV = cv2.absdiff(Vf, Vb)

            # compute the masks by comparing with the thresholds 
            maskH = (dH > threshold_H).astype(np.uint8) * 255
            maskS = (dS > threshold_S).astype(np.uint8) * 255
            maskV = (dV > threshold_V ).astype(np.uint8) * 255

            # combining 
            mask = cv2.bitwise_or(maskH, maskS)
            mask = cv2.bitwise_or(mask, maskV)

            # postprocessing 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            cv2.imshow(f"cam{cam_id} frame", frame_bgr)
            cv2.imshow(f"cam{cam_id} mask", mask)

            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        vid.release()

        cv2.destroyAllWindows()


def main():

    # PART 1: CREATING THE BACKGROUND MODEL
    create_background_models()

    # defining thresholds (overwritten later)

    threshold_H = 10     # hue difference threshold (0..179, but circular)
    threshold_S = 40     # saturation difference threshold (0..255)
    threshold_V = 40     # value difference threshold (0..255)

    #automatic threshold detection (choice 2)

    test_frame = cv2.imread("cam1_frame.png")
    ground_truth_mask = cv2.imread("cam1_mask.png")
    threshold_H, threshold_S, threshold_V = auto_set_thresholds(ground_truth_mask, test_frame)
    print(f"Thresholds: h= {threshold_H},s= {threshold_S},v= {threshold_V}")

    run_background_subtraction(threshold_H, threshold_S, threshold_V)


if __name__ == "__main__":
    main()