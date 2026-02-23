# Getting intrinsics of the cameras 

import os
print("CWD:", os.getcwd())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  

import cv2
import numpy as np
import glob

def get_manual_corners(image: np.ndarray, pattern_size: tuple[int, int]):

# This function implements the manual detection of the corners by clicking on the four external inner corners of the chessboard and performing a linear interpolation
# it returns all the interpolated corner grid as np.array

    original = image.copy() 
    instance = image.copy()

    clicked_points = []

    def click_event(event: int, x: int, y: int, flags: int, params):

    # This function stores the four clicks of the external inner corners. it was implemented modifying the tutorial by open cv 

        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4: # if there are 4 clickes the next clicks are ignored

            clicked_points.append((x,y))
            cv2.circle(instance, (x,y), 5, (0, 0, 255), -1) # drawing a little red circel in the click aqrea 

            cv2.imshow("manual corners", instance) # showind the image updated 

    cv2.namedWindow("manual corners", cv2.WINDOW_NORMAL) 
    cv2.imshow("manual corners", instance)

    cv2.setMouseCallback("manual corners", click_event) # this also follows the quoted tutorial 

    while True: # get all the four points

        key = cv2.waitKey(20) & 0xFF # waiting for the keyboard to press something

        if key == 27: # in case of ESC, we exit the program
            cv2.destroyWindow("manual corners")
            return None 
        
        if key == ord('d'):
            cv2.destroyWindow("manual corners")
            return None

        if key == ord('r'): # in case the user presser 'r', we reset the image 
            clicked_points.clear()
            instance[:] = original # resetting each instance pixel to the original 
            cv2.imshow("manual corners", instance)

        if len(clicked_points) == 4: # all the corners have been pressed
            break

    cv2.destroyWindow("manual corners") # once we collected all the corners, we can kill the window

    # here we do not order points but just store them in the right

    pts = np.array(clicked_points, dtype=np.float32)

    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    # interpolling the grid
    # pattern_size = (cols, rows)
    cols, rows = pattern_size

    grid = []
    
    for j in range(rows):
        for i in range(cols):
            u = i/(cols-1)
            v = j/(rows-1)
            top = (1 - u) * tl + u * tr
            bottom = (1 - u) * bl + u * br
            p = (1 - v) * top + v * bottom
            grid.append(p)


    corners = np.array(grid, dtype=np.float32)

    corners = corners.reshape(-1, 1, 2)

    vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    c0 = tuple(corners[0].ravel().astype(int))
    cv2.circle(vis, c0, 8, (0,0,255), -1)

    
    while True:
        cv2.imshow("Manual corners check (press q)", cv2.resize(vis, (960,540)))
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow("Manual corners check (press q)")

    print("TL:", tl, "TR:", tr, "BR:", br, "BL:", bl)
    return corners 

    # This function calibrates the camera and also provides an estimation of the intrinsic values (e.g. the standard deviation)
def do_calibration(name, obj, img):

        ret, mtx, dist, rvecs, tvecs, std_intr, std_ext, per_view_err = cv2.calibrateCameraExtended(obj, img, image_size, None, None)

        print(f"\n{name}")
        print("Images used:", len(obj))
        print("Camera matrix:\n", mtx)
        print("RMS reprojection error:", ret)
        print("Distortion:\n", dist.ravel())

        print("\nStandard deviation of intrinsic parameters:")
        print("fx std:", std_intr[0])
        print("fy std:", std_intr[1])
        print("cx std:", std_intr[2])
        print("cy std:", std_intr[3])

        # now we are gonna drop the 5 frames with the higest error to make calibration more stable and have a cleaner output

        errs = per_view_err.reshape(-1)
        worst_idx = np.argsort(errs)[-5:]

        # keep everything except those indices
        keep = [i for i in range(len(obj)) if i not in set(worst_idx)]
        obj2 = [obj[i] for i in keep]
        img2 = [img[i] for i in keep]

        ret2, mtx2, dist2, rvecs2, tvecs2, std_intr2, std_ext2, per_view_err2 = cv2.calibrateCameraExtended(obj2, img2, image_size, None, None)

        print("Images used:", len(obj2))
        print("Camera matrix:\n", mtx2)
        print("RMS reprojection error:", ret2)
        print("Distortion:\n", dist2.ravel())
       

        np.savez(f"{name}.npz",
                cameraMatrix=mtx2,
                distCoeffs=dist2,
                rvecs=rvecs2,
                tvecs=tvecs2,
                stdIntrinsics=std_intr2)
        print(f"Saved calibration to {name}.npz")

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.

pattern_size = (8, 6) # number of intern corners are squares -1, so for our chessboard is 9, 6
square_size = 0.115 # lenght of the squares converted to meters 


# getting frames of the video

for cam_id in range(1, 5):
    print(f"Processing camera {cam_id}")
    video_path = os.path.join(SCRIPT_DIR, "data", f"cam{cam_id}", "intrinsics.avi") 
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane
    # creating an empty 3d matrix of the chessboard 
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32) 
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) 
    objp *= square_size # here we scale for the square size (unlike the tutorial)


    images = []

    vid = cv2.VideoCapture(video_path)
    print("isOpened:", vid.isOpened())



    step = 100
    frame_idx = 0
    max_frames_used = 30


    while True:

        ret, frame = vid.read()
        if not ret:
            break

        if frame_idx % step == 0:
            images.append(frame)

        frame_idx += 1

        if len(images) >= max_frames_used:
            break

    vid.release()
    print("Sampled frames:", len(images))


    image_size = None


    # loop to detect the corners of the training images (automatically or manually)
    for i in images:    
        
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) 

        if image_size is None:
            image_size = gray.shape[::-1]
            print("image_size:", image_size)  # (width, height)
            print("center:", image_size[0]/2, image_size[1]/2)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # if found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        else:
            print("manual annotation needed")
            corners2 = get_manual_corners(i, pattern_size)  
            if corners2 is not None: 
                corners2 = cv2.cornerSubPix(gray, corners2, (11,11), (-1,-1), criteria)  
            if corners2 is None:
                print("FAILED:")
                continue
            else:
                print("OK")
        imgpoints.append(corners2)
        objpoints.append(objp)

        found = ret or (corners2 is not None)
        cv2.drawChessboardCorners(i, pattern_size, corners2, found)
        cv2.imshow('img', cv2.resize(i, (768, 1024)))
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()


    do_calibration(f"cam{cam_id}_intrinsics", objpoints, imgpoints)




