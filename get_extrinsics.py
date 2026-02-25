import cv2
import numpy as np
import os

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

def save_camera_config_xml(path: str, K: np.ndarray, dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
         """
         This functions save extrinsics + intrinsics in the format requried in a config file
         """

         os.makedirs(os.path.dirname(path), exist_ok=True)

         R, _ = cv2.Rodrigues(rvec)
         R = R.astype(np.float32)

         fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        
        # Intrisiscs 
         fs.write("CameraMatrix", K)
         fs.write("DistortionCoeffs", dist)

         # Extrinsics
         fs.write("Rvec", rvec)
         fs.write("Tvec", tvec)
         fs.write("RotationMatrix", R)

         fs.release()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for cam_id in range(1, 5):
        
        video_path = os.path.join(SCRIPT_DIR, "data", f"cam{cam_id}", "checkerboard.avi") 
        vid = cv2.VideoCapture(video_path)

        # getting a frame (we took one approximately in the middle)
        vid.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, frame = vid.read()
        vid.release()

        if not ret:
            print("Failed to read frame")
        
        # loading the intrinsics

        calibration_file = f"cam{cam_id}_intrinsics.npz"

        data = np.load(calibration_file)
        K = data["cameraMatrix"]
        dist = data["distCoeffs"]

        pattern_size = (8, 6)
        square_size = 0.115 

        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32) 
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) 
        objp *= square_size

        corners = get_manual_corners(frame, pattern_size)

        if corners is not None: 
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
              corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        elif corners is None: 
              print(f"camera {cam_id}: FAILED CORNER DETECTION")
              break
        
        ok, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok: 
              print(f"camera {cam_id}: solvePnp failed")
              break
        
        origin = np.array([[0, 0, 0]], dtype=np.float32)

        origin3d = np.array([[0,0,0]], np.float32)
        origin_proj, _ = cv2.projectPoints(origin3d, rvec, tvec, K, dist)
        origin_proj = origin_proj.reshape(2)

        corner0 = corners[0].reshape(2)

        print("pixel diff (proj origin vs corner[0]) =", np.linalg.norm(origin_proj - corner0))

            
        axis_lenght = 3 * square_size
        axis = np.float32([[0, 0, 0], [axis_lenght,0,0], [0,axis_lenght,0], [0,0,-axis_lenght]]).reshape(-1,3)

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, K, dist)

        def draw(img, imgpts):
            # function got from the tutorial to draw the axis 
            imgpts = imgpts.reshape(-1, 2).astype(int)
            origin = tuple(imgpts[0])
            img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255,0,0), 2)
            img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 2)
            img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,0,255), 2)
            return img

        copy = frame.copy()
        copy = draw(copy, imgpts)
        cv2.imshow(f'drawn image for camera {cam_id}',cv2.resize(copy, (768, 1024)))
        k = cv2.waitKey(0) & 0xFF

        config_path = os.path.join(SCRIPT_DIR, "data", f"cam{cam_id}", "config.xml")

        save_camera_config_xml(config_path, K, dist, rvec, tvec)






        
