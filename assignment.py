import glm
import random
import numpy as np
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
block_size = 1.0
lookup_table = {}
MASK_FILES = None
FRAME_IDX = 0
WORLD_SCALE = 0.05
#WORLD_OFFSET = np.array([-2.05, 0.45, -2.30], dtype=np.float32)

def np_R_to_glm_mat4(R: np.ndarray) -> glm.mat4:
    R = np.asarray(R, dtype=np.float32)
    assert R.shape == (3, 3), f"Expected (3,3), got {R.shape}"

    M = glm.mat4(1.0)
    # GLM matrices are column-major: M[col][row]
    for row in range(3):
        for col in range(3):
            M[col][row] = float(R[row, col])
    return M

def load_camera_config(cam_number):

    config_path = os.path.join(
        SCRIPT_DIR,
        "data",
        f"cam{cam_number}",
        "config.xml"
    )

    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open: {config_path}")

    camera_matrix = fs.getNode("CameraMatrix").mat()
    distortion_coeffs = fs.getNode("DistortionCoeffs").mat()
    rvec = fs.getNode("Rvec").mat()
    tvec = fs.getNode("Tvec").mat()
    rotation_matrix = fs.getNode("RotationMatrix").mat()
    fs.release()

    return camera_matrix, distortion_coeffs, rvec, tvec, rotation_matrix

# returns list of 4 lists, each list contains num_frames filepaths
def get_masks(num_frames=100):
    mask_files_by_cam = []
    for cam_id in range(1, 5):
        masks_dir = os.path.join(DATA_DIR, f"cam{cam_id}", "masks")
        files = sorted(
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.endswith(".png")
        )
        mask_files_by_cam.append(files[:num_frames])
    return mask_files_by_cam

def load_masks_for_frame(mask_files_by_cam, frame_idx):
    masks = []
    for cam_i in range(4):
        path = mask_files_by_cam[cam_i][frame_idx]
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # creates boolean mask (white=foreground=true / black=background=false)
        masks.append(m != 0)
    return masks

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    global lookup_table, MASK_FILES, FRAME_IDX

    # load mask file lists once
    if MASK_FILES is None:
        MASK_FILES = get_masks(num_frames=100)

    # load masks for the current frame
    masks = load_masks_for_frame(MASK_FILES, FRAME_IDX)

    # build lookup table
    if not lookup_table:
        x0, x1 = 0, width
        y0, y1 = 0, height
        z0, z1 = 0, depth
        IMG_W, IMG_H = 644, 486

        cams = []
        for cam_id in range(1, 5):
            K, dist, rvec, tvec, _ = load_camera_config(cam_id)
            cams.append((K, dist, rvec, tvec))
                    
        def to_opencv_point(X, Y, Z):
            return (
                X * WORLD_SCALE,
                Z * WORLD_SCALE,
                -Y * WORLD_SCALE + 0.5,
            )

        for ix in range(x0, x1):
            for iy in range(y0, y1):
                for iz in range(z0, z1):
                    X = ix * block_size - width / 2
                    Y = iy * block_size
                    Z = iz * block_size - depth / 2

                    Xcv, Ycv, Zcv = to_opencv_point(X, Y, Z)
                    point = np.array([[[Xcv, Ycv, Zcv]]], dtype=np.float32)

                    per_cam = []
                    for (K, dist, rvec, tvec) in cams:
                        imgpts, _ = cv2.projectPoints(point, rvec, tvec, K, dist)
                        u = float(imgpts[0, 0, 0])
                        v = float(imgpts[0, 0, 1])

                        if (not np.isfinite(u)) or (not np.isfinite(v)):
                            per_cam.append((0, 0, False))
                            continue

                        ui = int(round(u))
                        vi = int(round(v))

                        # chekcs if point is behind camera
                        R, _ = cv2.Rodrigues(rvec)
                        X_cam = R @ np.array([[Xcv], [Ycv], [Zcv]], dtype=np.float32) + tvec.reshape(3, 1)
                        z_cam = float(X_cam[2, 0])

                        valid = (z_cam > 0.0) and (0 <= ui < IMG_W) and (0 <= vi < IMG_H)
                        per_cam.append((ui, vi, valid))

                    lookup_table[(ix, iy, iz)] = per_cam

    # keep voxel only if it's foreground in all 4 views
    on_positions, on_colors = [], []

    for (ix, iy, iz), per_cam in lookup_table.items():
        keep = True
        for cam_i in range(4):
            u, v, valid = per_cam[cam_i]
            if (not valid) or (not masks[cam_i][v, u]):
                keep = False
                break

        if keep:
            X = ix * block_size - width / 2
            Y = iy * block_size
            Z = iz * block_size - depth / 2
            on_positions.append([X, Y, Z])
            on_colors.append([ix / width, iz / depth, iy / height])


    print("ON voxels:", len(on_positions))

    pass_valid = [0, 0, 0, 0]
    pass_fg = [0, 0, 0, 0]

    for per_cam in lookup_table.values():
        for cam_i in range(4):
            u, v, valid = per_cam[cam_i]
            if valid:
                pass_valid[cam_i] += 1
                if masks[cam_i][v, u]:
                    pass_fg[cam_i] += 1

    print("valid projections per cam:", pass_valid)
    print("valid+foreground per cam:", pass_fg)
    return on_positions, on_colors     
    

def get_cam_positions():
    # Camera centers from extrinsics, converted to the visualizer's world coordinates
    cam_positions = []
    cam_colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]

    for cam_id in range(1, 5):
        _, _, rvec, tvec, rotation_matrix = load_camera_config(cam_id)

        # OpenCV world -> camera: X_cam = R X_world + t
        R = rotation_matrix
        if R is None or R.size == 0:
            R, _ = cv2.Rodrigues(rvec)

        # Camera center in calibration/world coordinates
        C_cv = -R.T @ tvec.reshape(3, 1)
        C_cv = C_cv.reshape(3)

        # Convert from calibration axes to visualizer axes:
        # (Xcv, Ycv, Zcv) -> (x, y, z) = (Xcv, -Zcv, Ycv)
        C_vis = np.array([
            C_cv[0],
            -C_cv[2],
            C_cv[1],
        ], dtype=np.float32) / WORLD_SCALE

        cam_positions.append(C_vis.tolist())

    return cam_positions, cam_colors


def get_cam_rotation_matrices():
    cam_rotations = []

    # Converts vectors from calibration-world basis to visualizer-world basis
    cv_world_to_vis_world = np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0],
    ], dtype=np.float32)

    # camera.json points along +x in model space.
    # Map mesh axes -> OpenCV camera axes:
    #   mesh +x (forward) -> cv camera +z (forward)
    #   mesh +y (up)      -> cv camera -y (up, since cv +y is down)
    #   mesh +z           -> cv camera +x
    mesh_to_cv_cam = np.array([
        [0,  0, 1],
        [0, -1, 0],
        [1,  0, 0],
    ], dtype=np.float32)

    for cam_id in range(1, 5):
        _, _, rvec, _, rotation_matrix = load_camera_config(cam_id)

        R = rotation_matrix
        if R is None or R.size == 0:
            R, _ = cv2.Rodrigues(rvec)

        # OpenCV extrinsics give world -> camera.
        # For a camera object in the scene, we need camera -> world.
        R_cam_to_world_cv = R.T

        # Convert into the visualizer's world basis and account for mesh orientation
        R_mesh_to_world_vis = cv_world_to_vis_world @ R_cam_to_world_cv @ mesh_to_cv_cam

        cam_rotations.append(np_R_to_glm_mat4(R_mesh_to_world_vis))

    return cam_rotations


def debug_print_camera_centers():
    for cam_id in range(1, 5):
        K, dist, rvec, tvec, _ = load_camera_config(cam_id)
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        print("cam", cam_id, "center (world):", C.ravel())

def main():
    debug_print_camera_centers()

if __name__ == "__main__":
    main()
