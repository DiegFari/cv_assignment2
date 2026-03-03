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
WORLD_OFFSET = np.array([-2.05, 0.45, -2.30], dtype=np.float32)


def debug_print_camera_centers_converted():
    for cam_id in range(1, 5):
        _, _, rvec, tvec, _ = load_camera_config(cam_id)
        R, _ = cv2.Rodrigues(rvec)
        C_cv = (-R.T @ tvec).reshape(3)

        X, Y, Z = from_opencv_world(C_cv[0], C_cv[1], C_cv[2])
        print(f"cam {cam_id} center (your world): [{X:.3f}, {Y:.3f}, {Z:.3f}]")

def to_opencv_point(X, Y, Z):
    return (
        X * WORLD_SCALE + WORLD_OFFSET[0],
        Z * WORLD_SCALE  + WORLD_OFFSET[1],
        Y * WORLD_SCALE + WORLD_OFFSET[2],
    )

def from_opencv_world(Xcv, Ycv, Zcv):
    # Inverse of to_opencv_point (ignoring any camera projection stuff)
    X = (Xcv - WORLD_OFFSET[0]) / WORLD_SCALE
    Z = (Ycv - WORLD_OFFSET[1]) / WORLD_SCALE  # OpenCV Y was your Z
    Y = (Zcv - WORLD_OFFSET[2]) / WORLD_SCALE  # OpenCV Z was your Y
    return float(X), float(Y), float(Z)

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
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]], \
    #     [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    positions = []
    colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


    for cam_id in range(1, 5):
        K, dist, rvec, tvec, _ = load_camera_config(cam_id)

        R, _ = cv2.Rodrigues(rvec)          # world_cv -> cam
        C_cv = -R.T @ tvec.reshape(3, 1)    # camera center in OpenCV world coords




        X, Y, Z = from_opencv_world(C_cv[0, 0], C_cv[1, 0], C_cv[2, 0])
        positions.append([X, Y, Z])

    return positions, colors


    # R_matrices = []
    # for cam_id in range (1, 5): 
    #     K, dist, rvecs, tvecs, R = np_R_to_glm_mat4(load_camera_config(cam_id))
    #     R_matrices.append(R)
    # return R_matrices

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    # return cam_rotations

    cam_rotations = []
    P = np.array([[1,0,0],
                   [0,0,1],
                   [0,1,0]], dtype=np.float32)

    for cam_id in range(1, 5):
        K, dist, rvec, tvec, _ = load_camera_config(cam_id)

        R_wc_cv, _ = cv2.Rodrigues(rvec)     # world_cv -> cam
        R_cw_cv = R_wc_cv.T                  # cam -> world_cv

        R_cw_yours = P @ R_cw_cv             # cam -> world_yours (axis swap)

        cam_rotations.append(np_R_to_glm_mat4(R_cw_yours))

    return cam_rotations


def debug_print_camera_centers():
    for cam_id in range(1, 5):
        K, dist, rvec, tvec, _ = load_camera_config(cam_id)
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        print("cam", cam_id, "center (world):", C.ravel())

def main():
    debug_print_camera_centers()
    debug_print_camera_centers_converted()

if __name__ == "__main__":
    main()
