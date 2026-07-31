"""Camera intrinsics from the ChArUco shots in model/board_images/.

Writes camera_intrinsics.npz (camera_matrix, dist_coeffs, image_size) and a
per-view reprojection-error figure next to this file.
"""
import glob
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

COLS, ROWS = 6, 8
square_len = 0.030
marker_len = 0.0225
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
board = cv2.aruco.CharucoBoard((COLS, ROWS), square_len, marker_len, aruco_dict)


def calibrate_camera_intrinsic(image_filenames):
    detector = cv2.aruco.CharucoDetector(board)
    all_obj, all_img, used = [], [], []
    image_resolution = None

    for img_file in sorted(image_filenames):
        img_bgr = cv2.imread(img_file)
        if img_bgr is None:
            print("=> unreadable: {0}".format(img_file))
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        wh = img_gray.shape[:2][::-1]
        if image_resolution is None:
            image_resolution = wh
        elif wh != image_resolution:
            raise ValueError("{0}: size {1} != {2}".format(img_file, wh, image_resolution))

        charucoCorners, charucoIDs, _, _ = detector.detectBoard(img_gray)
        n = 0 if charucoCorners is None else len(charucoCorners)
        print("=> {0}: {1} corners".format(os.path.basename(img_file), n))
        if n < 6:                      # solvePnP/calibration needs a real view
            continue
        objPoints, imgPoints = board.matchImagePoints(charucoCorners, charucoIDs)
        all_obj.append(objPoints)
        all_img.append(imgPoints)
        used.append(os.path.basename(img_file))

    if len(all_obj) < 4:
        raise RuntimeError("only {0} usable views; aim for 20+".format(len(all_obj)))

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_resolution, None, None)

    print("views used:", len(all_obj))
    print("ret (RMS reprojection, px):", ret)
    print("camera_matrix:\n", camera_matrix)
    print("distortion_coefficients:", dist_coeffs.ravel())

    per_view = []
    for o, i, rv, tv in zip(all_obj, all_img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, camera_matrix, dist_coeffs)
        err = proj.reshape(-1, 2) - i.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))))
    return camera_matrix, dist_coeffs, image_resolution, ret, used, per_view


if __name__ == "__main__":
    files = glob.glob(os.path.join(HERE, "board_images", "*.jpeg"))
    K, D, size, rms, used, per_view = calibrate_camera_intrinsic(files)

    np.savez(os.path.join(HERE, "camera_intrinsics.npz"),
             camera_matrix=K, dist_coeffs=D, image_size=np.array(size), rms=rms)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(per_view)), per_view, color="tab:blue")
    ax.axhline(rms, color="tab:red", ls="--", label="overall RMS = {0:.3f} px".format(rms))
    ax.set_xticks(range(len(used)))
    ax.set_xticklabels([u.split()[-1].replace(".jpeg", "") for u in used],
                       rotation=90, fontsize=7)
    ax.set_ylabel("reprojection RMS (px)")
    ax.set_title("ChArUco intrinsics calibration - {0} views @ {1}x{2}\n"
                 "fx={3:.1f} fy={4:.1f} cx={5:.1f} cy={6:.1f}".format(
                     len(used), size[0], size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "calibration_reprojection_error.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
