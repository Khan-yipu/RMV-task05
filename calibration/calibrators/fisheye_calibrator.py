# calibrators/fisheye_calibrator.py

import cv2
import numpy as np
import os
import glob
from utils import save_to_yaml, find_chessboard_corners, create_folder

def calibrate(image_dir, pattern_size, square_size, output_dir, undistort):
    """鱼眼相机标定主函数"""
    print("--- 开始鱼眼相机标定 ---")
    
    objpoints, imgpoints, img_shape = find_chessboard_corners(image_dir, pattern_size, square_size)
    if objpoints is None:
        return
    objpoints = [p.reshape(-1, 1, 3) for p in objpoints]

    print("\n正在计算鱼眼相机参数...")
    
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    
    ret, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    if not ret:
        print("鱼眼标定失败！")
        return
    
    print("相机内参矩阵 (K):\n", K)
    print("畸变系数 (D):\n", D)
    
    # 保存参数
    params_path = os.path.join(output_dir, "params")
    create_folder(params_path)
    calib_data = {
        'camera_matrix': K,
        'dist_coeffs': D,
        'image_width': img_shape[0],
        'image_height': img_shape[1]
    }
    save_to_yaml(calib_data, os.path.join(params_path, 'fisheye_calib.yaml'))

    # 校正图像
    if undistort:
        print("\n正在校正鱼眼图像...")
        undistorted_dir = os.path.join(output_dir, "fisheye_undistorted")
        create_folder(undistorted_dir)
        images = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))+ glob.glob(os.path.join(image_dir, '*.bmp'))
        
        for fname in images:
            img = cv2.imread(fname)
            # 使用cv2.fisheye.undistortImage进行校正
            undistorted_img = cv2.fisheye.undistortImage(img, K, D, Knew=K)
            output_path = os.path.join(undistorted_dir, "undistorted_" + os.path.basename(fname))
            cv2.imwrite(output_path, undistorted_img)
            print(f"  - 已保存校正图像: {output_path}")

    print("--- 鱼眼相机标定完成 ---")
