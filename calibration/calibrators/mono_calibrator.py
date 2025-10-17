# calibrators/mono_calibrator.py

import cv2
import numpy as np
import os
import glob
from utils import save_to_yaml, find_chessboard_corners, create_folder

def calibrate(image_dir, pattern_size, square_size, output_dir, undistort):
    """单目标定主函数"""
    print("--- 开始单目标定 ---")
    
    # 查找角点
    objpoints, imgpoints, img_shape = find_chessboard_corners(image_dir, pattern_size, square_size)
    if objpoints is None:
        return

    print("\n正在计算相机参数...")
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    if not ret:
        print("标定失败！")
        return

    print("标定成功！")
    print("相机内参矩阵 (mtx):\n", mtx)
    print("畸变系数 (dist):\n", dist)
    
    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\n平均重投影误差: {mean_error / len(objpoints)}")

    # 保存参数
    params_path = os.path.join(output_dir, "params")
    create_folder(params_path)
    calib_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'image_width': img_shape[0],
        'image_height': img_shape[1]
    }
    save_to_yaml(calib_data, os.path.join(params_path, 'mono_calib.yaml'))

    # 校正图像并保存
    if undistort:
        print("\n正在校正图像...")
        undistorted_dir = os.path.join(output_dir, "mono_undistorted")
        create_folder(undistorted_dir)
        images = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))+ glob.glob(os.path.join(image_dir, '*.bmp'))
        
        for fname in images:
            img = cv2.imread(fname)
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            output_path = os.path.join(undistorted_dir, "undistorted_" + os.path.basename(fname))
            cv2.imwrite(output_path, undistorted_img)
            print(f"  - 已保存校正图像: {output_path}")

    print("--- 单目标定完成 ---")