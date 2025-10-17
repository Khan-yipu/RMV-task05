# calibrators/stereo_calibrator.py

import cv2
import numpy as np
import os
import glob
from utils import save_to_yaml, create_folder

def calibrate(left_dir, right_dir, pattern_size, square_size, output_dir, undistort):
    """双目标定主函数"""
    print("--- 开始双目标定 ---")
    
    # 准备3D世界坐标点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []  # 3D点
    imgpoints_l = []  # 左相机图像点
    imgpoints_r = []  # 右相机图像点

    left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')) + glob.glob(os.path.join(left_dir, '*.png')))+ glob.glob(os.path.join(left_dir, '*.bmp'))
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')) + glob.glob(os.path.join(right_dir, '*.png')))+ glob.glob(os.path.join(right_dir, '*.bmp'))

    if len(left_images) != len(right_images) or not left_images:
        print("错误: 左右相机图片数量不匹配或文件夹为空。")
        return

    img_shape = None
    print("正在从左右图像对中查找角点...")
    for i in range(len(left_images)):
        img_l = cv2.imread(left_images[i])
        img_r = cv2.imread(right_images[i])
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        # 分别在左右图像中查找角点
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

        # 只有当左右图像都成功找到角点时，才保留这对数据
        if ret_l and ret_r:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)
            print(f"  - {os.path.basename(left_images[i])} / {os.path.basename(right_images[i])}: 成功。")
        else:
            print(f"  - {os.path.basename(left_images[i])} / {os.path.basename(right_images[i])}: 失败。")

    if not objpoints:
        print("错误: 在任何图像对中都未能成功找到角点。")
        return

    # 先对左右相机分别进行单目标定，得到初始内参
    print("\n正在进行单目标定以获取初始内参...")
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # 进行双目标定
    print("\n正在进行双目标定...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_shape, flags=stereo_flags)

    if not ret:
        print("双目标定失败！")
        return

    print("双目标定成功！")
    print("左相机内参矩阵:\n", mtx_l)
    print("右相机内参矩阵:\n", mtx_r)
    print("旋转矩阵 (R):\n", R)
    print("平移向量 (T):\n", T)

    # 保存参数
    params_path = os.path.join(output_dir, "params")
    create_folder(params_path)
    calib_data = {
        'camera_matrix_left': mtx_l, 'dist_coeffs_left': dist_l,
        'camera_matrix_right': mtx_r, 'dist_coeffs_right': dist_r,
        'R': R, 'T': T, 'E': E, 'F': F,
        'image_width': img_shape[0], 'image_height': img_shape[1]
    }
    save_to_yaml(calib_data, os.path.join(params_path, 'stereo_calib.yaml'))

    # 进行立体校正并保存
    if undistort:
        print("\n正在进行立体校正并保存图像...")
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T, alpha=0)

        map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
        map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)
        
        undistorted_dir_l = os.path.join(output_dir, "stereo_undistorted", "left")
        undistorted_dir_r = os.path.join(output_dir, "stereo_undistorted", "right")
        create_folder(undistorted_dir_l)
        create_folder(undistorted_dir_r)

        for i in range(len(left_images)):
            img_l = cv2.imread(left_images[i])
            img_r = cv2.imread(right_images[i])

            undistorted_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
            undistorted_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

            # 在校正后的图像上画线以检查对齐情况
            for line in range(0, int(undistorted_l.shape[0] / 20)):
                undistorted_l[line * 20, :] = (0, 255, 0)
                undistorted_r[line * 20, :] = (0, 255, 0)

            output_path_l = os.path.join(undistorted_dir_l, "rectified_" + os.path.basename(left_images[i]))
            output_path_r = os.path.join(undistorted_dir_r, "rectified_" + os.path.basename(right_images[i]))
            cv2.imwrite(output_path_l, undistorted_l)
            cv2.imwrite(output_path_r, undistorted_r)
            print(f"  - 已保存校正图像对: {os.path.basename(left_images[i])} / {os.path.basename(right_images[i])}")
            
    print("--- 双目标定完成 ---")