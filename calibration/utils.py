# utils.py

import os
import yaml
import numpy as np
import cv2
import glob

def create_folder(path):
    """创建文件夹，如果不存在的话"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"文件夹已创建: {path}")

def save_to_yaml(data, file_path):
    """将数据（包含numpy数组）保存到YAML文件"""
    # 将字典中的numpy数组转换为列表
    data_to_save = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = value.tolist()
        else:
            data_to_save[key] = value

    with open(file_path, 'w') as f:
        yaml.dump(data_to_save, f, default_flow_style=False)
    print(f"标定参数已保存至: {file_path}")

def find_chessboard_corners(images_path, pattern_size, square_size):
    """
    在给定路径的图片中查找棋盘格角点
    :param images_path: 图片文件夹路径
    :param pattern_size: 棋盘格内部角点数 (width, height)
    :param square_size: 棋盘格方块的物理尺寸 (e.g., in mm)
    :return: objpoints, imgpoints, img_shape
    """
    # 准备3D世界坐标中的对象点 (0,0,0), (1,0,0), (2,0,0) ....,(w-1,h-1,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    images = glob.glob(os.path.join(images_path, '*.jpg'))
    if not images:
        images = glob.glob(os.path.join(images_path, '*.png'))
    if not images:
        images = glob.glob(os.path.join(images_path, '*.bmp'))
    if not images:
        print(f"错误: 在 '{images_path}' 中未找到 .jpg 或 .png 或 .BMP 格式的图片。")
        return None, None, None

    img_shape = None
    print(f"正在从 '{images_path}' 查找角点...")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            # 提高角点检测精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"  - {os.path.basename(fname)}: 成功找到角点。")
        else:
            print(f"  - {os.path.basename(fname)}: 未能找到角点。")

    if not objpoints:
        print("错误: 在任何图片中都未能成功找到角点。请检查棋盘格尺寸或图片质量。")
        return None, None, None

    return objpoints, imgpoints, img_shape