# main.py

import argparse
import os
from utils import create_folder

def main():
    parser = argparse.ArgumentParser(description="相机标定工具 (单目, 双目, 鱼眼)")
    
    parser.add_argument('--type', type=str, required=True, choices=['mono', 'stereo', 'fisheye'],
                        help="标定类型: 'mono', 'stereo', or 'fisheye'")
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help="包含标定图片的文件夹路径。对于双目标定，这是包含 'left' 和 'right' 子文件夹的父目录。")
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help="存放标定结果和校正图像的输出文件夹。")

    parser.add_argument('--pattern_size', type=str, required=True,
                        help="棋盘格内部角点数，格式为 '宽度x高度'，例如 '9x6'。")

    parser.add_argument('--square_size', type=float, required=True,
                        help="棋盘格方块的实际尺寸（例如，毫米mm）。")

    parser.add_argument('--undistort', action='store_true',
                        help="如果设置此参数，将对输入图像进行校正并保存。")

    args = parser.parse_args()

    # 解析棋盘格尺寸
    try:
        width, height = map(int, args.pattern_size.split('x'))
        pattern_size = (width, height)
    except ValueError:
        print("错误: --pattern_size 参数格式不正确。请使用 '宽度x高度' 格式，例如 '9x6'。")
        return

    # 创建输出文件夹
    create_folder(args.output_dir)

    # 根据类型调用相应的标定器
    if args.type == 'mono':
        from calibrators import mono_calibrator
        mono_calibrator.calibrate(
            image_dir=args.input_dir,
            pattern_size=pattern_size,
            square_size=args.square_size,
            output_dir=args.output_dir,
            undistort=args.undistort
        )
    elif args.type == 'fisheye':
        from calibrators import fisheye_calibrator
        fisheye_calibrator.calibrate(
            image_dir=args.input_dir,
            pattern_size=pattern_size,
            square_size=args.square_size,
            output_dir=args.output_dir,
            undistort=args.undistort
        )
    elif args.type == 'stereo':
        from calibrators import stereo_calibrator
        left_dir = os.path.join(args.input_dir, 'left')
        right_dir = os.path.join(args.input_dir, 'right')
        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            print(f"错误: 双目标定需要 '{args.input_dir}' 文件夹下包含 'left' 和 'right' 子文件夹。")
            return
        stereo_calibrator.calibrate(
            left_dir=left_dir,
            right_dir=right_dir,
            pattern_size=pattern_size,
            square_size=args.square_size,
            output_dir=args.output_dir,
            undistort=args.undistort
        )

if __name__ == '__main__':
    main()