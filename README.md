# RoboMaster 装甲板检测系统

基于ROS2的RoboMaster机器人装甲板检测系统，包含相机驱动和深度学习检测模型。

## 项目结构

- `src/` - ROS2节点源代码
  - `source/` - 相机和检测器节点
  - `config/` - 配置文件和RViz配置
  - `launch/` - 启动文件
- `model/` - 深度学习模型
  - 训练、测试和预测脚本
  - CNN装甲板检测模型
- `calibration/` - 相机标定工具
  - 单目和立体相机标定
  - 鱼眼相机标定

## 功能特性

- [x] 支持相机实时取流，完成稳定识别装甲板，已集成到ros
- [x] 相机标定
- [x] 完成pnp结算，求出装甲板的在相机坐标系下的坐标，并集成到ros
- [x] 训练模型，检测装甲板数字，提高装甲板的识别精度（但是未集成到ros)

## 使用方法

1. 构建项目：
   ```bash
   colcon build --packages-select hik_camera
   ```

2. 启动相机和装甲板识别节点
   ```
   ros2 launch hik_camera combined_launch.py
   ```

## 装甲板检测和pnp解算
![pnp](resource/pnp-solve.png)
