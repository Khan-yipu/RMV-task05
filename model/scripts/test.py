import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from models.armor_cnn import create_model
from utils.data_loader import get_test_loader
from utils.visualization import plot_confusion_matrix, visualize_predictions


def test_model(model_path):
    """测试模型"""
    # 加载模型
    model = create_model(num_classes=config.NUM_CLASSES)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "accuracy" in checkpoint:
                print(f"模型训练准确率: {checkpoint['accuracy']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = model.to(config.DEVICE)
    model.eval()

    print(f"加载模型: {model_path}")
    print(f"测试设备: {config.DEVICE}")

    # 获取测试数据加载器
    test_loader = get_test_loader()

    # 测试
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100

    # 打印结果
    print(f"\n📊 测试结果:")
    print(f"整体准确率: {accuracy:.2f}%")

    # 每个类别的准确率
    class_names = [str(i) for i in range(1, config.NUM_CLASSES + 1)]
    print(f"\n每个类别的准确率:")
    class_accuracy = {}
    for i in range(config.NUM_CLASSES):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == i) * 100
            class_accuracy[f"class_{i + 1}"] = class_acc
            print(f"  数字 {i + 1}: {class_acc:.2f}% ({np.sum(class_mask)} 张图片)")

    # 分类报告
    print(f"\n详细分类报告:")
    print(
        classification_report(
            all_labels, all_predictions, target_names=class_names, digits=4
        )
    )

    # 绘制混淆矩阵
    cm_path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=cm_path)

    # 可视化一些预测结果
    viz_path = os.path.join(config.LOG_DIR, "predictions_visualization.png")
    visualize_predictions(model, test_loader, class_names, save_path=viz_path)

    # 保存测试结果
    results = {
        "overall_accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "total_samples": len(all_labels),
        "predictions": all_predictions,
        "labels": all_labels,
    }

    results_path = os.path.join(config.LOG_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k not in ["predictions", "labels"]},
            f,
            indent=2,
        )

    print(f"\n测试结果已保存到: {results_path}")

    return accuracy, all_predictions, all_labels


def main():
    # 测试最佳模型
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        test_model(best_model_path)
    else:
        print(f"未找到模型文件: {best_model_path}")
        print("请先运行训练脚本 train.py")


if __name__ == "__main__":
    main()
