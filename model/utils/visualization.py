import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from config.config import config


def plot_training_curve(train_losses, val_accuracies, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color="orange", linewidth=2)
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"训练曲线已保存到: {save_path}")

    plt.show()


def plot_confusion_matrix(true_labels, predictions, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"混淆矩阵已保存到: {save_path}")

    plt.show()


def visualize_predictions(
    model, data_loader, class_names, num_images=12, save_path=None
):
    """可视化预测结果"""
    model.eval()
    device = next(model.parameters()).device

    # 获取一批数据
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(images[:num_images])
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)

    # 绘制结果
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        # 反标准化图像
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predictions[i].item()]
        confidence = confidences[i].item()

        color = "green" if true_label == pred_label else "red"
        axes[i].set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.3f}",
            color=color,
            fontsize=10,
            pad=10,
        )
        axes[i].axis("off")

    # 隐藏多余的子图
    for i in range(min(num_images, len(images)), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"预测可视化已保存到: {save_path}")

    plt.show()


def plot_sample_images(data_loader, class_names, save_path=None):
    """显示数据集中样本图片"""
    images, labels = next(iter(data_loader))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        if i < len(images):
            image = images[i].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            axes[i].imshow(image)
            axes[i].set_title(f"Label: {class_names[labels[i].item()]}", fontsize=10)
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
