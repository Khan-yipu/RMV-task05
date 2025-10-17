import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from models.armor_cnn import create_model


class Predictor:
    def __init__(self, model_path):
        self.model = create_model(num_classes=config.NUM_CLASSES)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = self.model.to(config.DEVICE)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.class_names = [str(i) for i in range(1, config.NUM_CLASSES + 1)]
        print(f"✅ 预测器初始化完成，使用模型: {os.path.basename(model_path)}")

    def predict(self, image_path):
        """预测单张图片"""
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
        except Exception as e:
            print(f"❌ 错误: 无法加载图像 {image_path}: {e}")
            return None, None, None, None

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(config.DEVICE)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = predicted.item()
            confidence_score = confidence.item()

            # 获取所有类别的概率
            all_probs = probabilities.squeeze().cpu().numpy()

        return predicted_class, confidence_score, all_probs, image, original_size

    def display_prediction(self, image_path):
        """显示预测结果"""
        result = self.predict(image_path)
        if result[0] is None:
            return

        predicted_class, confidence, all_probs, image, original_size = result

        # 显示结果
        plt.figure(figsize=(15, 5))

        # 显示原图像
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"原图像 ({original_size[0]}×{original_size[1]})", fontsize=12)
        plt.axis("off")

        # 显示预处理后的图像
        plt.subplot(1, 3, 2)
        processed_image = self.transform(image).numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_image = std * processed_image + mean
        processed_image = np.clip(processed_image, 0, 1)
        plt.imshow(processed_image)
        plt.title(f"预处理后 ({config.IMAGE_WIDTH}×{config.IMAGE_HEIGHT})", fontsize=12)
        plt.axis("off")

        # 显示概率分布
        plt.subplot(1, 3, 3)
        colors = [
            "lightcoral" if i != predicted_class else "lightgreen"
            for i in range(len(all_probs))
        ]
        bars = plt.bar(self.class_names, all_probs, color=colors, alpha=0.7)
        plt.ylim(0, 1)
        plt.title("各类别概率分布", fontsize=12, fontweight="bold")
        plt.xlabel("数字类别", fontsize=11)
        plt.ylabel("概率", fontsize=11)
        plt.grid(True, alpha=0.3)

        # 在柱状图上显示概率值
        for bar, prob in zip(bars, all_probs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()

        # 打印详细结果
        print(f"\n📈 预测结果:")
        print(f"  图像: {os.path.basename(image_path)}")
        print(f"  预测数字: {self.class_names[predicted_class]}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  原始尺寸: {original_size}")
        print(f"\n所有类别概率:")
        for i, prob in enumerate(all_probs):
            status = "✅" if i == predicted_class else "  "
            print(f"  {status} 数字 {self.class_names[i]}: {prob:.4f}")


def main():
    # 创建预测器
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        print("请先运行训练脚本 train.py")
        return

    predictor = Predictor(model_path)

    # 交互式预测
    while True:
        print("\n" + "=" * 60)
        print("装甲板数字识别预测系统")
        print("=" * 60)
        print("1. 输入图片路径进行预测")
        print("2. 输入 'sample' 测试样例图片")
        print("3. 输入 'quit' 退出")

        user_input = input("\n请选择操作: ").strip()

        if user_input.lower() == "quit":
            print("再见！")
            break
        elif user_input.lower() == "sample":
            # 查找样例图片
            sample_dirs = [config.TRAIN_DIR, config.TEST_DIR]
            sample_images = []
            for sample_dir in sample_dirs:
                if os.path.exists(sample_dir):
                    for class_dir in glob.glob(os.path.join(sample_dir, "*")):
                        if os.path.isdir(class_dir):
                            images = (
                                glob.glob(os.path.join(class_dir, "*.jpg"))
                                + glob.glob(os.path.join(class_dir, "*.png"))
                                + glob.glob(os.path.join(class_dir, "*.jpeg"))
                            )
                            sample_images.extend(images[:2])  # 每类取2张

            if sample_images:
                import random

                sample_image = random.choice(sample_images)
                print(f"使用样例图片: {sample_image}")
                predictor.display_prediction(sample_image)
            else:
                print("未找到样例图片")
        else:
            image_path = user_input
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                continue

            predictor.display_prediction(image_path)


if __name__ == "__main__":
    main()
