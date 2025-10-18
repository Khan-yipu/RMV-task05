import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from torchvision import transforms


class ONNXPredictor:
    def __init__(self, model_path):
        # 创建ONNX Runtime会话
        self.session = ort.InferenceSession(model_path)

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.IMAGE_SIZE, antialias=True),
                transforms.ColorJitter(contrast=1.5),  # 增加对比度
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 类别名称
        self.class_names = [str(i) for i in range(1, config.NUM_CLASSES + 1)]
        print(f"✅ ONNX预测器初始化完成，使用模型: {os.path.basename(model_path)}")

    def predict(self, image_path):
        """使用ONNX模型预测单张图片"""
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
        except Exception as e:
            print(f"❌ 错误: 无法加载图像 {image_path}: {e}")
            return None, None, None, None

        # 图像预处理 - 为后续步骤打下良好基础
        # 这是最关键的一步，目的是减少噪声、增强目标特征
        processed_image = self.preprocess_image(image)

        # 预处理
        image_tensor = self.transform(processed_image).unsqueeze(0).numpy()

        # ONNX推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image_tensor})

        # 处理输出
        probabilities = self.softmax(outputs[0][0])
        predicted_class = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_class])

        return predicted_class, confidence_score, probabilities, image, original_size

    def preprocess_image(self, image):
        """图像预处理：二值化、去噪、形态学操作"""
        # 转换为灰度图像
        gray_image = image.convert("L")
        
        # 转换为OpenCV格式
        cv_image = np.array(gray_image)
        
        # 二值化：使用Otsu算法自动确定最佳阈值
        _, binary_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 去噪：中值滤波去除孤立噪点，保留边缘
        denoised_image = cv2.medianBlur(binary_image, 3)
        
        # 形态学操作
        # 闭运算：填充数字内部可能存在的细小空洞
        kernel = np.ones((2, 2), np.uint8)
        closed_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel)
        
        # 开运算：消除数字边缘可能存在的细小毛刺
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)
        
        # 转换回PIL格式
        processed_pil_image = Image.fromarray(opened_image).convert("RGB")
        
        return processed_pil_image

    def softmax(self, x):
        """计算softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

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
    # ONNX模型路径
    onnx_model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.onnx")

    if not os.path.exists(onnx_model_path):
        print(f"❌ 未找到ONNX模型文件: {onnx_model_path}")
        print("请先运行导出脚本 export_onnx.py")
        return

    # 创建预测器
    predictor = ONNXPredictor(onnx_model_path)

    # 测试图像路径
    test_image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.png"
    )

    if not os.path.exists(test_image_path):
        print(f"❌ 未找到测试图像: {test_image_path}")
        return

    print(f"使用ONNX模型预测图像: {test_image_path}")
    predictor.display_prediction(test_image_path)


if __name__ == "__main__":
    main()
