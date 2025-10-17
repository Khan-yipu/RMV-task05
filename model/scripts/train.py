import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import os
from tqdm import tqdm
import numpy as np

import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from models.armor_cnn import create_model
from utils.data_loader import get_data_loaders
from utils.visualization import plot_training_curve


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        # 使用余弦退火学习率调度
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)

        # 记录训练过程
        self.train_losses = []
        self.val_accuracies = []
        self.best_acc = 0.0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

            # 更新进度条
            if batch_idx % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    }
                )

        return running_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def train(self, num_epochs):
        """训练模型"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"批次大小: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 40)

            # 训练
            epoch_loss = self.train_epoch()
            self.train_losses.append(epoch_loss)

            # 验证
            accuracy = self.validate()
            self.val_accuracies.append(accuracy)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"训练损失: {epoch_loss:.4f}")
            print(f"验证准确率: {accuracy:.2f}%")
            print(f"学习率: {current_lr:.6f}")

            # 保存最佳模型
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "accuracy": accuracy,
                        "loss": epoch_loss,
                    },
                    model_path,
                )
                print(f"  ✅ 保存最佳模型，准确率: {accuracy:.2f}%")

        # 保存最终模型
        final_model_path = os.path.join(config.MODEL_SAVE_DIR, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)

        # 计算总训练时间
        total_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总训练时间: {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"最佳验证准确率: {self.best_acc:.2f}%")

        # 绘制训练曲线
        curve_path = os.path.join(config.LOG_DIR, "training_curve.png")
        plot_training_curve(
            self.train_losses, self.val_accuracies, save_path=curve_path
        )

        return self.best_acc


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        # 获取数据加载器
        train_loader, val_loader = get_data_loaders()

        # 创建模型
        model = create_model(num_classes=config.NUM_CLASSES, model_type="standard")
        model = model.to(config.DEVICE)

        # 打印模型信息
        print("模型结构:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 创建训练器并开始训练
        trainer = Trainer(model, train_loader, val_loader)
        best_accuracy = trainer.train(config.NUM_EPOCHS)

        print(f"\n🎉 训练完成！最佳准确率: {best_accuracy:.2f}%")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
