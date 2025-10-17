import torch
import os


class Config:
    # 数据配置
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    NUM_CLASSES = 5

    # 图像尺寸配置 (20×28)
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 20
    IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)  # (height, width)

    # 训练配置
    BATCH_SIZE = 64  # 小图像可以用更大的batch size
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # 模型配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出配置
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # 创建目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


config = Config()
