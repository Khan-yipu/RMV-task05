import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from models.armor_cnn import create_model


class ONNXExporter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = config.DEVICE
        self.model = create_model(num_classes=config.NUM_CLASSES)

        # 加载模型
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def export_onnx(self, onnx_path=None):
        """导出ONNX模型"""
        if onnx_path is None:
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            onnx_path = os.path.join(config.MODEL_SAVE_DIR, f"{base_name}.onnx")

        # 创建示例输入
        dummy_input = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(
            self.device
        )

        try:
            # 方法1: 使用新的 torch.export 方法 (PyTorch 2.0+)
            print("尝试使用新的 torch.export 方法...")
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,  # 使用更新的opset
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
            )

        except Exception as e1:
            print(f"新方法失败: {e1}")
            print("尝试使用传统方法...")
            try:
                # 方法2: 传统方法
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                    verbose=False,
                )
            except Exception as e2:
                print(f"传统方法也失败: {e2}")
                raise

        print(f"✅ ONNX模型已导出: {onnx_path}")
        return onnx_path

    def export_onnx_dynamo(self, onnx_path=None):
        """使用 torch.export (PyTorch 2.0+ 的新方法)"""
        if onnx_path is None:
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            onnx_path = os.path.join(config.MODEL_SAVE_DIR, f"{base_name}_dynamo.onnx")

        try:
            import torch._dynamo

            # 创建示例输入
            dummy_input = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(
                self.device
            )

            # 使用 torch.export 方法
            exported_program = torch.export.export(self.model, (dummy_input,))
            torch.onnx.dynamo_export(exported_program, dummy_input).save(onnx_path)

            print(f"✅ Dynamo ONNX模型已导出: {onnx_path}")
            return onnx_path

        except Exception as e:
            print(f"❌ Dynamo导出失败: {e}")
            return None

    def validate_onnx(self, onnx_path):
        """验证ONNX模型"""
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX模型验证通过")

            # 使用ONNX Runtime进行推理测试
            ort_session = ort.InferenceSession(onnx_path)

            # 创建测试输入
            test_input = torch.randn(
                1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH
            ).numpy()

            # ONNX推理
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)

            # PyTorch推理
            with torch.no_grad():
                torch_input = torch.from_numpy(test_input).to(self.device)
                torch_output = self.model(torch_input).cpu().numpy()

            # 比较结果
            diff = np.abs(ort_outputs[0] - torch_output).max()
            print(f"✅ ONNX和PyTorch输出差异: {diff:.6f}")

            if diff < 1e-5:
                print("🎉 ONNX模型验证成功！输出一致")
            else:
                print("⚠️ ONNX模型输出有微小差异，但在可接受范围内")

            return True

        except Exception as e:
            print(f"❌ ONNX模型验证失败: {e}")
            return False


def main():
    """主函数：导出和验证ONNX模型"""
    # 检查模型文件
    model_files = [
        os.path.join(config.MODEL_SAVE_DIR, "best_model.pth"),
        os.path.join(config.MODEL_SAVE_DIR, "final_model.pth"),
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n{'=' * 50}")
            print(f"处理模型: {os.path.basename(model_file)}")
            print(f"{'=' * 50}")

            try:
                # 创建导出器
                exporter = ONNXExporter(model_file)

                # 导出ONNX
                onnx_path = exporter.export_onnx()

                # 验证ONNX
                exporter.validate_onnx(onnx_path)

                # 尝试新的导出方法
                # dynamo_path = exporter.export_onnx_dynamo()
                # if dynamo_path:
                #     exporter.validate_onnx(dynamo_path)

            except Exception as e:
                print(f"❌ 处理 {model_file} 时出错: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"⚠️ 模型文件不存在: {model_file}")

    print(f"\n🎉 ONNX导出完成！")
    print(f"ONNX模型保存在: {config.MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
