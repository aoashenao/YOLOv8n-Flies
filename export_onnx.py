"""
导出YOLO模型为ONNX格式
适用于边缘设备部署
"""

from ultralytics import YOLO
from pathlib import Path


def export_model_to_onnx(pt_path, simplify=True, opset=11):
    """
    导出.pt模型为ONNX格式

    参数:
        pt_path: .pt模型路径
        simplify: 是否简化ONNX图（推荐True）
        opset: ONNX opset版本（树莓派推荐11或12）
    """

    print(f"\n{'=' * 60}")
    print(f"开始导出: {Path(pt_path).name}")
    print(f"{'=' * 60}")

    # 加载模型
    model = YOLO(pt_path)

    # 导出ONNX
    model.export(
        format='onnx',
        simplify=simplify,
        opset=opset,
        imgsz=640,  # 输入尺寸
        dynamic=False  # 固定输入尺寸（树莓派推荐）
    )

    onnx_path = str(pt_path).replace('.pt', '.onnx')

    print(f"\n✅ ONNX导出成功!")
    print(f"   路径: {onnx_path}")

    # 检查文件大小
    import os
    pt_size = os.path.getsize(pt_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

    print(f"   原始模型: {pt_size:.2f} MB")
    print(f"   ONNX模型: {onnx_size:.2f} MB")
    print(f"{'=' * 60}\n")

    return onnx_path


if __name__ == '__main__':
    # 模型路径
    baseline_pt = r"runs\detect\V6_Baseline\weights\best.pt"
    flies_pt = r"runs\detect\V6_CG_All_P3P4V2\weights\best.pt"

    print("\n🚀 开始导出模型为ONNX格式")
    print("=" * 60)

    # 导出Baseline
    baseline_onnx = export_model_to_onnx(baseline_pt)

    # 导出Flies
    flies_onnx = export_model_to_onnx(flies_pt)

    print("\n✅ 所有模型导出完成!")
    print("\n📦 导出文件:")
    print(f"   1. {baseline_onnx}")
    print(f"   2. {flies_onnx}")
    print("\n💡 下一步: 将这两个.onnx文件复制到树莓派")