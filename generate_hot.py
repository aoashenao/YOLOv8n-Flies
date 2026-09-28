"""
分别保存每个热力图
为后续组合使用做准备
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_heatmap(model, img_path):
    """
    生成热力图

    返回:
        heatmap: 归一化热力图 [0, 1]
        img_rgb: 原始图像
    """
    # 读取图像
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 推理获取检测结果
    results = model(img_path, verbose=False)[0]

    # 创建热力图基础
    heatmap = np.zeros((h, w), dtype=np.float32)

    # 基于检测框生成高斯热力图
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 创建高斯分布
            y_grid, x_grid = np.ogrid[:h, :w]
            sigma_x = (x2 - x1) / 3
            sigma_y = (y2 - y1) / 3

            gaussian = score * np.exp(-((x_grid - cx) ** 2 / (2 * sigma_x ** 2) +
                                        (y_grid - cy) ** 2 / (2 * sigma_y ** 2)))
            heatmap += gaussian

    # 归一化到 [0, 1]
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap, img_rgb


def save_single_heatmap(heatmap, img_rgb, output_path, alpha=0.6):
    """
    保存单个热力图（不带标题和标注）

    参数:
        heatmap: 归一化热力图
        img_rgb: 原始图像
        output_path: 输出路径
        alpha: 热力图透明度
    """
    # 应用颜色映射
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # 叠加
    if img_rgb.shape[:2] != colored_heatmap.shape[:2]:
        colored_heatmap = cv2.resize(colored_heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, colored_heatmap, alpha, 0)

    # 保存（无边框，无坐标轴）
    fig = plt.figure(figsize=(img_rgb.shape[1] / 100, img_rgb.shape[0] / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(overlay)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_original_image(img_path, output_path):
    """
    保存原图（无标题）
    """
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(img_rgb.shape[1] / 100, img_rgb.shape[0] / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_rgb)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_detection_result(model, img_path, output_path):
    """
    保存检测结果（带标注框）
    """
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 推理
    results = model(img_path, verbose=False)[0]

    # 绘制检测框
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 保存
    fig = plt.figure(figsize=(img_rgb.shape[1] / 100, img_rgb.shape[0] / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_rgb)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("生成并保存独立热力图")
    print("=" * 60 + "\n")

    # 配置路径
    baseline_model_path =  r"D:\桌面\树莓派\YOLO部署\YOLO_V1\qq_3045834499\yolov8-42\42_demo\runs\2026V6\runs\detect\V6_Baseline\weights\best.pt"
    flies_model_path =  r"D:\桌面\树莓派\YOLO部署\YOLO_V1\qq_3045834499\yolov8-42\42_demo\runs\2026V6\runs\detect\V6_CG_All_P3P4V2\weights\best.pt"

    # 测试图片
    test_images = [
        r"D:\桌面\树莓派\YOLO部署\YOLO_V1\qq_3045834499\yolov8-42\42_demo\images\shadow_influ\0904_112546.jpg",
        r"D:\桌面\树莓派\YOLO部署\YOLO_V1\qq_3045834499\yolov8-42\42_demo\images\shadow_influ\0905_081849.jpg"
    ]

    # 输出目录
    output_dir = Path(r"D:\桌面\树莓派\YOLO部署\YOLO_V1\qq_3045834499\yolov8-42\42_demo\images\hot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查文件
    if not Path(baseline_model_path).exists():
        print(f"❌ 找不到 Baseline 模型")
        return

    if not Path(flies_model_path).exists():
        print(f"❌ 找不到 Flies 模型")
        return

    valid_images = [img for img in test_images if Path(img).exists()]
    if len(valid_images) < 2:
        print("❌ 需要至少2张测试图片")
        return

    # 加载模型
    print("⏳ 加载模型...")
    model_baseline = YOLO(baseline_model_path)
    model_flies = YOLO(flies_model_path)
    print("✅ 模型加载完成\n")

    # 处理每张图片
    for idx, img_path in enumerate(valid_images[:2], 1):
        print(f"{'=' * 60}")
        print(f"处理样本 {idx}: {Path(img_path).name}")
        print(f"{'=' * 60}")

        # 1. 保存原图
        original_path = output_dir / f"sample{idx}_original.png"
        save_original_image(img_path, original_path)
        print(f"  ✅ 原图: {original_path.name}")

        # 2. 保存检测结果（用Baseline）
        detection_path = output_dir / f"sample{idx}_detection.png"
        save_detection_result(model_baseline, img_path, detection_path)
        print(f"  ✅ 检测结果: {detection_path.name}")

        # 3. 生成并保存 Baseline 热力图
        print(f"  ⏳ 生成 Baseline 热力图...")
        heatmap_baseline, img_rgb = generate_heatmap(model_baseline, img_path)
        baseline_heatmap_path = output_dir / f"sample{idx}_baseline_heatmap.png"
        save_single_heatmap(heatmap_baseline, img_rgb, baseline_heatmap_path)
        print(f"  ✅ Baseline热力图: {baseline_heatmap_path.name}")

        # 4. 生成并保存 Flies 热力图
        print(f"  ⏳ 生成 Flies 热力图...")
        heatmap_flies, img_rgb = generate_heatmap(model_flies, img_path)
        flies_heatmap_path = output_dir / f"sample{idx}_flies_heatmap.png"
        save_single_heatmap(heatmap_flies, img_rgb, flies_heatmap_path)
        print(f"  ✅ Flies热力图: {flies_heatmap_path.name}")

        print()

    print("=" * 60)
    print("✅ 所有热力图已保存！")
    print("=" * 60)
    print(f"\n📁 保存位置: {output_dir}")
    print("\n生成的文件:")
    print("  - sample1_original.png          (样本1原图)")
    print("  - sample1_detection.png         (样本1检测结果)")
    print("  - sample1_baseline_heatmap.png  (样本1 Baseline热力图)")
    print("  - sample1_flies_heatmap.png     (样本1 Flies热力图)")
    print("  - sample2_original.png          (样本2原图)")
    print("  - sample2_detection.png         (样本2检测结果)")
    print("  - sample2_baseline_heatmap.png  (样本2 Baseline热力图)")
    print("  - sample2_flies_heatmap.png     (样本2 Flies热力图)")


if __name__ == '__main__':
    main()