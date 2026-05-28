import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import math


def sliding_window_detection(model, image, window_size=1024, overlap_ratio=0.25, conf_threshold=0.3):
    """
    使用滑动窗口进行目标检测
    """
    height, width = image.shape[:2]

    # 计算步长（根据重叠比例）
    step_size = int(window_size * (1 - overlap_ratio))

    all_detections = []

    # 滑动窗口
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            # 提取窗口区域
            window = image[y:y + window_size, x:x + window_size]

            # 如果窗口大小不足，跳过
            if window.shape[0] != window_size or window.shape[1] != window_size:
                continue

            # 使用模型进行预测
            results = model(window, conf=conf_threshold)

            # 处理检测结果
            for r in results:
                if r.masks is not None:
                    for i, mask in enumerate(r.masks.data):
                        # 转换掩码为numpy数组
                        mask_np = mask.cpu().numpy()

                        # 调整掩码坐标到原图坐标系
                        mask_resized = np.zeros((height, width), dtype=np.uint8)
                        mask_resized[y:y + window_size, x:x + window_size] = (mask_np * 255).astype(np.uint8)

                        # 获取边界框
                        if r.boxes is not None:
                            box = r.boxes.xyxy[i].cpu().numpy()
                            box[0] += x
                            box[1] += y
                            box[2] += x
                            box[3] += y

                            all_detections.append({
                                'mask': mask_resized,
                                'box': box,
                                'conf': r.boxes.conf[i].cpu().numpy(),
                                'class': r.boxes.cls[i].cpu().numpy()
                            })

    return all_detections


def merge_detections(detections, iou_threshold=0.5):
    """
    合并重叠的检测结果
    """
    if not detections:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)

    merged = []

    for det in detections:
        if det['conf'] < 0.3:  # 置信度阈值
            continue

        # 检查是否与已合并的结果重叠
        should_merge = False
        for merged_det in merged:
            iou = calculate_iou(det['box'], merged_det['box'])
            if iou > iou_threshold:
                should_merge = True
                break

        if not should_merge:
            merged.append(det)

    return merged


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def apply_segmentation_to_image(image, detections):
    """
    将分割结果应用到图像上
    """
    result_image = image.copy()

    for det in detections:
        mask = det['mask']

        # 创建彩色掩码
        color_mask = np.zeros_like(result_image)
        color_mask[:, :, 1] = mask  # 绿色通道

        # 应用掩码到图像
        result_image = cv2.addWeighted(result_image, 1, color_mask, 0.5, 0)

        # 绘制边界框
        box = det['box'].astype(int)
        cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 添加置信度标签
        label = f"{det['conf']:.2f}"
        cv2.putText(result_image, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return result_image


def process_images_with_sliding_window():
    """
    主处理函数
    """
    # 设置路径
    model_path = r"E:\ultralytics-main\训练结果漏铜\漏铜1280_window2\weights\best.pt"
    input_dir = r"E:\project\漏铜\XYD-901-1113-漏铜-FOV\漏铜原图集"
    output_dir = r"E:\ultralytics-main\电路板漏铜件检测\out_images"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    model = YOLO(model_path)
    print(f"模型加载完成: {model_path}")

    # 获取所有图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))

    if not image_files:
        print("未找到图像文件！")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    for i, image_path in enumerate(image_files):
        print(f"处理图像 {i + 1}/{len(image_files)}: {image_path.name}")

        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 使用滑动窗口进行检测
            detections = sliding_window_detection(model, image, window_size=1024)

            # 合并重叠的检测结果
            merged_detections = merge_detections(detections)

            print(f"  检测到 {len(merged_detections)} 个缺陷")

            # 应用分割结果到图像
            result_image = apply_segmentation_to_image(image, merged_detections)

            # 保存结果图像（BMP格式）
            output_filename = f"{image_path.stem}_segmented.bmp"
            output_path = os.path.join(output_dir, output_filename)

            # 使用PIL保存BMP格式，避免中文路径问题
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(result_rgb)
            pil_result.save(output_path, 'BMP')

            # 保存检测信息到文本文件
            info_filename = f"{image_path.stem}_detection_info.txt"
            info_path = os.path.join(output_dir, info_filename)

            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"图像: {image_path.name}\n")
                f.write(f"总缺陷数: {len(merged_detections)}\n")
                f.write(f"窗口大小: 1024x1024\n")
                f.write(f"重叠比例: 0.25\n\n")

                for j, det in enumerate(merged_detections):
                    box = det['box']
                    f.write(f"缺陷 {j + 1}:\n")
                    f.write(f"  置信度: {det['conf']:.3f}\n")
                    f.write(f"  类别: {det['class']}\n")
                    f.write(f"  边界框: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})\n")
                    f.write(f"  中心点: ({(box[0] + box[2]) / 2:.1f}, {(box[1] + box[3]) / 2:.1f})\n\n")

            print(f"  结果已保存: {output_filename}")

        except Exception as e:
            print(f"  处理图像时出错: {e}")
            continue

    print(f"\n处理完成！结果保存在: {output_dir}")
    print(f"共处理 {len(image_files)} 个图像")


if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行推理")

    process_images_with_sliding_window()