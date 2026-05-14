import os
import sys
from ultralytics import YOLO


def train_yolo_model(model_path=None, use_gpu=True):
    """使用YOLOv11训练模型 - 防漏检优化版"""

    # 定义训练参数 - 针对降低漏检优化
    train_params = {
        'data': r'E:\ultralytics-main\电路板漏铜件检测\loutongtxt_滑动裁剪1280 overlap=0.2\data.yaml',
        'epochs': 100,  # 增加训练轮数，让模型充分学习[7](@ref)
        'imgsz': 1280,  # 保持高分辨率
        'batch': 2,  # 批次大小
        'workers': 0,  # 数据加载线程数
        'project': r'E:\ultralytics-main\训练结果漏铜',
        'name': '漏铜1280_防漏检优化',  # 实验名称
        'cache': 'ram',  # 缓存图像到内存以加快训练速度
        'conf': 0.15,  # 降低置信度阈值，提高召回率（关键防漏检参数）[7](@ref)
        'val': True,  # 启用验证以获取mAP指标
        'plots': True,  # 生成训练图表
        'verbose': True,  # 显示详细信息

        # ========== 新增：防漏检优化参数 ==========
        'iou': 0.5,  # IoU阈值[7](@ref)
        'lr0': 0.001,  # 初始学习率[7](@ref)
        'lrf': 0.01,  # 最终学习率
        'cos_lr': True,  # 启用余弦学习率调度，使训练更稳定[7](@ref)
        'weight_decay': 0.0005,  # 权重衰减[7](@ref)
        'optimizer': 'AdamW',  # 优化器[7](@ref)
        'warmup_epochs': 3.0,  # 预热轮数[7](@ref)
        'warmup_momentum': 0.8,  # 预热动量
        'close_mosaic': 10,  # 最后10个epoch关闭马赛克增强[7](@ref)
        'amp': True,  # 启用混合精度训练[7](@ref)

        # 数据增强参数 - 增强模型泛化能力，特别关注小目标[1](@ref)
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,  # 饱和度增强
        'hsv_v': 0.4,  # 亮度增强
        'degrees': 10.0,  # 旋转增强
        'translate': 0.1,  # 平移增强
        'scale': 0.5,  # 缩放增强
        'shear': 2.0,  # 剪切增强
        'perspective': 0.001,  # 透视变换
        'flipud': 0.5,  # 上下翻转概率
        'fliplr': 0.5,  # 左右翻转概率
        'mosaic': 1.0,  # 马赛克增强概率
        'mixup': 0.1,  # MixUp增强概率[7](@ref)

        # 损失函数权重调整 - 提高检测敏感度[7](@ref)
        'box': 8.0,
        'cls': 0.6,  # 调整分类损失权重
        'dfl': 1.5,  # 调整DFL损失权重
        'label_smoothing': 0.05,  # 轻微标签平滑，防止过度自信

        # 训练策略优化
        'patience': 30,  # 早停耐心值[7](@ref)
        'save_period': 10,  # 保存周期
    }

    # 模型路径处理 - 增强灵活性
    if model_path and os.path.exists(model_path):
        train_params['model'] = model_path
        print(f"使用指定模型: {model_path}")
    elif model_path and not os.path.exists(model_path):
        # 如果指定路径不存在，尝试使用默认名称从预训练模型加载
        print(f"警告: 指定的模型路径不存在: {model_path}")
        print("尝试从Ultralytics加载预训练模型...")
        try:
            # 尝试直接加载模型名称（如yolo11s-seg.pt）
            model = YOLO(model_path.split('/')[-1])  # 提取模型名称
            train_params['model'] = model_path.split('/')[-1]
            print(f"使用预训练模型: {train_params['model']}")
        except:
            train_params['model'] = 'yolo11s-seg.pt'
            print(f"使用默认模型: {train_params['model']}")
    else:
        train_params['model'] = 'yolo11s-seg.pt'
        print("使用默认模型: yolo11s-seg.pt")

    # 设置设备
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                train_params['device'] = 0  # 使用第一个GPU
                print("使用GPU进行训练")
            else:
                train_params['device'] = 'cpu'
                print("GPU不可用，使用CPU进行训练")
        except ImportError:
            train_params['device'] = 'cpu'
            print("PyTorch未安装，使用CPU进行训练")
    else:
        train_params['device'] = 'cpu'
        print("使用CPU进行训练")

    print("=" * 60)
    print("训练配置摘要 (防漏检优化):")
    print(f"模型: {train_params['model']}")
    print(f"数据集: {train_params['data']}")
    print(f"训练轮数: {train_params['epochs']} (增加)")
    print(f"图像尺寸: {train_params['imgsz']}")
    print(f"批次大小: {train_params['batch']}")
    print(f"置信度阈值: {train_params['conf']} (降低)")
    print(f"学习率: {train_params['lr0']}")
    print(f"优化器: {train_params['optimizer']}")
    print(f"数据增强: 增强小目标检测能力")
    print("=" * 60)
    print("\n开始训练...")

    # 检查数据集配置文件是否存在
    if not os.path.exists(train_params['data']):
        print(f"错误: 找不到数据集配置文件 {train_params['data']}")
        return None, None

    # 检查数据集配置文件中的路径是否正确
    try:
        import yaml
        with open(train_params['data'], 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # 检查关键路径
        dataset_root = data_config.get('path', '')
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(dataset_root):
            dataset_root = os.path.join(os.path.dirname(train_params['data']), dataset_root)

        for key in ['train', 'val']:
            if key in data_config:
                path = data_config[key]
                # 转换为绝对路径进行检查
                if not os.path.isabs(path):
                    # 如果是相对路径，相对于data.yaml文件所在目录
                    path = os.path.join(dataset_root, path)
                if not os.path.exists(path):
                    print(f"警告: 数据集配置文件中的{key}路径不存在: {data_config[key]}")
                    print(f"实际检查路径: {path}")
    except Exception as e:
        print(f"警告: 无法读取数据集配置文件: {e}")

    try:
        # 加载模型
        print("正在加载模型...")
        model = YOLO(train_params['model'])

        print("开始训练 (防漏检优化版)...")
        # 执行训练
        results = model.train(
            data=train_params['data'],
            epochs=train_params['epochs'],
            imgsz=train_params['imgsz'],
            batch=train_params['batch'],
            workers=train_params['workers'],
            project=train_params['project'],
            name=train_params['name'],
            cache=train_params['cache'],
            conf=train_params['conf'],
            iou=train_params['iou'],
            lr0=train_params['lr0'],
            lrf=train_params['lrf'],
            cos_lr=train_params['cos_lr'],
            weight_decay=train_params['weight_decay'],
            optimizer=train_params['optimizer'],
            warmup_epochs=train_params['warmup_epochs'],
            warmup_momentum=train_params['warmup_momentum'],
            close_mosaic=train_params['close_mosaic'],
            amp=train_params['amp'],
            hsv_h=train_params['hsv_h'],
            hsv_s=train_params['hsv_s'],
            hsv_v=train_params['hsv_v'],
            degrees=train_params['degrees'],
            translate=train_params['translate'],
            scale=train_params['scale'],
            shear=train_params['shear'],
            perspective=train_params['perspective'],
            flipud=train_params['flipud'],
            fliplr=train_params['fliplr'],
            mosaic=train_params['mosaic'],
            mixup=train_params['mixup'],
            box=train_params['box'],
            cls=train_params['cls'],
            dfl=train_params['dfl'],
            label_smoothing=train_params['label_smoothing'],
            val=train_params['val'],
            plots=train_params['plots'],
            verbose=train_params['verbose'],
            patience=train_params['patience'],
            save_period=train_params['save_period'],
            device=train_params.get('device', 'cpu')
        )

        # 训练完成后进行详细验证
        print("\n开始详细验证模型性能...")
        metrics = model.val()

        # 输出详细的验证结果
        print("=" * 50)
        print("验证结果详情:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"精确率 (Precision): {metrics.box.p:.4f}")
        print(f"召回率 (Recall): {metrics.box.r:.4f}")  # 重点关注召回率
        print("=" * 50)

        # 分析召回率结果
        recall = metrics.box.r
        if recall < 0.7:
            print("⚠️  召回率较低，可能存在漏检问题")
            print("建议进一步措施:")
            print("1. 检查标注质量，确保所有目标都已正确标注")
            print("2. 增加难例样本数量")
            print("3. 尝试进一步降低conf阈值到0.1")
        elif recall < 0.8:
            print("✅ 召回率尚可，可继续优化")
        else:
            print("🎉 召回率良好!")

        # 显示最佳模型路径
        best_model_path = os.path.join(train_params['project'], train_params['name'], "weights", "best.pt")
        print(f"\n最佳模型已保存至: {best_model_path}")
        print(f"最新模型已保存至: {os.path.join(train_params['project'], train_params['name'], 'weights', 'last.pt')}")

        print("训练完成!")
        return results, model

    except Exception as e:
        print("训练过程中出现错误:")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数"""
    print("YOLOv11 漏铜缺陷检测模型训练脚本 - 防漏检优化版")
    print("=" * 60)
    print("优化目标: 降低漏检率，提高召回率")
    print("=" * 60)

    # 检查数据集配置文件是否存在
    data_yaml_path = r'E:\ultralytics-main\电路板漏铜件检测\loutongtxt_滑动裁剪1280 overlap=0.2\data.yaml'
    if not os.path.exists(data_yaml_path):
        print(f"错误: 找不到数据集配置文件 {data_yaml_path}")
        print("请确保数据集配置文件存在")
        return

    # 解析命令行参数
    model_path = None
    use_gpu = True

    # 增强命令行参数解析
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.lower() == '--cpu':
                use_gpu = False
                print("检测到 --cpu 参数，将使用CPU进行训练")
            elif arg.lower() == '--model' and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                print(f"从命令行参数获取模型路径: {model_path}")
                i += 1  # 跳过下一个参数
            elif not arg.startswith('--'):
                # 如果不是选项参数，认为是模型路径
                model_path = arg
                print(f"从命令行参数获取模型路径: {model_path}")
            i += 1

    # 如果没有通过命令行指定模型路径，可以在这里设置默认值
    if model_path is None:
        model_path = 'yolo11s-seg.pt'  # 默认模型
        print(f"使用默认模型: {model_path}")

    # 检查GPU可用性
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("警告: GPU不可用，将使用CPU训练")
                use_gpu = False
        except ImportError:
            print("警告: PyTorch未安装，将使用CPU训练")
            use_gpu = False

    # 开始训练
    results, model = train_yolo_model(model_path, use_gpu)

    if results is not None:
        print("\n训练成功完成!")

        # 显示训练结果摘要
        if hasattr(results, 'results_dict'):
            print("\n最终训练结果摘要:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value:.4f}")

            # 提供后续优化建议
            print("\n" + "=" * 60)
            print("后续优化建议:")
            print("1. 如果仍有漏检，可进一步降低conf阈值到0.1")
            print("2. 检查训练数据标注质量")
            print("3. 增加难例样本进行数据增强")
            print("=" * 60)
    else:
        print("\n训练失败!")


if __name__ == "__main__":
    main()