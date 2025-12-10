from ultralytics import YOLO
import os
from pathlib import Path

# 配置参数
data_yaml = "data.yaml"  # 使用相对路径，适合云服务器部署
project_name = "facial_expression_yolo11n"
experiment_name = "exp"

# 检查是否存在之前的训练权重
weights_dir = Path(project_name) / experiment_name / "weights"
last_weights = weights_dir / "last.pt"

# 判断是继续训练还是从头开始
if last_weights.exists():
    print(f"发现上次训练的权重: {last_weights}")
    print("将从上次中断的地方继续训练...")
    model = YOLO(str(last_weights))
    resume = True
else:
    print("开始新的训练...")
    model = YOLO('yolo11n.pt')  # 使用预训练的 YOLO11n 模型
    resume = False

# 训练参数 - 针对 RTX 6000 Ada (48GB) 优化
train_params = {
    'data': data_yaml,
    'epochs': 100,              # 训练轮数
    'imgsz': 640,               # 图像尺寸
    'batch': 256,               # 批次大小 (性能模式: 充分利用48GB显存)
    'device': 0,                # 使用 GPU 0
    'workers': 16,              # 增加数据加载线程 (22 vCPU 可用，设置 16 以匹配大 Batch)
    'cache': False,             # 禁用RAM缓存 (核心安全锁: 防止系统内存爆炸)
    'half': True,               # 使用半精度 (提升速度)
    'project': project_name,    # 项目名称
    'name': experiment_name,    # 实验名称
    'exist_ok': True,           # 允许覆盖已存在的项目
    'pretrained': True,         # 使用预训练权重
    'optimizer': 'auto',        # 优化器
    'verbose': True,            # 详细输出
    'seed': 42,                 # 随机种子
    'deterministic': True,      # 确定性训练
    'single_cls': False,        # 多类别检测
    'rect': False,              # 矩形训练
    'cos_lr': True,             # 余弦学习率调度
    'close_mosaic': 10,         # 最后N个epoch关闭mosaic增强
    'resume': resume,           # 是否继续训练
    'amp': True,                # 自动混合精度训练
    'fraction': 1.0,            # 使用全部数据
    'profile': False,           # 是否进行性能分析
    'freeze': None,             # 冻结层数
    'lr0': 0.01,                # 初始学习率
    'lrf': 0.01,                # 最终学习率 (lr0 * lrf)
    'momentum': 0.937,          # SGD 动量/Adam beta1
    'weight_decay': 0.0005,     # 权重衰减
    'warmup_epochs': 3.0,       # 预热轮数
    'patience': 20,             # 早停耐心值(20轮不改善则停止,节省成本)
    'warmup_momentum': 0.8,     # 预热初始动量
    'warmup_bias_lr': 0.1,      # 预热初始偏置学习率
    'box': 7.5,                 # box loss 权重
    'cls': 0.5,                 # cls loss 权重
    'dfl': 1.5,                 # dfl loss 权重
    'pose': 12.0,               # pose loss 权重
    'kobj': 1.0,                # keypoint obj loss 权重
    'label_smoothing': 0.0,     # 标签平滑
    'nbs': 64,                  # 名义批次大小
    'hsv_h': 0.015,             # 色调数据增强
    'hsv_s': 0.7,               # 饱和度数据增强
    'hsv_v': 0.4,               # 亮度数据增强
    'degrees': 0.0,             # 旋转数据增强
    'translate': 0.1,           # 平移数据增强
    'scale': 0.5,               # 缩放数据增强
    'shear': 0.0,               # 剪切数据增强
    'perspective': 0.0,         # 透视数据增强
    'flipud': 0.0,              # 上下翻转概率
    'fliplr': 0.5,              # 左右翻转概率
    'mosaic': 1.0,              # mosaic 数据增强概率
    'mixup': 0.0,               # mixup 数据增强概率(面部表情不需要)
    'copy_paste': 0.0,          # copy-paste 数据增强概率
    'auto_augment': 'randaugment',  # 自动数据增强策略
    'erasing': 0.0,             # 随机擦除概率(关闭以加速训练)
    'crop_fraction': 1.0,       # 裁剪比例
}

try:
    # 开始训练
    print("\n" + "="*80)
    print("开始训练 YOLO11n 面部表情检测模型")
    print("="*80)
    print(f"\n数据集: {data_yaml}")
    print(f"模型: YOLO11n")
    print(f"训练轮数: {train_params['epochs']}")
    print(f"批次大小: {train_params['batch']}")
    print(f"图像尺寸: {train_params['imgsz']}")
    print(f"设备: {train_params['device']}")
    print("\n按 Ctrl+C 可以中断训练,下次运行会自动继续训练\n")
    print("="*80 + "\n")
    
    # 训练模型
    results = model.train(**train_params)
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"\n最佳权重: {project_name}/{experiment_name}/weights/best.pt")
    print(f"最后权重: {project_name}/{experiment_name}/weights/last.pt")
    
except KeyboardInterrupt:
    print("\n" + "="*80)
    print("训练被手动中断")
    print("="*80)
    print(f"\n权重已保存至: {project_name}/{experiment_name}/weights/last.pt")
    print("下次运行此脚本将自动继续训练")
    
except Exception as e:
    print(f"\n训练过程中出现错误: {e}")
    print(f"权重可能已保存至: {project_name}/{experiment_name}/weights/last.pt")
    raise
