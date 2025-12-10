#!/bin/bash
# 云服务器快速启动脚本

echo "=========================================="
echo "面部表情检测 YOLO11n 训练启动脚本"
echo "=========================================="

# 检查 Python
echo -e "\n检查 Python 环境..."
python --version || python3 --version

# 检查 CUDA
echo -e "\n检查 CUDA/GPU..."
nvidia-smi

# 检查磁盘空间
echo -e "\n检查磁盘空间..."
df -h | grep -E "Filesystem|/dev/"
echo "提示: 建议将数据集解压到较大的数据盘 (通常挂载在 /root/autodl-tmp 或 /mnt/data)"

# 检查必需文件
echo -e "\n检查必需文件..."
required_files=("data.yaml" "train_yolo11n.py" "train/images" "valid/images" "test/images")
all_exist=true

for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo -e "\n错误: 缺少必需文件，请检查!"
    exit 1
fi

# 安装依赖
echo -e "\n安装/更新 Ultralytics..."
pip install ultralytics -q

# 显示配置信息
echo -e "\n=========================================="
echo "训练配置"
echo "=========================================="
echo "图像尺寸: 640x640"
echo "Batch Size: 128 (高性能模式)"
echo "最大轮数: 100"
echo "早停耐心: 20"
echo "Workers: 8"
echo "Cache: False (SSD直读模式)"
echo ""

# 询问运行模式
echo "选择运行模式:"
echo "1) 前台运行 (可实时查看输出)"
echo "2) 后台运行 (使用 nohup)"
read -p "请选择 [1/2]: " mode

if [ "$mode" = "2" ]; then
    echo -e "\n以后台模式启动训练..."
    nohup python train_yolo11n.py > training.log 2>&1 &
    echo "训练已在后台启动!"
    echo "查看日志: tail -f training.log"
    echo "查看进程: ps aux | grep train_yolo11n.py"
else
    echo -e "\n以前台模式启动训练..."
    echo "按 Ctrl+C 可以中断训练，权重会自动保存"
    echo "下次运行会自动从中断处继续"
    echo ""
    python train_yolo11n.py
fi
