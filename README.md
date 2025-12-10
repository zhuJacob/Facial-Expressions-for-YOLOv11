# Facial Expressions for YOLOv11

This repository contains code and configuration to train and run a YOLOv11-based face expression detection model.

Dataset source

- The dataset used for training is derived from: https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo

Contents

- `train_yolo11n.py` - Training script configured for RTX 6000 (recommended). Adjust `batch` and `workers` as needed.
- `realtime_yolo_infer.py` - Realtime webcam/inference script using `models/best.pt`.
- `data.yaml` - Dataset configuration file.
- `start_training.sh` - Helper script to prepare environment and start training on cloud servers.
- `云服务器部署指南.md` - Deployment notes for cloud servers (SSD usage, cache settings).

Quick start

1. Download the dataset from Kaggle (see `download_dataset.sh` for instructions) and place/unzip it into `data/`.
2. Place pre-trained weights at `models/best.pt` if you want to run realtime inference.
3. Train:

```bash
python train_yolo11n.py
```

4. Run realtime inference:

```bash
python realtime_yolo_infer.py
```

License & Attribution

- Dataset source: https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo (please follow Kaggle's dataset terms)
- Code: MIT (you can change this as desired)
