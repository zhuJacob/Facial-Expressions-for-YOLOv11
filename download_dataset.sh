#!/bin/bash
# Download dataset from Kaggle (requires kaggle CLI configured)
# Dataset source: https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo

# Usage:
# 1. Install kaggle and configure (https://github.com/Kaggle/kaggle-api)
# 2. Run this script from the repo root to download and unzip dataset

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

DATASET=aklimarimi/8-facial-expressions-for-yolo
OUTDIR=data

mkdir -p ${OUTDIR}

echo "Downloading dataset: ${DATASET} -> ${OUTDIR}"
kaggle datasets download -d ${DATASET} -p ${OUTDIR} --unzip

echo "Done. Place images and labels into train/valid/test following data.yaml paths or adjust data.yaml accordingly."
