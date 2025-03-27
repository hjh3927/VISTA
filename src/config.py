import torch
import os
import time

# 项目路径
PROJECT_PATH = '/home/hjh/repository/AIVbyPS'

# 数据和输出路径
FILE_NAME = "2Dhouse.jpg"
DATA_PATH = os.path.join(PROJECT_PATH, "data/demo", FILE_NAME)
T = time.time()
OUT_PATH = os.path.join(PROJECT_PATH, "out", f"{FILE_NAME}-{int(T)%20}")

# 输出子目录
ORIGIN_MASKS_PATH = os.path.join(OUT_PATH, "origin_masks")
PRE_MASKS_PATH = os.path.join(OUT_PATH, "pre_masks")
INIT_SVG_PATH = os.path.join(OUT_PATH, "init_svgs")
OPTIM_SVG_PATH = os.path.join(OUT_PATH, "optim_svgs")
TARGET_IMAGE_PATH = os.path.join(OUT_PATH, "target_img")

# 模型配置
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "pretrained_checkpoint/sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
