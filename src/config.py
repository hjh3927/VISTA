import uuid
import torch
import os

# 项目路径
PROJECT_PATH = '/home/hjh/repository/VISTA/VISTA'

# 数据和输出路径
FILE_NAME = "complex_5.png"
DATA_PATH = os.path.join(PROJECT_PATH, "dataset/", FILE_NAME)
unique_id = str(uuid.uuid4())[:8]
OUT_PATH = os.path.join(PROJECT_PATH, "out", f"{FILE_NAME}-{unique_id}")
TEMP_OUTPUTS_DIR = os.path.join(PROJECT_PATH, "temp_outputs")

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

#超参数配置
TARGET_SIZE = 512
PREDICTION_IOU_THRESHOLD = 0.85
STABILITY_SCORE_THRESHOLD = 0.93
CROP_N_LAYERS = 1
PRE_COLOR_THRESHOLD = 0.0
MIN_AREA = 20
BEZIER_MAX_ERROR = 1.0
LINE_THRESHOLD = 1.0
LEARNING_RATE = 0.1
NUM_ITERS = 1000
IS_STROKE = False
RM_COLOR_THRESHOLD = 0.01