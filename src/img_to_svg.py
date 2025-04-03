import os
import time
import cv2
import pydiffvg
import torch

from preprocessing import load_and_resize, save_target_image
from sam_inference import sam, preprocessing_mask
from svg_generator import generate_init_svg, svg_optimize

# 模型配置
CHECKPOINT_PATH = os.path.join("/home/hjh/repository/AIVbyPS", "pretrained_checkpoint/sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def img_to_svg(image_path, min_area, max_error, line_threshold, learning_rate, num_iters):
    st = time.time()
    # 临时路径
    temp_path = os.path.join(os.path.dirname(__file__), '..', 'temp')

    # 数据和输出路径
    file_name = "tmp.jpg"
    out_path = os.path.join(temp_path, f"{file_name.split('.')[0]}")  # 使用文件名（不含扩展名）创建目录

    # 输出子目录
    ORIGIN_MASKS_PATH = os.path.join(out_path, "origin_masks")
    PRE_MASKS_PATH = os.path.join(out_path, "pre_masks")
    INIT_SVG_PATH = os.path.join(out_path, "init_svgs")
    OPTIM_SVG_PATH = os.path.join(out_path, "optim_svgs")
    TARGET_IMAGE_PATH = os.path.join(out_path, "target_img")
    # 创建输出目录
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(ORIGIN_MASKS_PATH, exist_ok=True)
    os.makedirs(PRE_MASKS_PATH, exist_ok=True)
    os.makedirs(INIT_SVG_PATH, exist_ok=True)
    os.makedirs(OPTIM_SVG_PATH, exist_ok=True)
    os.makedirs(TARGET_IMAGE_PATH, exist_ok=True)

    # 图像预处理与保存目标图像
    image_resized = load_and_resize(image_path)
    # 保存目标图像
    target_img_path = save_target_image(image_resized, TARGET_IMAGE_PATH, file_name)
 
    # 读取并转换目标图像（用于后续处理）
    target_image = cv2.imread(target_img_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 设置设备
    pydiffvg.set_device(DEVICE)

    # SAM 掩码生成与预处理
    mask_path_list = sam(target_image, ORIGIN_MASKS_PATH, model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    pre_mask_path_list = preprocessing_mask(mask_path_list, PRE_MASKS_PATH, min_area=min_area)

    shapes = []
    shape_groups = []
    # 初始化 SVG
    shapes, shape_groups = generate_init_svg(shapes, shape_groups, DEVICE, pre_mask_path_list, target_image, out_svg_path=INIT_SVG_PATH, max_error=max_error, line_threshold=line_threshold)
    
    # 优化 SVG
    svg_optimize(shapes, shape_groups, target_image, DEVICE, OPTIM_SVG_PATH, learning_rate=learning_rate, num_iters=num_iters)
    
    print(f"处理完成，输出目录：{out_path}")
    print(f"总耗时--------------->: {time.time()-st:.2f} s")
    return f'{OPTIM_SVG_PATH}/final.svg'
