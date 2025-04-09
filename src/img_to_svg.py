import os
import time
import cv2
import pydiffvg
import torch

from sam_inference import sam, preprocessing_mask
from svg_generator import generate_init_svg, svg_optimize
from utils import load_and_resize, save_target_image

# 项目路径
PROJECT_PATH = '/home/hjh/repository/VISTA'

# 模型配置
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "pretrained_checkpoint/sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def img_to_svg(image_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold):
    st = time.time()
    # 临时路径
    TEMP_PATH = os.path.join(PROJECT_PATH, 'temp_outputs')

    # # 数据和输出路径
    file_name = os.path.basename(image_path)
    OUT_PATH = os.path.join(TEMP_PATH, f"{file_name.split('.')[0]}")  # 使用文件名（不含扩展名）创建目录

    # 输出子目录
    ORIGIN_MASKS_PATH = os.path.join(OUT_PATH, "origin_masks")
    PRE_MASKS_PATH = os.path.join(OUT_PATH, "pre_masks")
    INIT_SVG_PATH = os.path.join(OUT_PATH, "init_svgs")
    OPTIM_SVG_PATH = os.path.join(OUT_PATH, "optim_svgs")
    TARGET_IMAGE_PATH = os.path.join(OUT_PATH, "target_img")
    # 创建输出目录
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(ORIGIN_MASKS_PATH, exist_ok=True)
    os.makedirs(PRE_MASKS_PATH, exist_ok=True)
    os.makedirs(INIT_SVG_PATH, exist_ok=True)
    os.makedirs(OPTIM_SVG_PATH, exist_ok=True)
    os.makedirs(TARGET_IMAGE_PATH, exist_ok=True)

    # 图像预处理与保存目标图像
    image_resized = load_and_resize(image_path, target_size)
    # 保存目标图像
    target_img_path = save_target_image(image_resized, TARGET_IMAGE_PATH, file_name)
 
    # 读取并转换目标图像（用于后续处理）
    target_image = cv2.imread(target_img_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 设置设备
    pydiffvg.set_device(DEVICE)

    # SAM 掩码生成与预处理
    mask_path_list = sam(target_image, ORIGIN_MASKS_PATH, pred_iou_thresh, stability_score_thresh, crop_n_layers, model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    pre_mask_path_list = preprocessing_mask(mask_path_list, PRE_MASKS_PATH, target_image, min_area=min_area, pre_color_threshold=pre_color_threshold, device=DEVICE)

    shapes = []
    shape_groups = []
    frames = []
    # 初始化 SVG
    frames = []
    shapes, shape_groups, frames, index_mask_dict = generate_init_svg(shapes, shape_groups, DEVICE, pre_mask_path_list, target_image, frames, out_svg_path=INIT_SVG_PATH, max_error=bzer_max_error, line_threshold=line_threshold, is_stroke=is_stroke)
    
    # 优化 SVG
    svg_path, gif_path, shapes, shape_groups, current_loss = svg_optimize(shapes, shape_groups, target_image, DEVICE, OPTIM_SVG_PATH, frames, index_mask_dict, is_stroke=is_stroke, learning_rate=learning_rate, num_iters=num_iters, rm_color_threshold=rm_color_threshold)
    
    print(f"处理完成，输出目录：{OUT_PATH}")

    print(f"===========================================")
    print(f'target_size : {target_size}, pred_iou_thresh : {pred_iou_thresh}, stability_score_thresh : {stability_score_thresh}, crop_n_layers : {crop_n_layers}, min_area :{min_area}, pre_color_threshold : {pre_color_threshold}')
    print(f'line_threshold : {line_threshold}, bzer_max_error : {bzer_max_error}, learning_rate : {learning_rate}, is_stroke : {is_stroke}, num_iters : {num_iters}, rm_color_threshold : {rm_color_threshold}')
    print(f'Time Consuming: {time.time()-st:.2f} s')
    print(f'Shapes: {len(shapes)}')
    print(f"MES Loss: {current_loss:.4f}")
    print(f"===========================================")
    return svg_path, gif_path
