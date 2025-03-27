import os
import cv2
import pydiffvg

from config import DATA_PATH, OUT_PATH, ORIGIN_MASKS_PATH, PRE_MASKS_PATH, INIT_SVG_PATH, OPTIM_SVG_PATH, TARGET_IMAGE_PATH, CHECKPOINT_PATH
from config import MODEL_TYPE, DEVICE, MIN_AREA, MAX_ERROR, LINE_THRESHOLD, NUM_ITERS, LAMBDA1, LAMBDA2
from preprocessing import load_and_resize, save_target_image
from sam_inference import sam, preprocessing_mask
from svg_generator import generate_init_svg, svg_optimize

def main():
    # 创建输出目录
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(ORIGIN_MASKS_PATH, exist_ok=True)
    os.makedirs(PRE_MASKS_PATH, exist_ok=True)
    os.makedirs(INIT_SVG_PATH, exist_ok=True)
    os.makedirs(OPTIM_SVG_PATH, exist_ok=True)
    os.makedirs(TARGET_IMAGE_PATH, exist_ok=True)
    
    # 图像预处理与保存目标图像
    image_resized = load_and_resize(DATA_PATH)
    target_img_path = save_target_image(image_resized, TARGET_IMAGE_PATH, os.path.basename(DATA_PATH))
    
    # 读取并转换目标图像（用于后续处理）
    target_image = cv2.imread(target_img_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    
    # 设置设备
    pydiffvg.set_device(DEVICE)
    
    # SAM 掩码生成与预处理
    mask_path_list = sam(target_img_path, ORIGIN_MASKS_PATH, model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    pre_mask_path_list = preprocessing_mask(mask_path_list, PRE_MASKS_PATH, min_area=MIN_AREA)
    
    shapes = []
    shape_groups = []
    # 初始化 SVG
    shapes, shape_groups = generate_init_svg(shapes, shape_groups, DEVICE, pre_mask_path_list, target_image, out_svg_path=INIT_SVG_PATH, max_error=MAX_ERROR, line_threshold=LINE_THRESHOLD)
    
    # 优化 SVG
    svg_optimize(shapes, shape_groups, target_image, DEVICE, OPTIM_SVG_PATH, num_iters=NUM_ITERS, lamda1=LAMBDA1, lamda2=LAMBDA2)
    
    print(f"处理完成，输出目录：{OUT_PATH}")

if __name__ == '__main__':
    main()
