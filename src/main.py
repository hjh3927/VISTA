import json
import os
import time
import cv2
import pydiffvg

from config import DATA_PATH, OUT_PATH, ORIGIN_MASKS_PATH, PRE_MASKS_PATH, INIT_SVG_PATH, OPTIM_SVG_PATH, TARGET_IMAGE_PATH, CHECKPOINT_PATH
from config import MODEL_TYPE, DEVICE, TARGET_SIZE, PREDICTION_IOU_THRESHOLD, STABILITY_SCORE_THRESHOLD, CROP_N_LAYERS, PRE_COLOR_THRESHOLD, MIN_AREA, BEZIER_MAX_ERROR, LINE_THRESHOLD, LEARNING_RATE, NUM_ITERS, IS_STROKE, RM_COLOR_THRESHOLD
from sam_inference import sam, preprocessing_mask
from svg_generator import generate_init_svg, svg_optimize
from utils import load_and_resize, save_target_image

def main():
    st = time.time()
    # 创建输出目录
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(ORIGIN_MASKS_PATH, exist_ok=True)
    os.makedirs(PRE_MASKS_PATH, exist_ok=True)
    os.makedirs(INIT_SVG_PATH, exist_ok=True)
    os.makedirs(OPTIM_SVG_PATH, exist_ok=True)
    os.makedirs(TARGET_IMAGE_PATH, exist_ok=True)

    # 图像预处理与保存目标图像
    image_resized = load_and_resize(DATA_PATH, TARGET_SIZE)
    # 保存目标图像
    target_img_path = save_target_image(image_resized, TARGET_IMAGE_PATH, os.path.basename(DATA_PATH))
 
    # 读取并转换目标图像（用于后续处理）
    target_image = cv2.imread(target_img_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 设置设备
    pydiffvg.set_device(DEVICE)

    # SAM 掩码生成与预处理
    mask_path_list = sam(target_image, ORIGIN_MASKS_PATH, PREDICTION_IOU_THRESHOLD, STABILITY_SCORE_THRESHOLD, CROP_N_LAYERS, model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    pre_mask_path_list = preprocessing_mask(mask_path_list, PRE_MASKS_PATH, target_image, min_area=MIN_AREA, device=DEVICE, pre_color_threshold=PRE_COLOR_THRESHOLD)

    shapes = []
    shape_groups = []
    frames = []
    # 初始化 SVG
    shapes, shape_groups, frames, index_mask_dict = generate_init_svg(shapes, shape_groups, DEVICE, pre_mask_path_list, target_image, frames, out_svg_path=INIT_SVG_PATH, max_error=BEZIER_MAX_ERROR, line_threshold=LINE_THRESHOLD, is_stroke=IS_STROKE)
    
    # 优化 SVG
    svg_path, gif_path, shapes, shape_groups, current_loss = svg_optimize(shapes, shape_groups, target_image, DEVICE, OPTIM_SVG_PATH, frames, index_mask_dict, is_stroke=IS_STROKE, Points_lr=LEARNING_RATE, num_iters=NUM_ITERS, rm_color_threshold=RM_COLOR_THRESHOLD)
    all_time = time.time()-st
    shapes_count = len(shapes)   
    print(f"处理完成，输出目录：{OUT_PATH}")

    print(f"===========================================")
    print(f'Time Consuming: {time.time()-st:.2f} s')
    print(f'Shapes: {len(shapes)}')
    print(f"MES Loss: {current_loss:.4f}")
    print(f"===========================================")

    # 打包信息成 JSON 格式并保存
    result_info = {
        "output_directory": OUT_PATH,
        "target_size": TARGET_SIZE,
        "pred_iou_thresh": PREDICTION_IOU_THRESHOLD,
        "stability_score_thresh": STABILITY_SCORE_THRESHOLD,
        "crop_n_layers": CROP_N_LAYERS,
        "min_area": MIN_AREA,
        "pre_color_threshold": PRE_COLOR_THRESHOLD,
        "line_threshold": LINE_THRESHOLD,
        "bzer_max_error": BEZIER_MAX_ERROR,
        "learning_rate": LEARNING_RATE,
        "is_stroke": IS_STROKE,
        "num_iters": NUM_ITERS,
        "rm_color_threshold": RM_COLOR_THRESHOLD,
        "time_consuming": f"{all_time:.2f} s",
        "shapes": shapes_count,
        "mes_loss": f"{current_loss:.4f}",
    }

    # 保存到 ./temp_outputs/result.json
    result_file_path = os.path.join(OUT_PATH, 'result.json')
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)
        print(f"JSON result saved to {result_file_path}")
    except Exception as e:
        print(f"Failed to save JSON result: {str(e)}")

if __name__ == '__main__':
    main()
