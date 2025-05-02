import json
import os
import time
import cv2
import pydiffvg

from sam_inference import sam, preprocessing_mask
from svg_generator import generate_init_svg, svg_optimize
from utils import add_to_file, compute_path_point_nums, load_and_resize, save_target_image
from config import CHECKPOINT_PATH, MODEL_TYPE, DEVICE, TEMP_OUTPUTS_DIR


def img_to_svg(image_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold):
    st = time.time()

    # 数据和输出路径
    file_name = os.path.basename(image_path)
    out_path = os.path.join(TEMP_OUTPUTS_DIR, f"{file_name.split('.')[0]}")  # 使用文件名（不含扩展名）创建目录

    # 输出子目录
    origin_masks_path = os.path.join(out_path, "origin_masks")
    pre_masks_path = os.path.join(out_path, "pre_masks")
    init_svg_path = os.path.join(out_path, "init_svgs")
    optim_svg_path = os.path.join(out_path, "optim_svgs")
    target_image_path = os.path.join(out_path, "target_img")
    # 创建输出目录
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(origin_masks_path, exist_ok=True)
    os.makedirs(pre_masks_path, exist_ok=True)
    os.makedirs(init_svg_path, exist_ok=True)
    os.makedirs(optim_svg_path, exist_ok=True)
    os.makedirs(target_image_path, exist_ok=True)

    # 图像预处理
    image_resized = load_and_resize(image_path, target_size)
    # 保存目标图像
    target_img_path = save_target_image(image_resized, target_image_path, file_name)
 
    # 读取并转换目标图像（用于后续处理）
    target_image = cv2.imread(target_img_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 设置设备
    pydiffvg.set_device(DEVICE)

    # SAM 掩码生成与预处理
    mask_path_list = sam(target_image, origin_masks_path, pred_iou_thresh, stability_score_thresh, crop_n_layers, model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    pre_mask_path_list = preprocessing_mask(mask_path_list, pre_masks_path, target_image, min_area=min_area, pre_color_threshold=pre_color_threshold, device=DEVICE)

    shapes = []
    shape_groups = []
    frames = []

    # 初始化 SVG
    shapes, shape_groups, frames, index_mask_dict = generate_init_svg(shapes, shape_groups, DEVICE, pre_mask_path_list, target_image, frames, out_svg_path=init_svg_path, max_error=bzer_max_error, line_threshold=line_threshold, is_stroke=is_stroke)
    
    # 优化 SVG
    svg_path, gif_path, shapes, shape_groups, current_loss = svg_optimize(shapes, shape_groups, target_image, DEVICE, optim_svg_path, frames, index_mask_dict, is_stroke=is_stroke, Points_lr=learning_rate, num_iters=num_iters, rm_color_threshold=rm_color_threshold)
    path_point_nums = compute_path_point_nums(shapes)
    
    all_time = time.time()-st
    shapes_count = len(shapes)
    print(f"处理完成，输出目录：{out_path}")
    print(f"===========================================")
    print(f'Time Consuming: {all_time:.2f} s')
    print(f'Shapes: {shapes_count}')
    print(f"Path Point Numbers: {path_point_nums}")
    print(f"MES Loss: {current_loss:.4f}")
    print(f"===========================================")

    # 打包信息成 JSON 格式并保存
    result_info = {
        "output_directory": out_path,
        "target_size": target_size,
        "pred_iou_thresh": pred_iou_thresh,
        "stability_score_thresh": stability_score_thresh,
        "crop_n_layers": crop_n_layers,
        "min_area": min_area,
        "pre_color_threshold": pre_color_threshold,
        "line_threshold": line_threshold,
        "bzer_max_error": bzer_max_error,
        "learning_rate": learning_rate,
        "is_stroke": is_stroke,
        "num_iters": num_iters,
        "rm_color_threshold": rm_color_threshold,
        "time_consuming": round(all_time, 4),
        'path_point_nums': path_point_nums,
        "shapes": shapes_count,
        "path_point_nums": path_point_nums,
        "mes_loss": round(current_loss, 4),
    }

    # 保存到 ./temp_outputs/result.json
    info_file_path = os.path.join(out_path, 'info.json')
    result_file_path = os.path.join(out_path, 'result.json')
    add_to_file({'mse_loss': round(current_loss, 4)}, result_file_path)
    add_to_file({'path_point_nums': path_point_nums}, result_file_path)
    add_to_file({'shapes': shapes_count}, result_file_path)
    add_to_file({'time_consuming': round(all_time, 4)}, result_file_path)
    
    try:
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)
        print(f"JSON result saved to {result_file_path}")
    except Exception as e:
        print(f"Failed to save JSON result: {str(e)}")

    return svg_path, gif_path, out_path, all_time, shapes_count, current_loss, path_point_nums
