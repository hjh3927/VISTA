import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import color_similarity, find_background_seed, is_mask_included, mask_color_Kmeans

def sam(image, masks_path, pred_iou_thresh=0.80, stability_score_thresh=0.90, crop_n_layers=1, model_type='vit_h', checkpoint_path='', device='cpu'):
    """
    利用 SAM 模型生成掩码，并保存到 masks_path，返回生成的文件路径列表
    """
    # 初始化模型
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,             # 每侧采样点数
        pred_iou_thresh=pred_iou_thresh,           # 略微降低 IoU 阈值，防止过多小碎片
        stability_score_thresh=stability_score_thresh,    # 提高稳定性阈值，过滤低质量掩码
        crop_n_layers=crop_n_layers,                # 提高裁剪层数，使得整体结构更连贯
        min_mask_region_area=10         # 增大最小区域面积，去除极小碎片
    )

    st = time.time()

    print("运行 SAM 模型生成掩码...")
    masks = mask_generator.generate(image)
    print(f"生成 {len(masks)} 个掩码，耗时--------------->: {time.time()-st:.2f} s")

     # 根据掩码的面积进行排序
    mask_area_list = []

    for i, mask in enumerate(masks):
        # 获取掩码的二值化图像
        mask_data = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
        # 计算每个掩码的面积（即图像中值为 255 的像素数量）
        area = np.sum(mask_data == 255)

        # 将面积和对应的掩码索引保存到列表中
        mask_area_list.append((i, area, mask_data))

    # 按照面积从大到小排序
    sorted_mask_area_list = sorted(mask_area_list, key=lambda x: x[1], reverse=True)

    # 按照排序后的顺序保存掩码，使用新的排序顺序索引作为文件名
    mask_path_list = []
    for new_idx, (orig_idx, area, mask_data) in enumerate(sorted_mask_area_list):
        mask_img = Image.fromarray(mask_data)
        mask_file_path = os.path.join(masks_path, f'{new_idx}.png')
        mask_img.save(mask_file_path)
        mask_path_list.append(mask_file_path)

    return mask_path_list


def preprocessing_mask(mask_img_list, output_path, target_image, min_area=100, iou_threshold=0.8, pre_color_threshold=0.08, device='cpu'):
    """
    预处理二值掩码：
      1. 分割连通区域并保存有效的掩码。
      2. 移除交并比高于阈值的重复掩码。
      3. 移除与已有掩码颜色相似且包含的掩码。
    
    参数：
        mask_img_list (list): 输入的掩码图像路径列表。
        output_path (str): 输出目录路径。
        target_image (numpy.ndarray): 目标图像。
        min_area (int): 最小有效区域面积，面积小于该值的掩码将被忽略。
        iou_threshold (float): 交并比阈值，高于该阈值的掩码将被认为是重复的并被移除。
        pre_color_threshold (float): 颜色相似性阈值，低于该值认为颜色相似。
        device (str): 计算设备（'cpu' 或 'cuda'）。
    
    返回：
        tuple: (sorted_mask_paths, index_mask_dict)
            - sorted_mask_paths (list): 处理后的掩码路径列表（按面积从大到小排序）。
            - index_mask_dict (dict): 索引到掩码信息的映射，键按面积排序（1 为最大）。
    """
    st = time.time()
    print("预处理掩码...") 
    
    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_img_list]
    processed_masks = []  # 已处理的掩码列表
    pre_mask_list = []   # 存储 (面积, 掩码) 的列表
    index_mask_dict = {} # 原始索引到掩码的映射
    
    cnt = 1

    for count, image in enumerate(masks):
        flood_fill_image = image.copy()
        h, w = image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        seed = find_background_seed(flood_fill_image)
        cv2.floodFill(flood_fill_image, mask, seed, 255)
        filled_image = cv2.bitwise_or(image, cv2.bitwise_not(flood_fill_image))
        num_labels, labels = cv2.connectedComponents(filled_image)
        
        for i in range(1, num_labels):
            single_region = np.where(labels == i, 255, 0).astype(np.uint8)
            single_region_area = (single_region == 255).sum().item()
            if single_region_area <= min_area:
                print(f"区域面积 {single_region_area} 小于 {min_area}，跳过")
                continue

            # 获取当前掩码的颜色
            mask_image = Image.fromarray(single_region).convert('L')
            rgb_color = mask_color_Kmeans(target_image, mask_image)
            current_color = torch.tensor(rgb_color, device=device) / 255.0

            # 检查是否已有相似颜色的掩码并且被包含
            is_duplicate = False
            if processed_masks:
                # 处理已有掩码
                existing_masks_tensor = torch.stack([torch.from_numpy(m).to(device) for m in processed_masks], dim=0)
                mask_tensor = torch.from_numpy(single_region).to(device)
                # 计算交并比 (IoU)
                intersections = torch.logical_and(mask_tensor.unsqueeze(0), existing_masks_tensor).sum(dim=(1, 2))
                unions = torch.logical_or(mask_tensor.unsqueeze(0), existing_masks_tensor).sum(dim=(1, 2))
                ious = intersections / unions
                max_iou = ious.max().item()
                if max_iou > iou_threshold:
                    is_duplicate = True
                    print(f"掩码 {count}_{i} 与现有掩码重复，交并比为 {max_iou:.4f}，被跳过")

            # 检查颜色相似性并包含
            if not is_duplicate:
                for existing_mask_info in index_mask_dict.values():
                    existing_color = existing_mask_info["color"].cpu()
                    if color_similarity(existing_color, current_color, device) < pre_color_threshold:
                        existing_mask = existing_mask_info["mask"]
                        if is_mask_included(single_region, existing_mask, iou_threshold):
                            print(f"跳过第 {count}_{i} 个掩码，因为它与已有掩码颜色相似并且被包含")
                            is_duplicate = True
                            break

            if not is_duplicate:
                # 存储当前掩码信息
                index_mask_dict[cnt] = {"color": current_color, "mask": single_region}  # 存储 numpy 数组
                pre_mask_list.append((single_region_area, single_region))  # 记录面积、掩码和原始 cnt
                processed_masks.append(single_region)  # 保存已处理的掩码
                cnt += 1

    # 按照面积从大到小排序
    pre_mask_list.sort(key=lambda x: x[0], reverse=True)

    # 按照排序后的顺序保存掩码
    for idx, (_, mask) in enumerate(pre_mask_list):
        output_file_path = os.path.join(output_path, f'{idx}.png')
        Image.fromarray(mask).save(output_file_path)

    print(f"预处理掩码耗时--------------->: {time.time()-st:.2f} s")
    
    # 返回排序后的路径列表
    sorted_mask_paths = [os.path.join(output_path, f'{idx}.png') for idx, _ in enumerate(pre_mask_list)]
    return sorted_mask_paths


