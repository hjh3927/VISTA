import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from config import DEVICE

def sam(image, masks_path, pred_iou_thresh=0.80, stability_score_thresh=0.90, model_type='vit_h', checkpoint_path='', device='cpu'):
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
        crop_n_layers=1,                # 提高裁剪层数，使得整体结构更连贯
        min_mask_region_area=50         # 增大最小区域面积，去除极小碎片
    )

    st = time.time()

    print("运行 SAM 模型生成掩码...")
    masks = mask_generator.generate(image)
    print(f"生成 {len(masks)} 个掩码，耗时--------------->: {time.time()-st:.2f} s")
    
    mask_path_list = []
    for i, mask in enumerate(masks):
        image_data = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(image_data)
        mask_file_path = os.path.join(masks_path, f'{i}.png')
        mask_img.save(mask_file_path)
        mask_path_list.append(mask_file_path)

    return mask_path_list

def sort_masks_by_size(mask_list):
    """
    根据掩码区域大小排序，面积较大的排在前面
    """
    transform = ToTensor()
    def get_mask_area(mask_path):
        mask_image = Image.open(mask_path)
        mask_tensor = transform(mask_image)
        return (mask_tensor == 1).sum().item()
    mask_areas = [(mask_path, get_mask_area(mask_path)) for mask_path in mask_list]
    sorted_mask_areas = sorted(mask_areas, key=lambda x: x[1], reverse=True)
    return [mask_path for mask_path, _ in sorted_mask_areas]

def preprocessing_mask(mask_img_list, output_path, min_area=100, iou_threshold=0.8):
    """
    预处理二值掩码：
      1. 分割连通区域并保存有效的掩码。
      2. 移除交并比高于阈值的重复掩码。
    
    参数：
        mask_img_list (list): 输入的掩码图像路径列表。
        output_path (str): 输出目录路径。
        min_area (int): 最小有效区域面积，面积小于该值的掩码将被忽略。
        iou_threshold (float): 交并比阈值，高于该阈值的掩码将被认为是重复的并被移除。
    
    返回：
        list: 处理后的掩码路径列表。
    """
    st = time.time()
    print("预处理掩码...") 
    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_img_list]
    processed_masks = []
    pre_mask_list = []
    for count, image in enumerate(masks):
        num_labels, labels = cv2.connectedComponents(image)
        for i in range(1, num_labels):
            single_region = np.where(labels == i, 255, 0).astype(np.uint8)
            single_region_area = (single_region == 255).sum().item()
            if single_region_area <= min_area:
                print(f"区域面积 {single_region_area} 小于 {min_area}，跳过")
                continue

            # 检查当前掩码与已处理掩码之间的交并比，去除重复掩码
            is_duplicate = False
            if processed_masks:
                existing_masks_tensor = torch.stack([torch.from_numpy(m) for m in processed_masks], dim=0).to(DEVICE)
                mask_tensor = torch.from_numpy(single_region).to(DEVICE)
                intersections = torch.logical_and(mask_tensor.unsqueeze(0), existing_masks_tensor).sum(dim=(1, 2))
                unions = torch.logical_or(mask_tensor.unsqueeze(0), existing_masks_tensor).sum(dim=(1, 2))
                ious = intersections / unions
                max_iou = ious.max().item()
                if max_iou > iou_threshold:
                    is_duplicate = True
                    print(f"掩码 {count}_{i} 与现有掩码重复，交并比为 {max_iou:.4f}，被跳过")

            if not is_duplicate:
                # 保存新的有效掩码
                output_filename = f"{count}_{i}.png"
                output_file_path = os.path.join(output_path, output_filename)
                Image.fromarray(single_region).save(output_file_path)
                pre_mask_list.append(output_file_path)
                processed_masks.append(single_region)  # 保存已处理的掩码

    final_mask_list = sort_masks_by_size(pre_mask_list)
    print(f"预处理掩码耗时--------------->: {time.time()-st:.2f} s")

    return final_mask_list
