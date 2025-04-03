import os
import time
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def sam(image, masks_path, model_type='vit_h', checkpoint_path='', device='cpu'):
    """
    利用 SAM 模型生成掩码，并保存到 masks_path，返回生成的文件路径列表
    """
    # 初始化模型
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    st = time.time()

    print("运行 SAM 模型生成掩码...")
    masks = mask_generator.generate(image)
    print(f"生成 {len(masks)} 个掩码，耗时 {time.time()-st:.2f} s")
    
    mask_path_list = []
    for i, mask in enumerate(masks):
        image_data = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(image_data)
        mask_file_path = os.path.join(masks_path, f'{i}.png')
        mask_img.save(mask_file_path)
        mask_path_list.append(mask_file_path)
    return mask_path_list

def find_background_seed(image):
    """
    找到图像边界上第一个为0的像素作为背景种子
    """
    h, w = image.shape
    border_pixels = [(x, 0) for x in range(w)] + [(x, h-1) for x in range(w)] + \
                    [(0, y) for y in range(h)] + [(w-1, y) for y in range(h)]
    for seed in border_pixels:
        if image[seed[1], seed[0]] == 0:
            return seed
    return (0, 0)


def compute_iou(mask1, mask2):
    """
    计算两个二值掩码之间的交并比（IoU）。
    
    参数：
        mask1 (np.ndarray): 第一个二值掩码，形状为 (height, width)。
        mask2 (np.ndarray): 第二个二值掩码，形状为 (height, width)。
        
    返回：
        float: 两个掩码之间的交并比（IoU）。
    """
    intersection = np.logical_and(mask1, mask2).sum()  # 交集
    union = np.logical_or(mask1, mask2).sum()  # 并集
    return intersection / union if union != 0 else 0  # 防止除零

def preprocessing_mask(mask_img_list, output_path, min_area=100, iou_threshold=0.8):
    """
    预处理二值掩码：
      1. 使用 floodFill 填充外部背景，去除孔洞。
      2. 分割连通区域并保存有效的掩码。
      3. 移除交并比高于阈值的重复掩码。
    
    参数：
        mask_img_list (list): 输入的掩码图像路径列表。
        output_path (str): 输出目录路径。
        min_area (int): 最小有效区域面积，面积小于该值的掩码将被忽略。
        iou_threshold (float): 交并比阈值，高于该阈值的掩码将被认为是重复的并被移除。
    
    返回：
        list: 处理后的掩码路径列表。
    """
    print("预处理掩码...")
    pre_mask_list = []
    processed_masks = []  # 用于保存处理后的掩码图像

    for count, mask_img_path in enumerate(mask_img_list):
        image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"无法读取 {mask_img_path}，跳过...")
            continue
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

            # 检查当前掩码与已处理掩码之间的交并比，去除重复掩码
            is_duplicate = False
            for existing_mask in processed_masks:
                iou = compute_iou(single_region, existing_mask)
                if iou > iou_threshold:
                    is_duplicate = True
                    print(f"掩码 {count}_{i} 与现有掩码重复，交并比为 {iou:.4f}，被跳过")
                    break

            if not is_duplicate:
                # 保存新的有效掩码
                output_filename = f"{count}_{i}.png"
                output_file_path = os.path.join(output_path, output_filename)
                Image.fromarray(single_region).save(output_file_path)
                pre_mask_list.append(output_file_path)
                processed_masks.append(single_region)  # 保存已处理的掩码

    final_mask_list = sort_masks_by_size(pre_mask_list)
    return final_mask_list


def sort_masks_by_size(mask_list):
    """
    根据掩码区域大小排序，面积较大的排在前面
    """
    from torchvision.transforms import ToTensor
    transform = ToTensor()
    def get_mask_area(mask_path):
        mask_image = Image.open(mask_path)
        mask_tensor = transform(mask_image)
        return (mask_tensor == 1).sum().item()
    mask_areas = [(mask_path, get_mask_area(mask_path)) for mask_path in mask_list]
    sorted_mask_areas = sorted(mask_areas, key=lambda x: x[1], reverse=True)
    return [mask_path for mask_path, _ in sorted_mask_areas]
