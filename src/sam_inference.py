import os
import time
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def sam(image_path, masks_path, model_type='vit_h', checkpoint_path='', device='cpu'):
    """
    利用 SAM 模型生成掩码，并保存到 masks_path，返回生成的文件路径列表
    """
    # 初始化模型
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    st = time.time()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def preprocessing_mask(mask_img_list, output_path, min_area=100):
    """
    预处理二值掩码：
      1. 使用 floodFill 填充外部背景，去除孔洞。
      2. 分割连通区域并保存有效的掩码。
    """
    print("预处理掩码...")
    pre_mask_list = []
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
        if num_labels > 2:
            print("-----检测到多个区域，跳过-----")
            continue
        for i in range(1, num_labels):
            single_region = np.where(labels == i, 255, 0).astype(np.uint8)
            single_region_area = (single_region == 255).sum().item()
            if single_region_area <= min_area:
                print(f"区域面积 {single_region_area} 小于 {min_area}，跳过")
                continue
            output_filename = f"{count}.png"
            output_file_path = os.path.join(output_path, output_filename)
            Image.fromarray(single_region).save(output_file_path)
            pre_mask_list.append(output_file_path)
    pre_mask_list = sort_masks_by_size(pre_mask_list)
    return pre_mask_list

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
