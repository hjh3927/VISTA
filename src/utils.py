import json
import os
import cv2
import numpy as np
import pydiffvg
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
import torch
from PIL import Image

def load_and_resize(image_path: str, target_size: int = 512):
    """
    加载图像并转换为 RGB，如果有透明度则用白色背景填充。
    然后按比例缩放图像，返回 numpy 数组。
    """
    print("预处理目标图像...")
    image = Image.open(image_path)

    # 如果是调色板图像，先转成 RGBA
    if image.mode == "P":
        image = image.convert("RGBA")

    # 如果是带透明度的图像，使用白色背景合成
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))  # 白色背景
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    else:
        image = image.convert("RGB")

    # 缩放图像，保持宽高比
    w, h = image.size
    scale = target_size / max(w, h)
    target_size = (int(w * scale), int(h * scale))
    resized = image.resize(target_size, Image.Resampling.LANCZOS)

    return resized

  
def save_target_image(image, out_dir, file_name):
    out_file = os.path.join(out_dir, file_name)
    if not os.path.splitext(out_file)[1]:  # 检查是否有扩展名
        out_file += '.jpg'  # 如果没有，添加默认扩展名
    image.save(out_file)
    return out_file

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

def bezier_curve(t, P0, P1, P2, P3):
    """
    计算三阶贝塞尔曲线上的点，t 为 [0,1] 间的参数数组
    """
    t = np.array(t).reshape(-1, 1)
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

def fit_bezier_segment(points):
    """
    拟合单段贝塞尔曲线，返回四个控制点
    """
    n = len(points)
    t = np.linspace(0, 1, n)
    P0 = points[0]
    P3 = points[-1]
    P1_guess = points[int(n/3)]
    P2_guess = points[int(2*n/3)]
    def residuals(params):
        P1 = params[:2]
        P2 = params[2:]
        curve_points = bezier_curve(t, P0, P1, P2, P3)
        return (curve_points - points).flatten()
    result = least_squares(residuals, np.concatenate([P1_guess, P2_guess]))
    P1 = result.x[:2]
    P2 = result.x[2:]
    return P0, P1, P2, P3

def compute_error(points, P0, P1, P2, P3):
    """
    计算贝塞尔曲线与点集的最大误差
    """
    t = np.linspace(0, 1, len(points))
    curve_points = bezier_curve(t, P0, P1, P2, P3)
    errors = np.linalg.norm(curve_points - points, axis=1)
    return np.max(errors)

def point_to_line_distance(points, p1, p2):
    """
    计算点集到直线的距离
    """
    p1, p2, points = map(np.array, [p1, p2, points])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.zeros(len(points))
    point_vec = points - p1
    t = np.clip(np.dot(point_vec, line_vec) / (line_len**2), 0, 1)
    projections = p1 + t[:, None] * line_vec
    distances = np.linalg.norm(points - projections, axis=1)
    return distances

def fit_contour(contour_simple, contour, max_error=1.0, line_threshold=1.0):
    """
    使用最少的三阶贝塞尔曲线和直线拟合点集
    """
    structured_points = [contour_simple[0]]
    i = 0
    while i < len(contour_simple) - 1:
        for j in range(len(contour_simple)-1, i, -1):
            p1 = contour_simple[i]
            p2 = contour_simple[j]
            idx1 = np.where((contour == p1).all(axis=1))[0][0]
            idx2 = np.where((contour == p2).all(axis=1))[0][0]
            if idx2 >= idx1:
                segment = contour[idx1:idx2+1]
            else:
                segment = np.concatenate((contour[idx1:], contour[:idx2+1]))
            distances = point_to_line_distance(segment, p1, p2)
            max_distance = np.max(distances)
            if max_distance <= line_threshold:
                structured_points.append(p2)
                i = j
                break
            else:
                P0, P1, P2, P3 = fit_bezier_segment(segment)
                error = compute_error(segment, P0, P1, P2, P3)
                if error <= max_error:
                    structured_points.append([P1, P2, P3])
                    i = j
                    break
        if i != j :
            if max_distance <= error :
                structured_points.append(p2)
                i = j
            else :
                structured_points.append([P1, P2, P3])
                i = j
        
    return structured_points

def points_to_path(structured_points, closed=True):
    """
    将结构化的点列表转换为 pydiffvg.Path 对象
    """
    if not structured_points:
        return None
    points = []
    num_control_points = []
    start_point = structured_points[0]
    points.append(torch.tensor(start_point, dtype=torch.float32))
    for i, item in enumerate(structured_points[1:]):
        if isinstance(item, np.ndarray):
            points.append(torch.tensor(item, dtype=torch.float32))
            num_control_points.append(0)
        elif isinstance(item, list) and len(item) == 3:
            c1, c2, p2 = item
            points.extend([
                torch.tensor(c1, dtype=torch.float32),
                torch.tensor(c2, dtype=torch.float32),
                torch.tensor(p2, dtype=torch.float32)
            ])
            num_control_points.append(2)
        else:
            raise ValueError("列表元素必须是 numpy 数组或包含三个 numpy 数组的列表")
    points.pop()  # 移除最后一个多余点
    path = pydiffvg.Path(
        num_control_points=torch.tensor(num_control_points),
        points=torch.stack(points),
        stroke_width=torch.tensor(1.0),
        is_closed=closed
    )
    return path

def mask_to_path(mask, max_error=1.0, line_threshold=1.0):
    """
    根据二值 mask 提取轮廓并拟合生成 pydiffvg.Path 对象
    """
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return []
    contour = contours[0].squeeze()

    # 去除重复点，保留原始顺序
    unique_contour, unique_indices = np.unique(contour, axis=0, return_index=True)

    # 使用索引按原始顺序对去重后的点进行排序
    contour = contour[np.sort(unique_indices)]

    simplified = cv2.approxPolyDP(contour, 2.0, closed=True).squeeze()
    idx1 = np.where((contour == simplified[0]).all(axis=1))[0][0]
    if (contour[idx1-1] != simplified[-1]).any():
        simplified = np.vstack((simplified, contour[idx1-1]))
    structured_points = fit_contour(simplified, contour, max_error, line_threshold)
    path = points_to_path(structured_points, closed=True)
    return path

def mask_color_Kmeans(image, mask, n_clusters=1, threshold=0.9):
    """
    使用 K-means 聚类从图像对应的 mask 区域中提取主色，动态调整聚类数量
    """
    mask_np = np.array(mask)
    masked_pixels = image[mask_np > 0]
    
    # 计算 mask 像素占图像总像素的比例
    total_pixels = image.shape[0] * image.shape[1]
    mask_ratio = len(masked_pixels) / total_pixels
    
    # 如果 mask 占比超过阈值，判定为背景
    if mask_ratio > threshold:
        return (255, 255, 255)  
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(masked_pixels)
    
    main_color = kmeans.cluster_centers_[0].astype(int)
    
    return tuple(main_color)

def color_similarity(color1, color2, device):
    """
    计算两个颜色之间的欧氏距离，返回一个标量，值越小表示颜色越相似
    """
    color1 = color1.to(device)  # 确保 color1 在正确的设备上
    color2 = color2.to(device)  # 确保 color2 在正确的设备上
    return torch.sqrt(torch.sum((color1 - color2) ** 2))

def is_mask_included(current_mask, existing_mask, inclusion_threshold=0.8):
    """
    判断 current_mask 的大部分是否被 existing_mask 包含，基于交集和最小面积的比值进行判断。
    如果交集和较小 mask 的比值大于 inclusion_threshold，则认为 current_mask 被包含。
    
    参数：
        current_mask: 当前 mask，二值化后的 numpy 数组。
        existing_mask: 已有 mask，二值化后的 numpy 数组。
        inclusion_threshold: 包含判断的阈值，交集和较小面积的比值大于此值时，认为当前 mask 被包含。
    
    返回：
        bool: 如果 current_mask 被 existing_mask 完全包含，返回 True，否则返回 False。
    """
    # 将当前 mask 和已有 mask 转换为二值图
    current_mask_binary = (current_mask > 0).astype(np.uint8)
    existing_mask_binary = (existing_mask > 0).astype(np.uint8)

    # 计算交集区域
    intersection = cv2.bitwise_and(current_mask_binary, existing_mask_binary)

    # 计算交集的面积
    intersection_area = np.sum(intersection)

    # 计算当前 mask 和已有 mask 的面积
    current_area = np.sum(current_mask_binary)
    existing_area = np.sum(existing_mask_binary)

    # 获取较小的 mask 面积
    smaller_area = min(current_area, existing_area)

    # 防止除零错误
    if smaller_area == 0:
        return False

    # 计算交集和较小面积的比值
    inclusion_ratio = intersection_area / smaller_area

    # 判断比值是否大于阈值
    return inclusion_ratio >= inclusion_threshold


def render_svg_to_jpg(svg_path, output_jpg_path, width, height, background_color=(255, 255, 255), preserve_aspect_ratio=True):
    """
    Render an SVG file to a JPG image with specified width and height, scaling content to fill the canvas.
    
    Args:
        svg_path (str): Path to the input SVG file.
        output_jpg_path (str): Path to save the output JPG file.
        width (int): Desired width of the output image in pixels.
        height (int): Desired height of the output image in pixels.
        background_color (tuple): RGB color for the background (default: white, (255, 255, 255)).
        preserve_aspect_ratio (bool): If True, scale SVG content to preserve aspect ratio; if False, stretch to fill.
    
    Returns:
        bool: True if rendering is successful, False otherwise.
    """
    try:
        # Set device
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        
        # Load SVG file
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
        
        # Calculate scaling factors
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        
        if preserve_aspect_ratio:
            # Use the smaller scale to avoid stretching
            scale = min(scale_x, scale_y)
            scale_x = scale
            scale_y = scale
            # Center the content
            offset_x = (width - canvas_width * scale_x) / 2
            offset_y = (height - canvas_height * scale_y) / 2
        else:
            # Stretch to fill the canvas
            offset_x = 0
            offset_y = 0
        
        # Scale shapes and paths
        for shape in shapes:
            if hasattr(shape, 'points'):
                # Scale path points
                shape.points[:, 0] = shape.points[:, 0] * scale_x + offset_x
                shape.points[:, 1] = shape.points[:, 1] * scale_y + offset_y
            if hasattr(shape, 'stroke_width'):
                # Scale stroke width (optional)
                shape.stroke_width *= min(scale_x, scale_y)
        
        # Create rendering scene with target resolution
        scene = pydiffvg.RenderFunction.serialize_scene(
            width, height, shapes, shape_groups
        )
        
        # Initialize renderer
        render = pydiffvg.RenderFunction.apply
        
        # Render SVG to tensor
        img = render(
            width,           # render width
            height,          # render height
            2,               # num_samples_x
            2,               # num_samples_y
            0,               # seed
            None,            # background_image
            *scene
        )
        
        # Convert tensor to numpy array
        img = img[:, :, :3].cpu().numpy()  # Remove alpha channel, keep RGB
        img = (img * 255).astype(np.uint8)  # Scale to 0-255
        
        # Create background image
        background = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        
        # Blend image with background (handle transparency)
        alpha = img[:, :, 3:4] / 255.0 if img.shape[-1] == 4 else np.ones((height, width, 1))
        blended_img = (img[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Save as JPG
        pil_img = Image.fromarray(blended_img)
        pil_img.save(output_jpg_path, "JPEG", quality=95)
        
        print(f"Rendered SVG to JPG: {output_jpg_path}")
        return True
    
    except Exception as e:
        print(f"Error rendering SVG to JPG: {str(e)}")
        return False  

def compute_path_point_nums(shapes) :
    cnt = 0
    for path in  shapes :
        cnt += len(path.points)
    
    return cnt

def add_to_file(data_to_add, timing_file):
    data = {}
    # read all data that you have so far:
    if os.path.exists(timing_file):
        with open(timing_file, 'r') as f:
            data = json.load(f)
    # update dict:
    for k in data_to_add:
        data[k] = data_to_add[k]
    # write dict to file:
    with open(timing_file, 'w') as f:
        json.dump(data, f, indent=2)