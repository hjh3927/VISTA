import os
import cv2
import time
import torch
import numpy as np
import pydiffvg
from PIL import Image
from sklearn.cluster import KMeans
from utils import fit_bezier_segment, compute_error, point_to_line_distance

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
    simplified = cv2.approxPolyDP(contour, 2.0, closed=True).squeeze()
    idx1 = np.where((contour == simplified[0]).all(axis=1))[0][0]
    if (contour[idx1-1] != simplified[-1]).any():
        simplified = np.vstack((simplified, contour[idx1-1]))
    structured_points = fit_contour(simplified, contour, max_error, line_threshold)
    path = points_to_path(structured_points, closed=True)
    return path

def mask_color_Kmeans(image, mask, n_clusters=1):
    """
    使用 K-means 聚类从图像对应的 mask 区域中提取主色
    """
    mask_np = np.array(mask)
    masked_pixels = image[mask_np > 0]
    if len(masked_pixels) == 0:
        return (0, 0, 0)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(masked_pixels)
    main_color = kmeans.cluster_centers_[0].astype(int)
    return tuple(main_color)

def generate_init_svg(shapes, shape_groups, device, pre_mask_path_list, target_image, out_svg_path, max_error=1.0, line_threshold=1.0):
    """
    根据预处理后的 mask 生成初始 SVG，每个 mask 对应一个路径，赋予颜色
    """
    print("初始化 SVG...")
    st = time.time()
    height, width, _ = target_image.shape
    i = 1
    for mask_path in pre_mask_path_list:
        mask_image = Image.open(mask_path).convert('L')
        path = mask_to_path(mask_image, max_error, line_threshold)
        if path is None:
            continue
        path.points = path.points.to(device)
        rgb_color = mask_color_Kmeans(target_image, mask_image)
        color = torch.zeros(4, device=device)
        color[:3] = torch.tensor(rgb_color, device=device) / 255.0
        color[3] = 1.0
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i], device=device),
            fill_color=color,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        )
        shapes.append(path)
        shape_groups.append(group)
        pydiffvg.save_svg(os.path.join(out_svg_path, f'{i}.svg'), width, height, shapes, shape_groups)
        i += 1
    et = time.time()
    pydiffvg.save_svg(os.path.join(out_svg_path, f'final.svg'), width, height, shapes, shape_groups)
    print(f"SVG 初始化耗时: {et-st:.2f} s")
    return shapes, shape_groups


def svg_optimize(shapes, shape_groups, target_image, device, svg_out_path, learning_rate=0.1, num_iters=100, lamda1=0.1, lamda2=0.1,
                 early_stopping_patience=10, early_stopping_delta=1e-5):
    """
    优化 SVG，通过对路径点和颜色参数的反向传播更新，最小化与目标图像的误差。
    新增早停策略：在连续 early_stopping_patience 次迭代中损失没有下降 early_stopping_delta 时提前停止。
    """
    print("开始 SVG 优化...")
    image_target = torch.from_numpy(target_image).float() / 255.0
    image_target = image_target.to(device)
    canvas_height, canvas_width = target_image.shape[0], target_image.shape[1]
    pydiffvg.save_svg(os.path.join(svg_out_path, f'init.svg'), canvas_width, canvas_height, shapes, shape_groups)
    
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    stroke_color_var = []
    for path in shapes:
        path.points = path.points.to(device)
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width = path.stroke_width.to(device)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.fill_color = group.fill_color.to(device)
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
        group.stroke_color = group.stroke_color.to(device)
        group.stroke_color.requires_grad = True
        stroke_color_var.append(group.stroke_color)
    
    optim = torch.optim.Adam(points_vars + stroke_width_vars + color_vars + stroke_color_var, lr=learning_rate)
    # optim = torch.optim.Adam(points_vars + color_vars, lr=learning_rate)
    render = pydiffvg.RenderFunction.apply

    best_loss = float('inf')
    no_improve_count = 0

    for iter in range(num_iters):
        optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img_render = render(canvas_width, canvas_height, 2, 2, iter, None, *scene_args)
        img_render = img_render[:, :, :3].to(device)
        mse_loss = torch.mean((img_render - image_target) ** 2)
        num_paths = len(shapes)
        path_penalty = lamda1 * num_paths
        loss = mse_loss + path_penalty

        loss.backward()
        optim.step()

        # 早停逻辑
        current_loss = loss.item()
        if current_loss + early_stopping_delta < best_loss:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if iter % 10 == 0:
            print(f"迭代 {iter}, Loss: {current_loss:.4f}")
            pydiffvg.save_svg(os.path.join(svg_out_path, f'iter_{iter}.svg'), canvas_width, canvas_height, shapes, shape_groups)
        
        if no_improve_count >= early_stopping_patience:
            print(f"早停：连续 {early_stopping_patience} 次迭代损失无明显下降，提前停止优化。")
            break

    pydiffvg.save_svg(os.path.join(svg_out_path, 'final.svg'), canvas_width, canvas_height, shapes, shape_groups)
