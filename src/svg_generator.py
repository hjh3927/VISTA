import os
import cv2
import time
import imageio
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


def generate_init_svg(shapes, shape_groups, device, pre_mask_path_list, target_image, frames, out_svg_path, max_error=1.0, line_threshold=1.0):
    """
    根据预处理后的 mask 生成初始 SVG，每个 mask 对应一个路径，赋予颜色，
    并在初始化过程中生成渲染帧，存入 frames 列表中（便于生成动图）。
    """
    print("初始化 SVG...")
    st = time.time()
    height, width, _ = target_image.shape

    # 加入白色背景
    bg_points = torch.tensor([
        [0.0, 0.0],           # 左下角
        [width, 0.0],         # 右下角
        [width, height],      # 右上角
        [0.0, height]         # 左上角
    ])
    bg_path = pydiffvg.Path(
        num_control_points=torch.LongTensor([0, 0, 0, 0]),
        points=bg_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True,
    )
    bg_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0])
    )
    shapes.append(bg_path)
    shape_groups.append(bg_group)

    i = 1
    for j, mask_path in enumerate(pre_mask_path_list):
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
        
        # 每隔一定数量的 mask 保存一次 SVG 和渲染帧
        # if j % 3 == 0:
            # 保存当前 SVG 文件
        pydiffvg.save_svg(os.path.join(out_svg_path, f'{i}.svg'), width, height, shapes, shape_groups)
            
            # 渲染当前场景并保存为帧
        scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        # 传入 0 作为迭代编号，渲染当前场景
        img_render = render(width, height, 2, 2, 0, None, *scene_args)
        # 仅保留 RGB 通道，假设输出范围为 [0,1]
        img_render = img_render[:, :, :3]
        frame = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
        i += 1

    # 保存最终的初始化 SVG
    pydiffvg.save_svg(os.path.join(out_svg_path, 'init.svg'), width, height, shapes, shape_groups)
    print(f"SVG 初始化耗时--------------->: {time.time()-st:.2f} s")
    return shapes, shape_groups, frames


def svg_optimize(shapes, shape_groups, target_image, device, svg_out_path, frames, learning_rate=0.1, num_iters=1000,
                 early_stopping_patience=10, early_stopping_delta=1e-5, is_stroke=True):
    """
    优化 SVG，通过对路径点、颜色、描边宽度和描边颜色参数的反向传播更新，
    最小化与目标图像的误差。支持早停策略和动态调整学习率。
    
    参数：
        shapes: SVG路径列表
        shape_groups: SVG组列表
        target_image: 目标图像
        device: 计算设备（CPU/GPU）
        svg_out_path: SVG输出路径
        frames: 用于生成GIF的帧列表
        learning_rate: 初始学习率
        num_iters: 最大迭代次数
        early_stopping_patience: 早停耐心值
        early_stopping_delta: 早停损失阈值
        is_stroke: 是否优化描边（True：优化描边宽度和颜色，False：仅优化路径点和填充颜色）
    
    返回：
        tuple: (svg_path, gif_path) - 优化后的SVG文件路径和GIF动画路径
    """
    st = time.time()
    print("开始 SVG 优化...")
    
    # 准备目标图像
    image_target = torch.from_numpy(target_image).float() / 255.0
    image_target = image_target.to(device)
    canvas_height, canvas_width = target_image.shape[0], target_image.shape[1]
    pydiffvg.save_svg(os.path.join(svg_out_path, f'init.svg'), canvas_width, canvas_height, shapes, shape_groups)
    
    # 初始化优化变量
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    stroke_color_var = []
    
    # 设置路径点和描边宽度
    for path in shapes:
        path.points = path.points.to(device)
        path.points.requires_grad = True
        points_vars.append(path.points)
        if is_stroke:
            path.stroke_width = path.stroke_width.to(device)
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    
    # 设置填充颜色和描边颜色
    for group in shape_groups:
        group.fill_color = group.fill_color.to(device)
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
        if is_stroke:
            group.stroke_color = group.stroke_color.to(device)
            group.stroke_color.requires_grad = True
            stroke_color_var.append(group.stroke_color)
    
    # 创建优化器，根据is_stroke选择优化参数
    optim_params = points_vars + color_vars
    if is_stroke:
        optim_params += stroke_width_vars + stroke_color_var
    optim = torch.optim.Adam(optim_params, lr=learning_rate)
    
    # 使用ReduceLROnPlateau调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
    
    render = pydiffvg.RenderFunction.apply
    best_loss = float('inf')
    no_improve_count = 0

    # 优化循环
    for iter in range(num_iters):
        optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img_render = render(canvas_width, canvas_height, 2, 2, iter, None, *scene_args)
        img_render = img_render[:, :, :3].to(device)
        loss = torch.mean((img_render - image_target) ** 2)

        loss.backward()
        optim.step()
        scheduler.step(loss)

        # 早停逻辑
        current_loss = loss.item()
        if current_loss + early_stopping_delta < best_loss:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 每10次迭代保存中间结果
        if iter % 10 == 0:
            current_lr = optim.param_groups[0]['lr']
            print(f"迭代 {iter}, Loss: {current_loss:.4f}, 当前学习率: {current_lr:.6f}")
            pydiffvg.save_svg(os.path.join(svg_out_path, f'iter_{iter}.svg'), canvas_width, canvas_height, shapes, shape_groups)
            frame = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        
        if no_improve_count >= early_stopping_patience:
            print(f"早停：连续 {early_stopping_patience} 次迭代损失无明显下降，提前停止优化。")
            break

    # 保存最终SVG和GIF
    svg_path = os.path.join(svg_out_path, 'final.svg')
    pydiffvg.save_svg(svg_path, canvas_width, canvas_height, shapes, shape_groups)
    gif_path = os.path.join(svg_out_path, 'animation.gif')
    imageio.mimsave(gif_path, frames, duration=15)

    print(f"SVG 优化耗时--------------->: {time.time()-st:.2f} s")
    return svg_path, gif_path