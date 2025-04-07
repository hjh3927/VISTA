import os
import time
import imageio
import torch
import numpy as np
import pydiffvg
from PIL import Image
from utils import color_similarity, is_mask_included, mask_color_Kmeans, mask_to_path


def generate_init_svg(shapes, shape_groups, device, pre_mask_path_list, target_image, frames, out_svg_path, max_error=1.0, line_threshold=1.0, color_threshold=0.08, min_area=300):
    """
    根据预处理后的 mask 生成初始 SVG 每个 mask 对应一个路径，赋予颜色，
    并在初始化过程中生成渲染帧，存入 frames 列表中（便于生成动图）。
    """
    print("初始化 SVG...")
    st = time.time()
    height, width, _ = target_image.shape
    index_mask_dict = {}  # 索引到掩码的映射

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
        path.points = path.points.to(device)
        group_t = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([0]),  # 路径 ID
                        fill_color=torch.tensor([1.0,0,0,1.0]),  
                        # fill_color=torch.tensor([1.0,1.0,1.0,1.0]), 
                        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])  # 黑色描边
                    )
        shapes_t = [path]
        shape_groups_t = [group_t]
        
        rgb_color = mask_color_Kmeans(target_image, mask_image)
        color = torch.zeros(4, device=device)
        color[:3] = torch.tensor(rgb_color, device=device) / 255.0
        color[3] = 1.0
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i], device=device),
            fill_color=color,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        )
        shapes.append(path)
        shape_groups.append(group)

        index_mask_dict[i] = np.array(mask_image)  # 存储 mask 图像

        file_name = os.path.basename(mask_path).split('.')[0]
        pydiffvg.save_svg(os.path.join(out_svg_path, f'{file_name}.svg'), width, height, shapes, shape_groups)
        pydiffvg.save_svg(os.path.join(out_svg_path, f'single_{file_name}.svg'), width, height, shapes_t, shape_groups_t)
        # 渲染当前场景并保存为帧
        scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img_render = render(width, height, 2, 2, 0, None, *scene_args)
        img_render = img_render[:, :, :3]
        frame = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
        i += 1
    
    # 保存最终的初始化 SVG
    pydiffvg.save_svg(os.path.join(out_svg_path, 'init.svg'), width, height, shapes, shape_groups)
    print(f"SVG 初始化耗时--------------->: {time.time()-st:.2f} s")
    return shapes, shape_groups, frames, index_mask_dict



def svg_optimize(shapes, shape_groups, target_image, device, svg_out_path, frames, index_mask_dict, learning_rate=0.1, num_iters=1000,
                 early_stopping_patience=10, early_stopping_delta=5e-5, is_stroke=True, rm_color_threshold=0.1):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.6, patience=10)

    render = pydiffvg.RenderFunction.apply
    best_loss = float('inf')
    no_improve_count = 0

    # 优化循环
    for iter in range(num_iters):
        optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img_render = render(canvas_width, canvas_height, 2, 2, iter, None, *scene_args)
        img_render = img_render[:, :, :3].to(device)
        mse_loss = torch.mean((img_render - image_target) ** 2)

        loss = mse_loss
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
        # if iter % 10 == 0:
        current_lr = optim.param_groups[0]['lr']
        print(f"迭代 {iter}, Loss: {current_loss:.4f}, 当前学习率: {current_lr:.2f}")
        pydiffvg.save_svg(os.path.join(svg_out_path, f'iter_{iter}.svg'), canvas_width, canvas_height, shapes, shape_groups)
        frame = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)

        if no_improve_count >= early_stopping_patience:
            print(f"早停：连续 {early_stopping_patience} 次迭代损失无明显下降，提前停止优化。")
            pydiffvg.save_svg(os.path.join(svg_out_path, f'op_final.svg'), canvas_width, canvas_height, shapes, shape_groups)
            break

    to_remove = []
    print("移除多余 path ...")
    cnt = 0
    # 从后往前遍历 shapes 和 shape_groups，避免删除元素时修改索引
    lens = len(shape_groups)
    for i in range(lens - 1, lens//2, -1):  # 倒序遍历
        current_mask = index_mask_dict.get(i)
        if current_mask is not None:
            # 获取当前 mask 的颜色
            current_color = shape_groups[i].fill_color[:3].cpu()  # 获取该 shape 的颜色
            
            # 如果颜色相似且面积小于阈值，标记该shape和对应的mask为移除
            for j, other_group in enumerate(shape_groups):
                if i != j:
                    existing_color = other_group.fill_color[:3].cpu()  # 获取已有路径的颜色
                    if color_similarity(existing_color, current_color, device) < rm_color_threshold:
                        existing_mask = index_mask_dict.get(j)
                        if existing_mask is not None and is_mask_included(current_mask, existing_mask):
                            cnt += 1
                            print(f"移除第 {i} 个 shape，因为它的颜色与已有 shape 相似且被包含")
                            to_remove.append(i)
                            break

    print(f"共移除 {cnt} 个 path")

    # 执行移除操作
    for idx in sorted(to_remove, reverse=True):
        del shapes[idx]
        del shape_groups[idx]
        del index_mask_dict[idx]
    # 重新设置 shape_ids
    for i, shape in enumerate(shape_groups) :
        shape.shape_ids = torch.tensor([i])

    # 保存最终SVG和GIF
    svg_path = os.path.join(svg_out_path, 'final.svg')
    pydiffvg.save_svg(svg_path, canvas_width, canvas_height, shapes, shape_groups)
    frame = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
    frames.append(frame)
    gif_path = os.path.join(svg_out_path, 'animation.gif')
    imageio.mimsave(gif_path, frames, duration=15)

    print(f"SVG 优化耗时--------------->: {time.time()-st:.2f} s")
    return svg_path, gif_path
