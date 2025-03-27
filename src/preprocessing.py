import os
import numpy as np
from PIL import Image


def load_and_resize(image_path: str, target_size: int = 512):
    """
    加载并缩放图像，保持宽高比，返回 numpy 数组
    """
    print("预处理目标图像...")
    image = Image.open(image_path).convert("RGB") 
    w, h = image.size
    
    scale = target_size / max(w, h)
    new_size = (int(w*scale), int(h*scale))    
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return np.array(resized)
  
def save_target_image(image_array, out_dir, file_name):
    """
    将 numpy 图像保存为 PIL 格式图片
    """
    img_pil = Image.fromarray(image_array)
    out_file = os.path.join(out_dir, file_name)
    img_pil.save(out_file)
    return out_file
