import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
from app_main import img_to_svg 
import os
import shutil
import tempfile
import time

from config import TEMP_OUTPUTS_DIR

app = FastAPI()

# 配置自定义日志记录器
logger = logging.getLogger("VectorConverter")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 关闭日志传播
logger.propagate = False  # 阻止日志传播到根日志记录器

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../static")), name="static")
if not os.path.exists(TEMP_OUTPUTS_DIR):
    os.makedirs(TEMP_OUTPUTS_DIR)
app.mount("/temp_outputs", StaticFiles(directory=TEMP_OUTPUTS_DIR), name="temp_outputs")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "../static/index.html"), "r", encoding="utf-8") as f:
        return f.read()

async def run_algorithm(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold):
    try:
        svg_path, gif_path, output,  all_time, shapes_count, mes_loss, path_point_nums = img_to_svg(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold)
        svg_url = svg_path.replace(TEMP_OUTPUTS_DIR, "/temp_outputs")
        gif_url = gif_path.replace(TEMP_OUTPUTS_DIR, "/temp_outputs")
        
        # 整理需要返回的信息
        log_info = {
            "output_directory": output,  # 输出目录
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
            "time_consuming": f"{all_time:.2f} s",  # 处理耗时
            "path_point_nums": path_point_nums,  # 路径点数量
            "shapes": shapes_count,  # 形状数量
            "mes_loss": f"{mes_loss:.4f}"  # 损失值
        }
        return {"svg_url": svg_url, "gif_url": gif_url, "log_info": log_info} 
    except Exception as e:
        logger.error(f"Algorithm failed: {str(e)}")
        raise

@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_size: int = Form(512),
    pred_iou_thresh: float = Form(0.80),
    stability_score_thresh: float = Form(0.90),
    crop_n_layers: int = Form(1),
    min_area: int = Form(10),
    pre_color_threshold: float = Form(0.01),
    line_threshold: float = Form(1.0),
    bzer_max_error: float = Form(1.0),
    learning_rate: float = Form(0.1),
    is_stroke: bool = Form(True),
    num_iters: int = Form(1000),
    rm_color_threshold: float = Form(0.10)
):
    logger.info(f"Received file: {file.filename}")
    logger.info("Starting image processing...")

    # 获取上传文件名（不含扩展名）并添加唯一标识
    original_filename = os.path.splitext(file.filename)[0]  
    unique_id = str(uuid.uuid4())[:8]  # 短唯一标识，避免冲突
    base_filename = f"{original_filename}_{unique_id}"  

    # 创建临时目录
    temp_dir = tempfile.gettempdir()
    temp_img_path = os.path.join(temp_dir, f"{base_filename}.jpg")

    # 保存上传图片到临时文件
    try:
        with open(temp_img_path, "wb") as temp_img:
            shutil.copyfileobj(file.file, temp_img)
    except Exception as e:
        logger.error(f"Failed to save temporary file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 调用算法，直接输出到控制台
    try:
        result = await run_algorithm(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Processing failed: {str(e)}"})

    # 清理任务
    file_name = os.path.basename(temp_img_path).split('.')[0]
    rm_path = os.path.join("temp_outputs", file_name)
    background_tasks.add_task(delete_file_after_delay, rm_path, delay=600)
    background_tasks.add_task(delete_file_after_delay, temp_img_path, delay=600)
    
    logger.info("Image processing completed.")
    return JSONResponse(content=result)

def delete_file_after_delay(file_path, delay=300):
    time.sleep(delay)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    elif os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)