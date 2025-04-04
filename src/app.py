from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from img_to_svg import img_to_svg
import os
import shutil
import tempfile
import time

app = FastAPI()

# 挂载静态文件目录（从项目根目录引用static）
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../static")), name="static")
# 挂载临时输出目录
if not os.path.exists("temp_outputs"):
    os.makedirs("temp_outputs")
app.mount("/temp_outputs", StaticFiles(directory="temp_outputs"), name="temp_outputs")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "../static/index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_size: int = Form(512),
    pred_iou_thresh: float = Form(0.80),
    stability_score_thresh: float = Form(0.90),
    min_area: int = Form(10),
    line_threshold: float = Form(1.0),
    bzer_max_error: float = Form(1.0),
    learning_rate: float = Form(0.1),
    is_stroke: bool = Form(True),
    num_iters: int = Form(1000)
):

    # 保存上传图片到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        shutil.copyfileobj(file.file, temp_img)
        temp_img_path = temp_img.name

    # 调用图像处理和SVG生成函数（假设在svg_generator.py中）
    svg_path, gif_path = img_to_svg(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, min_area, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters)
    
    # 将路径转换为前端可访问的URL
    svg_url = svg_path.replace(os.path.abspath("temp_outputs"), "/temp_outputs")
    gif_url = gif_path.replace(os.path.abspath("temp_outputs"), "/temp_outputs")
    
    
    # 清理任务
    file_name = os.path.basename(temp_img_path).split('.')[0]
    rm_path = os.path.join("temp_outputs", file_name)
    background_tasks.add_task(delete_file_after_delay, rm_path, delay=600)
    background_tasks.add_task(delete_file_after_delay, temp_img_path, delay=600)

    return JSONResponse({"svg_url": svg_url, "gif_url": gif_url})

def delete_file_after_delay(file_path, delay=300):
    """后台任务：延迟删除整个目录"""
    time.sleep(delay)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    elif os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)