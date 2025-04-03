from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from img_to_svg import img_to_svg
import os
import shutil
import tempfile
import time

app = FastAPI()

# 挂载静态文件目录和临时生成文件目录（用于存放生成的 SVG）
app.mount("/static", StaticFiles(directory="static"), name="static")
if not os.path.exists("temp_outputs"):
    os.makedirs("temp_outputs")
app.mount("/temp_outputs", StaticFiles(directory="temp_outputs"), name="temp_outputs")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_area: int = Form(10),
    max_error: float = Form(1.0),
    line_threshold: float = Form(1.0),
    learning_rate: float = Form(0.1),
    num_iters: int = Form(1000)
):
    # 保存上传图片到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        shutil.copyfileobj(file.file, temp_img)
        temp_img_path = temp_img.name

    # 调用图像处理和SVG生成函数（假设返回生成的 SVG 文件路径）
    svg_path = generate_svg(temp_img_path, min_area, max_error, line_threshold, learning_rate, num_iters)
    
    # 将生成的 SVG 移动到 temp_outputs 目录下，便于预览和下载
    svg_filename = os.path.basename(svg_path)
    new_svg_path = os.path.join("temp_outputs", svg_filename)
    shutil.move(svg_path, new_svg_path)

    # 立即删除上传的临时图片
    os.remove(temp_img_path)

    # 后台任务：延迟 300 秒后删除生成的 SVG 文件（实际环境可用更稳健的定时任务）
    background_tasks.add_task(delete_file_after_delay, new_svg_path, delay=300)

    # 返回 SVG 文件的 URL（前端可通过该 URL展示预览和下载）
    return JSONResponse({"svg_url": f"/temp_outputs/{svg_filename}"})

def generate_svg(image_path, min_area, max_error, line_threshold, learning_rate, num_iters):
    # 这里调用主要算法，返回生成 SVG 文件的临时路径
    svg_path = img_to_svg(image_path, min_area, max_error, line_threshold, learning_rate, num_iters)
    return svg_path

def delete_file_after_delay(file_path, delay=300):
    """后台任务：延迟删除文件"""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
