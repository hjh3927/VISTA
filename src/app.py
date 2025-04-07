import asyncio
from contextlib import redirect_stdout
from io import StringIO
import logging
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from img_to_svg import img_to_svg 
import os
import shutil
import tempfile
import time

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorConverter")
logger.setLevel(logging.INFO)

connected_clients = set()

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../static")), name="static")
if not os.path.exists("temp_outputs"):
    os.makedirs("temp_outputs")
app.mount("/temp_outputs", StaticFiles(directory="temp_outputs"), name="temp_outputs")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "../static/index.html"), "r", encoding="utf-8") as f:
        return f.read()

# 自定义输出流
class WebSocketOutput(StringIO):
    def __init__(self):
        super().__init__()
        self.buffer = []

    async def write_async(self, message):
        self.buffer.append(message)
        for client in list(connected_clients):
            try:
                await client.send_text(message.strip())
            except Exception as e:
                logger.error(f"Failed to send output: {e}")
                connected_clients.remove(client)

    def write(self, message):
        asyncio.run_coroutine_threadsafe(self.write_async(message), asyncio.get_event_loop())

    def flush(self):
        pass

async def run_algorithm(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold):
    try:
        svg_path, gif_path = img_to_svg(temp_img_path, target_size, pred_iou_thresh, stability_score_thresh, crop_n_layers, min_area, pre_color_threshold, line_threshold, bzer_max_error, learning_rate, is_stroke, num_iters, rm_color_threshold)
        svg_url = svg_path.replace(os.path.abspath("temp_outputs"), "/temp_outputs")
        gif_url = gif_path.replace(os.path.abspath("temp_outputs"), "/temp_outputs")
        return {"svg_url": svg_url, "gif_url": gif_url}
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
    min_area: int = Form(50),
    pre_color_threshold: float = Form(0.0),
    line_threshold: float = Form(1.0),
    bzer_max_error: float = Form(1.0),
    learning_rate: float = Form(0.1),
    is_stroke: bool = Form(True),
    num_iters: int = Form(1000),
    rm_color_threshold: float = Form(0.0)
):
    output = WebSocketOutput()
    logger.info(f"Received file: {file.filename}")
    logger.info("Starting image processing...")

    # 保存上传图片到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        shutil.copyfileobj(file.file, temp_img)
        temp_img_path = temp_img.name

    # 重定向 print 输出并调用算法
    try:
        with redirect_stdout(output):
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

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("WebSocket client connected")
    try:
        while True:
            await websocket.send_text("Heartbeat")
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info("WebSocket client disconnected")

class WebSocketHandler(logging.Handler):
    async def emit_async(self, record):
        log_entry = self.format(record)
        for client in list(connected_clients):
            try:
                await client.send_text(log_entry)
            except Exception as e:
                logger.error(f"Failed to send log: {e}")
                connected_clients.remove(client)

    def emit(self, record):
        asyncio.run_coroutine_threadsafe(self.emit_async(record), asyncio.get_event_loop())

# 配置日志处理器
ws_handler = WebSocketHandler()
ws_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ws_handler.setFormatter(formatter)
logger.addHandler(ws_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def delete_file_after_delay(file_path, delay=300):
    time.sleep(delay)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    elif os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)