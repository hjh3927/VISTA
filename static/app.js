// Utility functions for the Vector Converter application
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

function isImageFile(file) {
    return file && file.type.startsWith('image/');
}

function generateUniqueId() {
    return 'id-' + Math.random().toString(36).substr(2, 9);
}

function safeJsonParse(str, fallback = null) {
    try {
        return JSON.parse(str);
    } catch (e) {
        console.error('JSON parse error:', e);
        return fallback;
    }
}

function createSvgElement(tag, attrs = {}) {
    const svgNS = "http://www.w3.org/2000/svg";
    const el = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) {
        el.setAttribute(key, value);
    }
    return el;
}

// Main application logic
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');
    const browseButton = document.getElementById('browse-button');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const fileNameElement = document.getElementById('file-name');
    const removeFileButton = document.getElementById('remove-file');
    const processButton = document.getElementById('process-button');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStatus = document.getElementById('progress-status');
    const initialState = document.getElementById('initial-state');
    const resultsContainer = document.getElementById('results-container');
    const svgPreview = document.getElementById('svg-preview');
    const svgPreviewContainer = document.getElementById('svg-preview-container');
    const downloadSvgButton = document.getElementById('download-svg');
    const toggleBackgroundButton = document.getElementById('toggle-background');
    const animationPreview = document.getElementById('animation-preview');
    const replayAnimationButton = document.getElementById('replay-animation');
    const downloadGifButton = document.getElementById('download-gif');
    const advancedToggle = document.getElementById('advanced-toggle');
    const advancedParams = document.getElementById('advanced-params');
    const advancedIcon = document.getElementById('advanced-icon');
    const resetParamsButton = document.getElementById('reset-params');
    const logOutput = document.getElementById('log-output');
    const cropNLayersCheckbox = document.getElementById('crop-n-layers');
    const isStrokeCheckbox = document.getElementById('is-stroke');

    // 初始化状态
    let uploadedFile = null;
    let originalFileName = '';  // 保存上传文件名（不含扩展名）
    let svgContent = '';        // 保存 SVG 内容
    let gifUrl = '';            // 保存 GIF URL
    let backgroundState = 'grid';

    // 默认值初始化
    isStrokeCheckbox.checked = true;
    cropNLayersCheckbox.checked = true;

    // 定义滑块参数及其默认值（与后端一致）
    const sliders = [
        { id: 'target-size', valueId: 'target-size-value', default: 512 },
        { id: 'pred-iou-thresh', valueId: 'pred-iou-thresh-value', default: 0.80 },
        { id: 'stability-score-thresh', valueId: 'stability-score-thresh-value', default: 0.90 },
        { id: 'min-area', valueId: 'min-area-value', default: 10 },
        { id: 'pre-color-threshold', valueId: 'pre-color-threshold-value', default: 0.01 },
        { id: 'line-threshold', valueId: 'line-threshold-value', default: 1.0 },
        { id: 'bzer-max-error', valueId: 'bzer-max-error-value', default: 1.0 },
        { id: 'learning-rate', valueId: 'learning-rate-value', default: 0.10 },
        { id: 'num-iters', valueId: 'num-iters-value', default: 1000 },
        { id: 'rm-color-threshold', valueId: 'rm-color-threshold-value', default: 0.10 }
    ];

    // 文件上传处理
    const handleFileUpload = (file) => {
        if (!isImageFile(file)) {
            alert('Please select an image file.');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            alert('File size exceeds 20MB limit.');
            return;
        }
        uploadedFile = file;
        originalFileName = file.name.split('.').slice(0, -1).join('.'); // 提取文件名，不含扩展名
        fileNameElement.textContent = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            processButton.disabled = false;
        };
        reader.readAsDataURL(file);
    };

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    browseButton.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('border-blue-600');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('border-blue-600');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-blue-600');
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });

    removeFileButton.addEventListener('click', () => {
        uploadedFile = null;
        originalFileName = '';
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        processButton.disabled = true;
    });

    // 参数实时更新显示值
    sliders.forEach(slider => {
        const inputElement = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        valueElement.textContent = inputElement.value; // 初始化显示
        inputElement.addEventListener('input', () => {
            valueElement.textContent = inputElement.value;
        });
    });

    // 重置参数到默认值
    resetParamsButton.addEventListener('click', () => {
        sliders.forEach(slider => {
            const inputElement = document.getElementById(slider.id);
            const valueElement = document.getElementById(slider.valueId);
            inputElement.value = slider.default;
            valueElement.textContent = slider.default;
        });
        isStrokeCheckbox.checked = true;
        cropNLayersCheckbox.checked = true;
    });

    // 高级参数切换
    advancedToggle.addEventListener('click', () => {
        advancedParams.classList.toggle('hidden');
        advancedIcon.classList.toggle('rotate-90');
    });

    // 切换背景
    toggleBackgroundButton.addEventListener('click', () => {
        svgPreviewContainer.classList.remove('bg-grid', 'bg-white', 'bg-black');
        if (backgroundState === 'grid') {
            svgPreviewContainer.classList.add('bg-white');
            backgroundState = 'white';
        } else if (backgroundState === 'white') {
            svgPreviewContainer.classList.add('bg-black');
            backgroundState = 'black';
        } else {
            svgPreviewContainer.classList.add('bg-grid');
            backgroundState = 'grid';
        }
        console.log('Background state:', backgroundState);
    });

    // 下载 SVG
    downloadSvgButton.addEventListener('click', () => {
        if (!svgContent) return;
        const blob = new Blob([svgContent], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${originalFileName}.svg`; // 使用上传文件名
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });

    // 重播动画
    replayAnimationButton.addEventListener('click', () => {
        if (!gifUrl) return;
        animationPreview.src = '';
        setTimeout(() => {
            animationPreview.src = gifUrl;
        }, 50);
    });

    // 下载 GIF
    downloadGifButton.addEventListener('click', async () => {
        if (!gifUrl) return;
        const response = await fetch(gifUrl);
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${originalFileName}.gif`; // 使用上传文件名
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });

    // 模拟进度条
    function simulateProgress() {
        let progress = 0;
        const totalTime = 35000;
        const targetProgress = 99;
        const intervalTime = 200;
        const increment = targetProgress / (totalTime / intervalTime);

        progressBar.style.width = `${progress}%`;
        progressPercentage.textContent = `${Math.round(progress)}%`;

        const interval = setInterval(() => {
            if (progress < targetProgress) {
                progress += increment;
                if (progress > targetProgress) progress = targetProgress;
                progressBar.style.width = `${progress}%`;
                progressPercentage.textContent = `${Math.round(progress)}%`;
            }
        }, intervalTime);

        return function completeProgress() {
            clearInterval(interval);
            progress = 100;
            progressBar.style.width = `${progress}%`;
            progressPercentage.textContent = `${Math.round(progress)}%`;
            progressStatus.textContent = 'Processing complete!';
        };
    }

    // WebSocket 日志
    const ws = new WebSocket(`ws://${window.location.host}/ws/logs`);
    ws.onopen = () => {
        console.log('WebSocket connected');
        logOutput.textContent = 'Connected to log stream.\n';
    };
    ws.onmessage = (event) => {
        const message = event.data;
        if (message !== "Heartbeat" && !message.includes("INFO -")) {
            console.log('Received output:', message);
            logOutput.textContent += `${message}\n`;
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    };
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        logOutput.textContent += 'Error connecting to log stream.\n';
    };
    ws.onclose = () => {
        console.log('WebSocket closed');
        logOutput.textContent += 'Log stream closed.\n';
    };

    // 处理图片
    processButton.addEventListener('click', async () => {
        console.log('Process button clicked');
        logOutput.textContent = '';
        if (!uploadedFile) return;

        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressPercentage.textContent = '0%';
        progressStatus.textContent = 'Processing...';
        processButton.disabled = true;

        try {
            const params = {
                target_size: parseInt(document.getElementById('target-size').value),
                pred_iou_thresh: parseFloat(document.getElementById('pred-iou-thresh').value),
                stability_score_thresh: parseFloat(document.getElementById('stability-score-thresh').value),
                crop_n_layers: cropNLayersCheckbox.checked ? 1 : 0,
                min_area: parseInt(document.getElementById('min-area').value),
                pre_color_threshold: parseFloat(document.getElementById('pre-color-threshold').value),
                line_threshold: parseFloat(document.getElementById('line-threshold').value),
                bzer_max_error: parseFloat(document.getElementById('bzer-max-error').value),
                learning_rate: parseFloat(document.getElementById('learning-rate').value),
                is_stroke: isStrokeCheckbox.checked,
                num_iters: parseInt(document.getElementById('num-iters').value),
                rm_color_threshold: parseFloat(document.getElementById('rm-color-threshold').value)
            };

            const formData = new FormData();
            formData.append('file', uploadedFile);
            for (const [key, value] of Object.entries(params)) {
                formData.append(key, value);
            }

            const completeProgress = simulateProgress();
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorDetail = await response.text();
                throw new Error(`Processing failed: ${errorDetail}`);
            }

            const result = await response.json();
            completeProgress();

            handleProcessResult({
                svg: await fetch(result.svg_url).then(res => res.text()),
                animationUrl: result.gif_url
            });
        } catch (error) {
            console.error('Processing failed:', error);
            progressStatus.textContent = 'Processing failed. Please try again.';
            progressBar.style.width = '0%';
            processButton.disabled = false;
        }
    });

    // 处理结果
    function handleProcessResult({ svg, animationUrl }) {
        initialState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        svgPreview.innerHTML = svg;
        animationPreview.src = animationUrl;

        svgContent = svg;  // 保存 SVG 内容
        gifUrl = animationUrl;  // 保存 GIF URL

        progressContainer.classList.add('hidden');
        processButton.disabled = false;
    }

    // 初始化 Lucide 图标
    lucide.createIcons();

    // 主题切换
    if (localStorage.getItem('theme') === 'dark' ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches && !localStorage.getItem('theme'))) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }

    document.getElementById('theme-toggle').addEventListener('click', () => {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        } else {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
    });

    function createLogWebSocket() {
        let ws;
        let heartbeatInterval;
        const reconnectDelay = 5000; // 重连间隔 5 秒

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws/logs`);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                logOutput.textContent = 'Connected to log stream.\n';

                // 启动心跳：客户端每 10 秒发一次 ping
                heartbeatInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('ping');
                    }
                }, 10000);
            };

            ws.onmessage = (event) => {
                const message = event.data;
                // 过滤掉心跳和不重要的日志
                if (message !== "Heartbeat" && !message.includes("INFO -")) {
                    console.log('🟢 Log:', message);
                    logOutput.textContent += `${message}\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                }
            };

            ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                logOutput.textContent += 'Error connecting to log stream.\n';
                ws.close(); // 出错后触发 onclose
            };

            ws.onclose = () => {
                console.warn('⚠️ WebSocket closed. Reconnecting in 5s...');
                logOutput.textContent += 'Log stream closed. Reconnecting...\n';
                clearInterval(heartbeatInterval);
                setTimeout(connect, reconnectDelay); // 自动重连
            };
        }

        connect(); // 初始连接
    }
    createLogWebSocket(); // 创建 WebSocket 连接    

});