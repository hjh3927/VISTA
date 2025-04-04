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
    const fileName = document.getElementById('file-name');
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
    const advancedToggle = document.getElementById('advanced-toggle');
    const advancedParams = document.getElementById('advanced-params');
    const advancedIcon = document.getElementById('advanced-icon');
    const resetParamsButton = document.getElementById('reset-params');

    let backgroundState = 'grid'; // 初始为网格背景
    const isStrokeCheckbox = document.getElementById('is-stroke');
    isStrokeCheckbox.checked = true; // 确保页面加载时默认开启

    // Parameter sliders and values
    const sliders = [
        { id: 'target-size', valueId: 'target-size-value', default: 512 },
        { id: 'pred-iou-thresh', valueId: 'pred-iou-thresh-value', default: 0.80 },
        { id: 'stability-score-thresh', valueId: 'stability-score-thresh-value', default: 0.90 },
        { id: 'min-area', valueId: 'min-area-value', default: 10 },
        { id: 'line-threshold', valueId: 'line-threshold-value', default: 1.0 },
        { id: 'bzer-max-error', valueId: 'bzer-max-error-value', default: 1.0 },
        { id: 'learning-rate', valueId: 'learning-rate-value', default: 0.1 },
        { id: 'num-iters', valueId: 'num-iters-value', default: 1000 }
    ];

    // Track uploaded file
    let uploadedFile = null;
    let svgData = null;
    let animationUrl = null;

    // Initialize slider values
    sliders.forEach(slider => {
        const inputElement = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        valueElement.textContent = inputElement.value;
        inputElement.addEventListener('input', () => {
            valueElement.textContent = inputElement.value;
        });
    });

    // Reset parameters to defaults
    resetParamsButton.addEventListener('click', () => {
        sliders.forEach(slider => {
            const inputElement = document.getElementById(slider.id);
            const valueElement = document.getElementById(slider.valueId);
            inputElement.value = slider.default;
            valueElement.textContent = slider.default;
        });
        isStrokeCheckbox.checked = true; // 重置时保持开启
    });

    // Toggle advanced parameters
    advancedToggle.addEventListener('click', () => {
        advancedParams.classList.toggle('hidden');
        advancedIcon.classList.toggle('rotate-90');
    });

    // File Upload Handling
    fileInput.addEventListener('change', handleFileSelect);
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
        if (e.dataTransfer.files.length) handleFileFromEvent(e.dataTransfer.files[0]);
    });
    removeFileButton.addEventListener('click', () => {
        uploadedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        processButton.disabled = true;
    });

    toggleBackgroundButton.addEventListener('click', () => {
        svgPreviewContainer.classList.remove('bg-grid', 'bg-white', 'bg-black');
        if (backgroundState === 'grid') {
            svgPreviewContainer.classList.add('bg-white');
            backgroundState = 'white';
        } else if (backgroundState === 'white') {
            svgPreviewContainer.classList.add('bg-black');
            backgroundState = 'transparent';
        } else {
            svgPreviewContainer.classList.add('bg-grid');
            backgroundState = 'grid';
        }
        console.log('Background state:', backgroundState);
    });

    // Download SVG button
    downloadSvgButton.addEventListener('click', () => {
        if (!svgData) return;
        const blob = new Blob([svgData], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'vectorized_image.svg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Replay animation button
    replayAnimationButton.addEventListener('click', () => {
        if (!animationUrl) return;
        animationPreview.src = '';
        setTimeout(() => {
            animationPreview.src = animationUrl;
        }, 50);
    });

    // Functions
    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFileFromEvent(e.target.files[0]);
        }
    }

    function handleFileFromEvent(file) {
        if (!isImageFile(file)) { // 使用 utils.js 的函数
            alert('Please select an image file.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10MB limit.');
            return;
        }
        uploadedFile = file;
        fileName.textContent = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            processButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function simulateProgress() {
        let progress = 0;
        const totalTime = 35000; // 35秒
        const targetProgress = 99;
        const intervalTime = 200;
        const increment = (targetProgress / (totalTime / intervalTime));

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

    processButton.addEventListener('click', async () => {
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
                min_area: parseFloat(document.getElementById('min-area').value),
                line_threshold: parseFloat(document.getElementById('line-threshold').value),
                bzer_max_error: parseFloat(document.getElementById('bzer-max-error').value),
                learning_rate: parseFloat(document.getElementById('learning-rate').value),
                is_stroke: isStrokeCheckbox.checked,
                num_iters: parseInt(document.getElementById('num-iters').value)
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

    function handleProcessResult(result) {
        svgData = result.svg;
        animationUrl = result.animationUrl + '?t=' + new Date().getTime();
        svgPreview.innerHTML = svgData;
        animationPreview.src = '';
        setTimeout(() => {
            animationPreview.src = animationUrl;
        }, 50);
        initialState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        processButton.disabled = false;
    }
});

// Initialize Lucide icons and theme toggle 
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    // Check for saved theme
    if (localStorage.getItem('theme') === 'dark' ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches && !localStorage.getItem('theme'))) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }

    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', () => {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        } else {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
    });
});