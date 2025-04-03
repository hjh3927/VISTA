// 上传文件预览
document.getElementById('file').addEventListener('change', function (event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      document.getElementById('imgPreview').src = e.target.result;
    }
    reader.readAsDataURL(file);
  }
});

document.getElementById('uploadForm').addEventListener('submit', async function (event) {
  event.preventDefault();
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = '';

  // 重置 SVG 预览与下载按钮（iframe 的 src 属性）
  document.getElementById('svgPreview').src = '';
  const downloadContainer = document.getElementById('downloadContainer');
  downloadContainer.classList.add('hidden');
  // 重置状态提示
  const statusMessage = document.getElementById('statusMessage');
  statusMessage.classList.add('hidden');
  statusMessage.textContent = '';

  // 显示并初始化进度条
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  progressContainer.classList.remove('hidden');
  progressBar.style.width = "0%";
  progressText.textContent = "0%";

  const formData = new FormData(this);

  // 进度条模拟参数：20秒内递增到99%
  const duration = 40000; // 40秒
  const targetProgress = 99;
  const intervalTime = 100; // 每100ms更新一次
  const totalSteps = duration / intervalTime; // 总步数
  const increment = targetProgress / totalSteps; // 每步增加
  let currentProgress = 0;

  // 模拟进度条更新
  const progressInterval = setInterval(() => {
    currentProgress += increment;
    if (currentProgress >= targetProgress) {
      currentProgress = targetProgress;
      clearInterval(progressInterval);
    }
    progressBar.style.width = currentProgress.toFixed(0) + "%";
    progressText.textContent = currentProgress.toFixed(0) + "%";
  }, intervalTime);

  let fetchCompleted = false;

  try {
    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });
    fetchCompleted = true;

    // 请求结束时，清除进度定时器，并将进度条更新到100%
    clearInterval(progressInterval);
    progressBar.style.width = "100%";
    progressText.textContent = "100%";

    if (response.ok) {
      const data = await response.json();
      const svgUrl = data.svg_url;

      // 更新 SVG 预览（iframe 的 src 属性）
      document.getElementById('svgPreview').src = svgUrl;

      // 设置下载按钮：通过 fetch 获取 Blob 对象后创建下载链接
      const downloadBtn = document.getElementById('downloadBtn');
      fetch(svgUrl)
        .then(res => res.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          downloadBtn.href = url;
          downloadBtn.download = "output.svg";
        });

      // 显示状态提示
      statusMessage.textContent = 'SVG generated successfully!';
      statusMessage.classList.remove('hidden');

      downloadContainer.classList.remove('hidden');
      resultDiv.textContent = '';
    } else {
      resultDiv.textContent = 'Error processing image. Please try again.';
    }
  } catch (error) {
    resultDiv.textContent = 'An error occurred: ' + error.message;
  } finally {
    // 如果请求在40秒内完成，且进度条未达到99%，直接跳转到100%
    if (fetchCompleted && currentProgress < targetProgress) {
      clearInterval(progressInterval);
      progressBar.style.width = "100%";
      progressText.textContent = "100%";
    }
    // 隐藏进度条延时1秒
    setTimeout(() => {
      progressContainer.classList.add('hidden');
    }, 1000);
  }
});
