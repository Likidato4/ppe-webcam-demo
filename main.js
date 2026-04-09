const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictionList = document.getElementById("predictionList");
const statusText = document.getElementById("statusText");
const statusPill = document.getElementById("statusPill");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const detectionCountEl = document.getElementById("detectionCount");
const topConfidenceEl = document.getElementById("topConfidence");

let model = null;
let stream = null;
let animationId = null;
let isRunning = false;
let isLoadingModel = false;

const CONFIDENCE_THRESHOLD = 0.35;
const MAX_PREDICTIONS_TO_SHOW = 6;

const MODEL_CONFIG = {
  publishable_key: "rf_y6gXGN9vlnZWxkTtzuThRNaBaQe2",
  model: "projectdrone2-y6jvn",
  version: 19
};

function setStatus(message, mode = "default") {
  statusText.textContent = message;
  statusPill.classList.remove("status-live", "status-warning");

  if (mode === "live") {
    statusPill.classList.add("status-live");
  } else if (mode === "warning") {
    statusPill.classList.add("status-warning");
  }
}

function setButtons({ startDisabled, stopDisabled }) {
  startBtn.disabled = startDisabled;
  stopBtn.disabled = stopDisabled;
}

function updateSummary(predictions = []) {
  detectionCountEl.textContent = predictions.length;

  if (!predictions.length) {
    topConfidenceEl.textContent = "0%";
    return;
  }

  const highest = Math.max(...predictions.map((p) => p.confidence || 0));
  topConfidenceEl.textContent = `${(highest * 100).toFixed(1)}%`;
}

function showEmptyState(message) {
  predictionList.innerHTML = `
    <div class="empty-state">
      ${message}
    </div>
  `;
  updateSummary([]);
}

async function loadModel() {
  if (model) return model;

  if (isLoadingModel) {
    while (isLoadingModel) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    return model;
  }

  try {
    isLoadingModel = true;
    setStatus("Loading PPE model...", "warning");

    model = await roboflow
      .auth({
        publishable_key: MODEL_CONFIG.publishable_key
      })
      .load({
        model: MODEL_CONFIG.model,
        version: MODEL_CONFIG.version
      });

    setStatus("Model loaded. Ready for webcam.", "default");
    return model;
  } catch (error) {
    console.error("Model loading error:", error);
    setStatus("Failed to load model.", "warning");
    showEmptyState(
      "The PPE model could not be loaded. Check your Roboflow API key, model ID, and version."
    );
    throw error;
  } finally {
    isLoadingModel = false;
  }
}

function resizeCanvasToVideo() {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}

function normalizePredictions(predictions) {
  if (!Array.isArray(predictions)) return [];

  return predictions
    .filter((pred) => typeof pred.confidence === "number")
    .filter((pred) => pred.confidence >= CONFIDENCE_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence);
}

function renderPredictions(predictions) {
  if (!predictions.length) {
    showEmptyState("Model is live, but no PPE objects are currently detected.");
    return;
  }

  updateSummary(predictions);

  const displayPredictions = predictions.slice(0, MAX_PREDICTIONS_TO_SHOW);

  predictionList.innerHTML = displayPredictions
    .map((pred) => {
      const confidence = (pred.confidence * 100).toFixed(1);
      return `
        <div class="prediction-item">
          <div class="prediction-row">
            <div class="prediction-label">${pred.class}</div>
            <div class="prediction-score">${confidence}%</div>
          </div>
          <div class="progress">
            <div class="progress-bar" style="width: ${confidence}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function drawPredictions(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!predictions.length) return;

  predictions.forEach((pred) => {
    if (
      typeof pred.x !== "number" ||
      typeof pred.y !== "number" ||
      typeof pred.width !== "number" ||
      typeof pred.height !== "number"
    ) {
      return;
    }

    const x = canvas.width - (pred.x + pred.width / 2);
    const y = pred.y - pred.height / 2;
    const width = pred.width;
    const height = pred.height;
    const confidence = (pred.confidence * 100).toFixed(1);
    const label = `${pred.class} ${confidence}%`;

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    ctx.font = "600 14px Inter, Arial, sans-serif";
    const textPaddingX = 8;
    const textHeight = 28;
    const textWidth = ctx.measureText(label).width;

    const labelX = Math.max(0, x);
    const labelY = Math.max(0, y - textHeight - 6);

    ctx.fillStyle = "#22c55e";
    ctx.fillRect(
      labelX,
      labelY,
      textWidth + textPaddingX * 2,
      textHeight
    );

    ctx.fillStyle = "#08111f";
    ctx.fillText(label, labelX + textPaddingX, labelY + 18);
  });
}

async function detectFrame() {
  if (!isRunning || !model || video.readyState < 2) {
    return;
  }

  try {
    const rawPredictions = await model.detect(video);
    const predictions = normalizePredictions(rawPredictions);

    drawPredictions(predictions);
    renderPredictions(predictions);
    setStatus("Live PPE inference active", "live");
  } catch (error) {
    console.error("Inference error:", error);
    setStatus("Inference error. Check model settings.", "warning");
    showEmptyState(
      "The model encountered an inference error. Verify your deployment settings and try again."
    );
    stopCamera(false);
    return;
  }

  animationId = requestAnimationFrame(detectFrame);
}

async function startCamera() {
  if (isRunning) return;

  try {
    setButtons({ startDisabled: true, stopDisabled: false });
    setStatus("Preparing webcam...", "warning");

    await loadModel();

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    resizeCanvasToVideo();
    videoPlaceholder.classList.add("hidden");

    isRunning = true;
    setStatus("Webcam connected. Starting inference...", "live");
    showEmptyState("Scanning for PPE detections...");

    detectFrame();
  } catch (error) {
    console.error("Camera start error:", error);
    setStatus("Camera access denied or unavailable.", "warning");
    showEmptyState(
      "Unable to access the webcam. Run this project on localhost or HTTPS, then allow camera permissions."
    );
    setButtons({ startDisabled: false, stopDisabled: true });
  }
}

function stopCamera(updateStatus = true) {
  isRunning = false;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  video.pause();
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  videoPlaceholder.classList.remove("hidden");

  if (updateStatus) {
    setStatus("Camera stopped", "default");
    showEmptyState(
      'Camera stopped. Click <strong>Enable Webcam</strong> to restart the live demo.'
    );
  } else {
    updateSummary([]);
  }

  setButtons({ startDisabled: false, stopDisabled: true });
}

window.addEventListener("resize", () => {
  if (video.srcObject) {
    resizeCanvasToVideo();
  }
});

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", () => stopCamera(true));

setButtons({ startDisabled: false, stopDisabled: true });
showEmptyState("No predictions yet. Start the webcam to begin PPE inference.");
