// ================================
// Sign Language Detector Web App
// ================================

const API_BASE = "";
let recordStream = null;
let classifyStream = null;
let sentenceStream = null;
let classifyWs = null;
let sentenceWs = null;
let isCapturing = false;
let captureCount = 0;
let captureTarget = 100;
let currentLabel = "";
let currentPrediction = null;

// ================================
// Initialization
// ================================

document.addEventListener("DOMContentLoaded", () => {
  initializeTabs();
  initializeButtons();
  checkStatus();
  loadLabels();
  loadDataStats();
});

// ================================
// Tab Management
// ================================

function initializeTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabName = button.dataset.tab;

      // Update active states
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      button.classList.add("active");
      document.getElementById(`${tabName}-tab`).classList.add("active");

      // Stop streams when switching tabs
      stopAllStreams();
    });
  });
}

// ================================
// Button Event Listeners
// ================================

function initializeButtons() {
  // Record tab
  document.getElementById("add-label-btn").addEventListener("click", addLabel);
  document
    .getElementById("start-camera-btn")
    .addEventListener("click", startRecordCamera);
  document
    .getElementById("capture-btn")
    .addEventListener("click", toggleCapture);

  // Train tab
  document.getElementById("train-btn").addEventListener("click", trainModel);

  // Classification tab
  document
    .getElementById("start-classify-btn")
    .addEventListener("click", startClassification);
  document
    .getElementById("stop-classify-btn")
    .addEventListener("click", stopClassification);

  // Sentence tab
  document
    .getElementById("start-sentence-btn")
    .addEventListener("click", startSentenceRecognition);
  document
    .getElementById("stop-sentence-btn")
    .addEventListener("click", stopSentenceRecognition);
  document
    .getElementById("add-to-sentence-btn")
    .addEventListener("click", addToSentence);
  document
    .getElementById("clear-sentence-btn")
    .addEventListener("click", clearSentence);
}

// ================================
// API Functions
// ================================

async function checkStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`);
    const data = await response.json();

    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    if (data.status === "online") {
      statusDot.style.background = "var(--success-color)";
      statusText.textContent = data.model_loaded ? "Model Ready" : "No Model";
    } else {
      statusDot.style.background = "var(--danger-color)";
      statusText.textContent = "Offline";
    }
  } catch (error) {
    showToast("Failed to connect to server", "error");
  }
}

async function loadLabels() {
  try {
    const response = await fetch(`${API_BASE}/api/labels`);
    const data = await response.json();

    const container = document.getElementById("labels-container");
    container.innerHTML = "";

    if (Object.keys(data.labels).length === 0) {
      container.innerHTML =
        '<p style="color: var(--text-secondary); text-align: center; grid-column: 1/-1;">No signs added yet. Add your first sign above!</p>';
      return;
    }

    // Load data stats to show counts
    const statsResponse = await fetch(`${API_BASE}/api/data/stats`);
    const statsData = await statsResponse.json();

    for (const [labelName, description] of Object.entries(data.labels)) {
      const count = statsData.stats[labelName] || 0;
      const card = createLabelCard(labelName, description, count);
      container.appendChild(card);
    }
  } catch (error) {
    console.error("Error loading labels:", error);
  }
}

function createLabelCard(name, description, count) {
  const card = document.createElement("div");
  card.className = "label-card";
  card.innerHTML = `
        <div class="label-card-header">
            <div class="label-name">${name}</div>
            <div class="label-count">${count} samples</div>
        </div>
        <div class="label-description">${description}</div>
        <div class="label-actions">
            <button class="btn btn-secondary btn-small" onclick="selectLabel('${name}')">Record More</button>
            <button class="btn btn-secondary btn-small" onclick="deleteLabel('${name}')">Delete</button>
        </div>
    `;
  return card;
}

async function addLabel() {
  const labelName = document.getElementById("label-name").value.trim();
  const labelDescription = document
    .getElementById("label-description")
    .value.trim();

  if (!labelName || !labelDescription) {
    showToast("Please enter both sign name and description", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("label_name", labelName);
  formData.append("label_description", labelDescription);

  try {
    const response = await fetch(`${API_BASE}/api/labels`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      showToast(`Sign "${labelName}" added successfully!`, "success");
      document.getElementById("label-name").value = "";
      document.getElementById("label-description").value = "";
      loadLabels();
    }
  } catch (error) {
    showToast("Failed to add label", "error");
  }
}

async function deleteLabel(labelName) {
  if (
    !confirm(
      `Are you sure you want to delete "${labelName}" and all its samples?`
    )
  ) {
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/labels/${labelName}`, {
      method: "DELETE",
    });

    const data = await response.json();

    if (data.success) {
      showToast(`Sign "${labelName}" deleted`, "success");
      loadLabels();
      loadDataStats();
    }
  } catch (error) {
    showToast("Failed to delete label", "error");
  }
}

function selectLabel(labelName) {
  document.getElementById("label-name").value = labelName;
  showToast(`Selected "${labelName}" for recording`, "success");
}

async function loadDataStats() {
  try {
    const response = await fetch(`${API_BASE}/api/data/stats`);
    const data = await response.json();

    document.getElementById("total-classes").textContent = data.num_classes;
    document.getElementById("total-images").textContent = data.total_images;
  } catch (error) {
    console.error("Error loading stats:", error);
  }
}

async function trainModel() {
  const btn = document.getElementById("train-btn");
  const output = document.getElementById("training-output");

  btn.disabled = true;
  btn.innerHTML = '<div class="loading"></div> Training...';
  output.classList.add("show");
  output.innerHTML = "<pre>Starting training process...</pre>";

  try {
    const response = await fetch(`${API_BASE}/api/train`, {
      method: "POST",
    });

    const data = await response.json();

    if (data.success) {
      output.innerHTML += `<pre style="color: var(--success-color);">\n${data.message}</pre>`;
      showToast("Model trained successfully!", "success");
      checkStatus();
    } else {
      output.innerHTML += `<pre style="color: var(--danger-color);">\nTraining failed</pre>`;
      showToast("Training failed", "error");
    }
  } catch (error) {
    output.innerHTML += `<pre style="color: var(--danger-color);">\nError: ${error.message}</pre>`;
    showToast("Training failed", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
            </svg>
            Train Model
        `;
  }
}

// ================================
// Camera & Recording
// ================================

async function startRecordCamera() {
  const video = document.getElementById("record-video");
  const btn = document.getElementById("start-camera-btn");

  try {
    recordStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });

    video.srcObject = recordStream;
    btn.textContent = "✓ Camera Active";
    btn.disabled = true;
    document.getElementById("capture-btn").disabled = false;
    showToast("Camera started", "success");
  } catch (error) {
    showToast("Failed to access camera", "error");
  }
}

async function toggleCapture() {
  if (isCapturing) {
    stopCapture();
  } else {
    startCapture();
  }
}

function startCapture() {
  const labelName = document.getElementById("label-name").value.trim();
  const numImages = parseInt(document.getElementById("num-images").value);

  if (!labelName) {
    showToast("Please enter a sign name first", "warning");
    return;
  }

  isCapturing = true;
  captureCount = 0;
  captureTarget = numImages;
  currentLabel = labelName;

  const btn = document.getElementById("capture-btn");
  btn.innerHTML = '<div class="loading"></div> Capturing...';

  captureLoop();
}

async function captureLoop() {
  if (!isCapturing) return;

  const video = document.getElementById("record-video");
  const canvas = document.getElementById("record-canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  const frameData = canvas.toDataURL("image/jpeg", 0.8);

  const formData = new FormData();
  formData.append("label_name", currentLabel);
  formData.append("num_images", captureTarget);
  formData.append("frame_data", frameData);

  try {
    const response = await fetch(`${API_BASE}/api/capture`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      captureCount = data.count;
      document.getElementById(
        "capture-counter"
      ).textContent = `${captureCount}/${captureTarget}`;

      if (captureCount >= captureTarget) {
        stopCapture();
        showToast(
          `Captured ${captureCount} samples for "${currentLabel}"!`,
          "success"
        );
        loadLabels();
        loadDataStats();
      } else {
        setTimeout(captureLoop, 100);
      }
    }
  } catch (error) {
    showToast("Capture failed", "error");
    stopCapture();
  }
}

function stopCapture() {
  isCapturing = false;
  const btn = document.getElementById("capture-btn");
  btn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <circle cx="12" cy="12" r="6" fill="currentColor"/>
        </svg>
        Start Capturing
    `;
}

// ================================
// Classification
// ================================

async function startClassification() {
  const video = document.getElementById("classify-video");
  const canvas = document.getElementById("classify-canvas");
  const startBtn = document.getElementById("start-classify-btn");
  const stopBtn = document.getElementById("stop-classify-btn");

  try {
    classifyStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
    });

    video.srcObject = classifyStream;

    // Wait for video to load
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });

    startBtn.disabled = true;
    stopBtn.disabled = false;

    // Connect WebSocket
    classifyWs = new WebSocket(`ws://${window.location.host}/ws/inference`);

    classifyWs.onopen = () => {
      showToast("Classification started", "success");
      classificationLoop();
    };

    classifyWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updatePredictionDisplay(data, "classify");
    };

    classifyWs.onerror = () => {
      showToast("WebSocket error", "error");
      stopClassification();
    };
  } catch (error) {
    showToast("Failed to start classification", "error");
  }
}

function classificationLoop() {
  if (!classifyWs || classifyWs.readyState !== WebSocket.OPEN) return;

  const video = document.getElementById("classify-video");
  const canvas = document.getElementById("classify-canvas");
  const ctx = canvas.getContext("2d");

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameData = canvas.toDataURL("image/jpeg", 0.8);

  classifyWs.send(JSON.stringify({ frame: frameData }));

  setTimeout(() => requestAnimationFrame(classificationLoop), 100);
}

function stopClassification() {
  if (classifyWs) {
    classifyWs.close();
    classifyWs = null;
  }

  if (classifyStream) {
    classifyStream.getTracks().forEach((track) => track.stop());
    classifyStream = null;
  }

  document.getElementById("start-classify-btn").disabled = false;
  document.getElementById("stop-classify-btn").disabled = true;
  document.getElementById("prediction-label").textContent = "No hand detected";
  document.getElementById("prediction-confidence").textContent = "";

  showToast("Classification stopped", "success");
}

// ================================
// Sentence Builder
// ================================

async function startSentenceRecognition() {
  const video = document.getElementById("sentence-video");
  const canvas = document.getElementById("sentence-canvas");
  const startBtn = document.getElementById("start-sentence-btn");
  const stopBtn = document.getElementById("stop-sentence-btn");
  const addBtn = document.getElementById("add-to-sentence-btn");

  try {
    sentenceStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
    });

    video.srcObject = sentenceStream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });

    startBtn.disabled = true;
    stopBtn.disabled = false;
    addBtn.disabled = false;

    sentenceWs = new WebSocket(`ws://${window.location.host}/ws/inference`);

    sentenceWs.onopen = () => {
      showToast("Sentence recognition started", "success");
      sentenceLoop();
    };

    sentenceWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      currentPrediction = data.prediction;
      updatePredictionDisplay(data, "sentence");
    };

    sentenceWs.onerror = () => {
      showToast("WebSocket error", "error");
      stopSentenceRecognition();
    };
  } catch (error) {
    showToast("Failed to start recognition", "error");
  }
}

function sentenceLoop() {
  if (!sentenceWs || sentenceWs.readyState !== WebSocket.OPEN) return;

  const video = document.getElementById("sentence-video");
  const canvas = document.getElementById("sentence-canvas");
  const ctx = canvas.getContext("2d");

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameData = canvas.toDataURL("image/jpeg", 0.8);

  sentenceWs.send(JSON.stringify({ frame: frameData }));

  setTimeout(() => requestAnimationFrame(sentenceLoop), 100);
}

function stopSentenceRecognition() {
  if (sentenceWs) {
    sentenceWs.close();
    sentenceWs = null;
  }

  if (sentenceStream) {
    sentenceStream.getTracks().forEach((track) => track.stop());
    sentenceStream = null;
  }

  document.getElementById("start-sentence-btn").disabled = false;
  document.getElementById("stop-sentence-btn").disabled = true;
  document.getElementById("add-to-sentence-btn").disabled = true;
  document.getElementById("sentence-prediction-label").textContent =
    "No hand detected";
  document.getElementById("sentence-prediction-confidence").textContent = "";

  currentPrediction = null;
  showToast("Recognition stopped", "success");
}

async function addToSentence() {
  if (!currentPrediction) {
    showToast("No sign detected", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("sign", currentPrediction);

  try {
    const response = await fetch(`${API_BASE}/api/sentence/add`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      updateSentenceDisplay(data.text);
      showToast(`Added "${currentPrediction}" to sentence`, "success");
    }
  } catch (error) {
    showToast("Failed to add to sentence", "error");
  }
}

async function clearSentence() {
  try {
    const response = await fetch(`${API_BASE}/api/sentence/clear`, {
      method: "POST",
    });

    const data = await response.json();

    if (data.success) {
      updateSentenceDisplay("");
      showToast("Sentence cleared", "success");
    }
  } catch (error) {
    showToast("Failed to clear sentence", "error");
  }
}

function updateSentenceDisplay(text) {
  const sentenceText = document.getElementById("sentence-text");
  sentenceText.textContent = text || "No signs added yet";
}

// ================================
// UI Updates
// ================================

function updatePredictionDisplay(data, mode) {
  const prefix = mode === "sentence" ? "sentence-" : "";
  const labelEl = document.getElementById(`${prefix}prediction-label`);
  const confidenceEl = document.getElementById(
    `${prefix}prediction-confidence`
  );

  if (data.prediction) {
    labelEl.textContent = data.prediction;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(
      1
    )}% confidence`;
    labelEl.style.color = "var(--primary-light)";
  } else {
    labelEl.textContent = "No hand detected";
    confidenceEl.textContent = "";
    labelEl.style.color = "var(--text-secondary)";
  }
}

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success:
      '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    error:
      '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    warning:
      '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 9V13M12 17H12.01M21 12C21 16.97 16.97 21 12 21C7.03 21 3 16.97 3 12C3 7.03 7.03 3 12 3C16.97 3 21 7.03 21 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
  };

  toast.innerHTML = `
        ${icons[type] || icons.success}
        <div class="toast-message">${message}</div>
    `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-in";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ================================
// Cleanup
// ================================

function stopAllStreams() {
  if (recordStream) {
    recordStream.getTracks().forEach((track) => track.stop());
    recordStream = null;
  }

  stopClassification();
  stopSentenceRecognition();
  stopCapture();
}

window.addEventListener("beforeunload", stopAllStreams);
