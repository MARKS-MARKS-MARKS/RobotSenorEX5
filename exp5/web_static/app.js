const state = {
  sessions: [],
  selectedSession: null,
  activeTab: "annotated",
  currentRun: null,
  pollTimer: null,
  roiNorm: null,
  drawingRoi: false,
  roiStart: null,
};

const nodes = {
  healthBadge: document.querySelector("#healthBadge"),
  runBadge: document.querySelector("#runBadge"),
  sessionList: document.querySelector("#sessionList"),
  refreshSessions: document.querySelector("#refreshSessions"),
  runButton: document.querySelector("#runButton"),
  placeButton: document.querySelector("#placeButton"),
  newItem: document.querySelector("#newItem"),
  itemSize: document.querySelector("#itemSize"),
  detector: document.querySelector("#detector"),
  autoRoi: document.querySelector("#autoRoi"),
  useUnknownObstacles: document.querySelector("#useUnknownObstacles"),
  drawRoiButton: document.querySelector("#drawRoiButton"),
  clearRoiButton: document.querySelector("#clearRoiButton"),
  roiHint: document.querySelector("#roiHint"),
  roiBox: document.querySelector("#roiBox"),
  activeTitle: document.querySelector("#activeTitle"),
  mainImage: document.querySelector("#mainImage"),
  imageStage: document.querySelector(".image-stage"),
  tabs: document.querySelector("#tabs"),
  summaryGrid: document.querySelector("#summaryGrid"),
  recommendation: document.querySelector("#recommendation"),
  objectList: document.querySelector("#objectList"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = data.sessions;
    state.selectedSession = state.selectedSession || state.sessions[0]?.name || null;
    nodes.healthBadge.textContent = "后端已连接";
    renderSessions();
    renderSelectedSessionPreview();
    FaceMotion.set("idle", "数据就绪", "选择一个 session 后，我会陪你盯着推理状态。");
  } catch (error) {
    nodes.healthBadge.textContent = "连接失败";
    FaceMotion.set("error", "连接失败", error.message);
  }
}

function renderSessions() {
  nodes.sessionList.innerHTML = "";
  if (!state.sessions.length) {
    nodes.sessionList.innerHTML = '<div class="recommendation">没有找到 data/sessions 数据。</div>';
    return;
  }
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.className = `session-card ${session.name === state.selectedSession ? "active" : ""}`;
    button.innerHTML = `
      <strong>${session.name}</strong>
      <span>${session.has_replay_detections ? `含 ${session.detection_count} 个 replay 标注` : "无 replay 标注"}</span>
    `;
    button.addEventListener("click", () => {
      clearTimeout(state.pollTimer);
      state.selectedSession = session.name;
      state.currentRun = null;
      state.roiNorm = null;
      renderSessions();
      renderSelectedSessionPreview();
      resetResult();
    });
    nodes.sessionList.appendChild(button);
  }
}

function renderSelectedSessionPreview() {
  const session = getSelectedSession();
  nodes.activeTitle.textContent = session ? `Session · ${session.name}` : "请选择一个 session";
  if (!session) {
    setImage(null);
    return;
  }
  setImage(session.assets.color);
}

function resetResult() {
  nodes.runBadge.textContent = "等待任务";
  nodes.runBadge.className = "status-pill muted";
  setSummary({});
  nodes.recommendation.textContent = "等待推理结果。";
  nodes.objectList.innerHTML = "";
  nodes.placeButton.disabled = true;
  renderRoiBox();
  FaceMotion.set("idle", "准备好了", "选择参数后点击开始推理。");
}

async function createRun() {
  if (!state.selectedSession) {
    FaceMotion.set("warn", "还没选数据", "先在左侧选一个 data session。");
    return;
  }
  nodes.runButton.disabled = true;
  nodes.placeButton.disabled = true;
  nodes.runBadge.textContent = "提交检测";
  nodes.runBadge.className = "status-pill muted";
  nodes.recommendation.textContent = "检测完成后，再根据新物体尺寸选择插入空位。";
  FaceMotion.set("processing", "开始检测", "我正在读取 RGB-D 数据并运行识别。");
  try {
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        session: state.selectedSession,
        detector: nodes.detector.value,
        auto_roi: nodes.autoRoi.checked,
        roi_norm: state.roiNorm,
        use_unknown_obstacles: nodes.useUnknownObstacles.checked,
      }),
    });
    state.currentRun = run;
    pollRun(run.id);
  } catch (error) {
    nodes.runButton.disabled = false;
    nodes.placeButton.disabled = true;
    nodes.runBadge.textContent = "任务失败";
    FaceMotion.set("error", "启动失败", error.message);
  }
}

async function updatePlacement() {
  if (!state.currentRun || state.currentRun.status !== "done") {
    FaceMotion.set("warn", "还没检测", "先点击“重新检测当前图片”，再选择插入空位。");
    return;
  }
  nodes.placeButton.disabled = true;
  nodes.runBadge.textContent = "选择空位";
  FaceMotion.set("processing", "正在选空", "不重新检测图片，只按新物体类别和尺寸重新推荐。");
  try {
    const run = await api(`/api/runs/${state.currentRun.id}/placement`, {
      method: "POST",
      body: JSON.stringify({
        new_item: nodes.newItem.value,
        item_size: nodes.itemSize.value,
      }),
    });
    state.currentRun = run;
    renderRun(run, { placementUpdated: true });
  } catch (error) {
    FaceMotion.set("error", "选空失败", error.message);
  } finally {
    nodes.placeButton.disabled = false;
  }
}

async function pollRun(runId) {
  clearTimeout(state.pollTimer);
  try {
    const run = await api(`/api/runs/${runId}`);
    state.currentRun = run;
    renderRun(run);
    if (run.status === "queued" || run.status === "running") {
      state.pollTimer = setTimeout(() => pollRun(runId), 1200);
    } else {
      nodes.runButton.disabled = false;
      nodes.placeButton.disabled = run.status === "done" ? false : true;
    }
  } catch (error) {
    nodes.runButton.disabled = false;
    FaceMotion.set("error", "状态丢失", error.message);
  }
}

function renderRun(run, options = {}) {
  nodes.runBadge.textContent = statusText(run.status);
  nodes.runBadge.className = `status-pill ${run.status === "done" ? "" : "muted"}`;
  if (run.status === "queued" || run.status === "running") {
    FaceMotion.set("processing", "推理中", "模型可能需要一点时间，尤其是首次加载 GroundingDINO。");
    return;
  }
  if (run.status === "error") {
    FaceMotion.set("error", "处理失败", run.error || "未知错误");
    nodes.recommendation.textContent = run.error || "处理失败。";
    return;
  }
  const result = run.result || {};
  setSummary(result.counts || {});
  if (options.placementUpdated || run.item_size) {
    renderRecommendation(result.recommendation_summary);
  } else {
    nodes.recommendation.textContent = "检测已完成。输入新物体名称和尺寸后，点击“根据新物体选择空位”。";
  }
  renderObjects(result.objects || []);
  const rec = result.recommendation_summary;
  if ((options.placementUpdated || run.item_size) && (!rec || !rec.shelf_level)) {
    FaceMotion.set("warn", "没有合适空位", "当前候选空位放不下这个新物体，可以调小尺寸或重框 ROI。");
  } else if (options.placementUpdated || run.item_size) {
    FaceMotion.set("success", "找到空位", "我已经按类别相近和空间限制重新推荐。");
  } else {
    FaceMotion.set("success", "检测完成", "物体和空位已识别，现在可以选择要插入的新物体。");
  }
  renderActiveImage();
}

function renderActiveImage() {
  const run = state.currentRun;
  const session = getSelectedSession();
  const assets = run?.assets || {};
  const sourceMap = {
    annotated: assets["annotated.png"],
    top: assets["top_view.png"],
    map: assets["map_3d.png"],
    source: session?.assets.color,
    depth: session?.assets.depth,
  };
  setImage(sourceMap[state.activeTab] || null);
  setTimeout(renderRoiBox, 0);
}

function setImage(src) {
  if (!src) {
    nodes.mainImage.removeAttribute("src");
    nodes.imageStage.classList.remove("has-image");
    return;
  }
  nodes.mainImage.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  nodes.imageStage.classList.add("has-image");
  nodes.mainImage.onload = renderRoiBox;
}

function setSummary(counts) {
  const values = [
    ["Objects", counts.objects || 0],
    ["Shelves", counts.shelves || 0],
    ["Spaces", counts.empty_spaces || 0],
    ["Obstacles", counts.obstacles || 0],
  ];
  nodes.summaryGrid.innerHTML = values.map(([label, value]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join("");
}

function renderRecommendation(rec) {
  if (!rec || !rec.shelf_level) {
    nodes.recommendation.textContent = "没有可用推荐位置。";
    return;
  }
  const center = (rec.center_m || []).map((value) => `${(value * 1000).toFixed(1)} mm`).join(", ");
  const itemSize = nodes.itemSize.value || "默认尺寸";
  nodes.recommendation.innerHTML = `
    <strong>把 ${rec.new_item_category || "新物体"} 放到第 ${rec.shelf_level} 层高亮空位</strong>
    <p>新物体尺寸：${itemSize} m</p>
    <p>推荐中心：${center}</p>
    <p>${rec.reason || ""}</p>
  `;
}

function renderObjects(objects) {
  nodes.objectList.innerHTML = objects.length
    ? objects.map((item) => `
        <div class="object-item">
          <strong>${item.category || "Unknown"} / ${item.label}</strong>
          <span>score ${(item.score || 0).toFixed(2)} · box ${item.box?.join(", ")}</span>
        </div>
      `).join("")
    : '<div class="recommendation">暂无物体结果。</div>';
}

function getSelectedSession() {
  return state.sessions.find((item) => item.name === state.selectedSession);
}

function statusText(status) {
  return {
    queued: "排队中",
    running: "推理中",
    done: "已完成",
    error: "失败",
  }[status] || status;
}

nodes.refreshSessions.addEventListener("click", loadSessions);
nodes.runButton.addEventListener("click", createRun);
nodes.placeButton.addEventListener("click", updatePlacement);
nodes.drawRoiButton.addEventListener("click", () => {
  state.activeTab = "source";
  setActiveTabButton("source");
  renderActiveImage();
  state.drawingRoi = true;
  nodes.imageStage.classList.add("roi-drawing");
  nodes.roiHint.textContent = "在原图上按住拖拽柜体区域，松开后保存 ROI。";
  FaceMotion.set("idle", "手动 ROI", "拖拽框住柜体，越贴合柜子越能减少背景空位。");
});
nodes.clearRoiButton.addEventListener("click", () => {
  state.roiNorm = null;
  state.drawingRoi = false;
  nodes.imageStage.classList.remove("roi-drawing");
  nodes.roiHint.textContent = "已清除手动 ROI，将使用自动 ROI。";
  renderRoiBox();
});
nodes.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-tab]");
  if (!button) {
    return;
  }
  state.activeTab = button.dataset.tab;
  setActiveTabButton(state.activeTab);
  renderActiveImage();
});
nodes.imageStage.addEventListener("pointerdown", (event) => {
  if (!state.drawingRoi || state.activeTab !== "source" || !nodes.imageStage.classList.contains("has-image")) {
    return;
  }
  event.preventDefault();
  nodes.imageStage.setPointerCapture?.(event.pointerId);
  const point = eventToImageNorm(event);
  if (!point) {
    return;
  }
  state.roiStart = point;
  state.roiNorm = [point.x, point.y, point.x, point.y];
  renderRoiBox();
});
nodes.imageStage.addEventListener("pointermove", (event) => {
  if (!state.drawingRoi || !state.roiStart) {
    return;
  }
  event.preventDefault();
  const point = eventToImageNorm(event);
  if (!point) {
    return;
  }
  state.roiNorm = [
    Math.min(state.roiStart.x, point.x),
    Math.min(state.roiStart.y, point.y),
    Math.max(state.roiStart.x, point.x),
    Math.max(state.roiStart.y, point.y),
  ];
  renderRoiBox();
});
window.addEventListener("pointerup", () => {
  if (!state.drawingRoi || !state.roiStart) {
    return;
  }
  state.roiStart = null;
  state.drawingRoi = false;
  nodes.imageStage.classList.remove("roi-drawing");
  if (state.roiNorm && (state.roiNorm[2] - state.roiNorm[0] < 0.02 || state.roiNorm[3] - state.roiNorm[1] < 0.02)) {
    state.roiNorm = null;
    nodes.roiHint.textContent = "ROI 太小，已忽略。";
  } else {
    nodes.autoRoi.checked = false;
    nodes.roiHint.textContent = `手动 ROI：${state.roiNorm.map((value) => value.toFixed(3)).join(", ")}`;
  }
  renderRoiBox();
});
window.addEventListener("resize", renderRoiBox);

loadSessions();

function setActiveTabButton(tab) {
  nodes.tabs.querySelectorAll(".tab-button").forEach((item) => item.classList.toggle("active", item.dataset.tab === tab));
}

function eventToImageNorm(event) {
  const rect = getDisplayedImageRect();
  if (!rect) {
    return null;
  }
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) {
    return null;
  }
  return { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
}

function getDisplayedImageRect() {
  const image = nodes.mainImage;
  if (!image.naturalWidth || !image.naturalHeight) {
    return null;
  }
  const rect = image.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 ? rect : null;
}

function renderRoiBox() {
  if (!state.roiNorm || state.activeTab !== "source") {
    nodes.roiBox.classList.remove("active");
    return;
  }
  const rect = getDisplayedImageRect();
  const stageRect = nodes.imageStage.getBoundingClientRect();
  if (!rect) {
    nodes.roiBox.classList.remove("active");
    return;
  }
  const [x1, y1, x2, y2] = state.roiNorm;
  nodes.roiBox.style.left = `${rect.left - stageRect.left + x1 * rect.width}px`;
  nodes.roiBox.style.top = `${rect.top - stageRect.top + y1 * rect.height}px`;
  nodes.roiBox.style.width = `${Math.max(1, (x2 - x1) * rect.width)}px`;
  nodes.roiBox.style.height = `${Math.max(1, (y2 - y1) * rect.height)}px`;
  nodes.roiBox.classList.add("active");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
