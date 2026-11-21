"use strict";

const HTTP_CODES = Object.freeze({
  BAD_REQUEST: 400,
  FORBIDDEN: 403,
  GONE: 410,
  INTERNAL_ERROR: 500,
  NOT_FOUND: 404,
  OK: 200,
  TOO_MANY_REQUESTS: 429
});

const UI_CONSTANTS = Object.freeze({
  FIRST_INDEX: 0,
  INCREMENT_ONE: 1,
  STAGE_COUNT: 4
});

const VIEW_IDS = Object.freeze([
  "view-startpage",
  "view-base-home",
  "view-new-rescue",
  "view-portal-dashboard",
  "view-reference-layers"
]);

const ENDPOINTS = Object.freeze({
  BATCH_GET: "/api/batch",
  BATCH_RUN: "/api/batch/run",
  DEPLOY_SWITCH: "/api/deploy/switch",
  GDPR_ERASE: "/api/gdpr/erase",
  LAYERS: "/api/layers",
  RAG_QUERY: "/api/rag/query",
  RESCUES_LIST: "/api/rescues",
  RESCUE_CREATE: "/api/rescue/create",
  RESET_ISSUE: "/api/reset/issue",
  RESET_REDEEM: "/api/reset/redeem",
  STATUS: "/api/status"
});

const ERROR_SCREEN_COPY = Object.freeze({
  [HTTP_CODES.BAD_REQUEST]: Object.freeze({
    hint: "Verify input fields contain no SQLi or XSS tokens.",
    title: "400 Bad Request - Input Validation Blocked"
  }),
  [HTTP_CODES.FORBIDDEN]: Object.freeze({
    hint: "Verify your Responder ID matches the batch owner_id.",
    title: "403 Forbidden - BOLA or Strict CORS Protection"
  }),
  [HTTP_CODES.NOT_FOUND]: Object.freeze({
    hint: "Confirm the requested relief batch ID exists.",
    title: "404 Not Found - Relief Resource Missing"
  }),
  [HTTP_CODES.GONE]: Object.freeze({
    hint: "Request a fresh single-use 30-minute reset link.",
    title: "410 Gone - Password Reset Link Expired or Used"
  }),
  [HTTP_CODES.TOO_MANY_REQUESTS]: Object.freeze({
    hint: "Wait for the 60-second sliding rate window to reset.",
    title: "429 Too Many Requests - Cloud Cost Rate Limiter"
  }),
  [HTTP_CODES.INTERNAL_ERROR]: Object.freeze({
    hint: "An immediate breakage alert was dispatched to on-call.",
    title: "500 Internal Error - Global Catch-All Boundary"
  })
});

function getElement(id) {
  return document.getElementById(id);
}

function switchActiveScreen(targetViewId) {
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < VIEW_IDS.length;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const viewEl = getElement(VIEW_IDS[idx]);
    if (viewEl) {
      if (VIEW_IDS[idx] === targetViewId) {
        viewEl.classList.remove("hidden");
      } else {
        viewEl.classList.add("hidden");
      }
    }
  }
  const navButtons = document.querySelectorAll(".nav-tab-btn");
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < navButtons.length;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const btn = navButtons[idx];
    if (btn.getAttribute("data-target") === targetViewId) {
      btn.classList.add("nav-tab-active");
    } else {
      btn.classList.remove("nav-tab-active");
    }
  }
}

function showErrorBoundary(statusCode, serverScreen) {
  const panel = getElement("error-boundary-panel");
  const titleEl = getElement("error-boundary-title");
  const msgEl = getElement("error-boundary-message");
  const hintEl = getElement("error-boundary-hint");
  if (!panel || !titleEl || !msgEl || !hintEl) {
    return;
  }
  const fallback =
    ERROR_SCREEN_COPY[statusCode] ||
    ERROR_SCREEN_COPY[HTTP_CODES.INTERNAL_ERROR];
  titleEl.textContent =
    serverScreen && serverScreen.title
      ? `${serverScreen.status_code}: ${serverScreen.title}`
      : fallback.title;
  msgEl.textContent =
    serverScreen && serverScreen.message
      ? serverScreen.message
      : "Operation halted safely by the pre-deployment boundary.";
  hintEl.textContent =
    serverScreen && serverScreen.action_hint
      ? serverScreen.action_hint
      : fallback.hint;
  panel.classList.remove("hidden");
}

function clearErrorBoundary() {
  const panel = getElement("error-boundary-panel");
  if (panel) {
    panel.classList.add("hidden");
  }
}

async function requestJson(url, options) {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (response.status !== HTTP_CODES.OK || !data.ok) {
      showErrorBoundary(response.status, data.error_screen);
      return null;
    }
    clearErrorBoundary();
    return data;
  } catch (_err) {
    showErrorBoundary(HTTP_CODES.INTERNAL_ERROR, null);
    return null;
  }
}

function updateStageCardsAndLayers(batch) {
  const stageLogs = batch.stage_logs || [];
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < UI_CONSTANTS.STAGE_COUNT;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const card = getElement(`stage-card-${idx}`);
    const detail = getElement(`stage-detail-${idx}`);
    if (card && detail && stageLogs[idx]) {
      card.classList.add("stage-card-active");
      detail.textContent =
        `${stageLogs[idx].status}: ${stageLogs[idx].detail}`;
    }
  }
  const ragEl = getElement("rag-advisory-output");
  if (ragEl && batch.rag_advisory) {
    ragEl.textContent = batch.rag_advisory;
  }
  const zarrEl = getElement("startpage-zarr-summary");
  if (zarrEl && batch.zarr_manifest) {
    zarrEl.textContent =
      `OME-Zarr: ${batch.zarr_manifest.store_name} | ` +
      `Chunks: ${batch.zarr_manifest.chunk_count} | ` +
      `Clusters: ${batch.transitivity_clusters}`;
  }
  const layers = batch.reference_layers || [];
  const layerElIds = [
    "layer-detail-h3",
    "layer-detail-dem",
    "layer-detail-thermal",
    "layer-detail-dashcam"
  ];
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < layerElIds.length;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const lEl = getElement(layerElIds[idx]);
    if (lEl && layers[idx]) {
      lEl.textContent = layers[idx].summary_label;
    }
  }
}

async function refreshSystemStatus() {
  const statusData = await requestJson(ENDPOINTS.STATUS, {
    method: "GET"
  });
  if (!statusData) {
    return;
  }
  const regionEl = getElement("pill-region");
  const slotEl = getElement("pill-slot");
  const ledgerEl = getElement("pill-ledger");
  const connEl = getElement("connection-summary");
  if (regionEl) {
    regionEl.textContent = `Region: ${statusData.cloud_region}`;
  }
  if (slotEl) {
    slotEl.textContent =
      `Active Slot: ${statusData.active_slot.toUpperCase()} ` +
      `(Rollbacks: ${statusData.rollback_count})`;
  }
  if (ledgerEl) {
    ledgerEl.textContent = statusData.ledger_chain_valid
      ? "Ledger: SHA-256 Verified"
      : "Ledger: Chain Alert";
  }
  if (connEl) {
    connEl.textContent =
      `Indexes: ${statusData.custom_db_indexes.join(", ")} | ` +
      `Logs: ${statusData.log_ring_count} | ` +
      `Alerts: ${statusData.alert_count}`;
  }
}

async function handleLaunchBatch() {
  const batchId = getElement("input-batch-id").value;
  const ownerId = getElement("input-owner-id").value;
  const subjectId = getElement("input-subject-id").value;
  const transportMode = getElement("select-transport-mode").value;
  const terrainSector = getElement("select-terrain-sector").value;
  const weatherState = getElement("input-weather-state").value;
  const payload = await requestJson(ENDPOINTS.RESCUE_CREATE, {
    body: JSON.stringify({
      batch_id: batchId,
      owner_id: ownerId,
      subject_id: subjectId,
      terrain_sector: terrainSector,
      transport_mode: transportMode,
      weather_state: weatherState
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST"
  });
  if (payload && payload.batch) {
    updateStageCardsAndLayers(payload.batch);
    switchActiveScreen("view-base-home");
    const metaEl = getElement("batch-telemetry-summary");
    if (metaEl) {
      metaEl.textContent =
        `Batch ${payload.batch.batch_id} | ` +
        `Detections: ${payload.batch.total_detections} | ` +
        `Clusters: ${payload.batch.transitivity_clusters} | ` +
        `H3 Res: ${payload.batch.effective_h3_resolution} (U=0.0)`;
    }
    await refreshSystemStatus();
  }
}

async function handleQueryRag() {
  const transportMode = getElement("select-transport-mode").value;
  const terrainSector = getElement("select-terrain-sector").value;
  const weatherState = getElement("input-weather-state").value;
  const res = await requestJson(ENDPOINTS.RAG_QUERY, {
    body: JSON.stringify({
      query: terrainSector,
      transport_mode: transportMode,
      weather_state: weatherState
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST"
  });
  const ragEl = getElement("rag-advisory-output");
  if (res && ragEl) {
    ragEl.textContent = res.advisory;
  }
}

async function handleRefreshPortal() {
  const operatorId = getElement("input-portal-operator").value;
  const res = await requestJson(
    `${ENDPOINTS.RESCUES_LIST}?owner_id=${encodeURIComponent(operatorId)}`,
    {
      headers: { "X-Operator-Id": operatorId },
      method: "GET"
    }
  );
  const grid = getElement("portal-rescues-grid");
  if (!res || !grid) {
    return;
  }
  grid.textContent = "";
  const rescues = res.rescues || [];
  if (rescues.length === UI_CONSTANTS.FIRST_INDEX) {
    const emptyCard = document.createElement("article");
    emptyCard.className = "stage-card";
    emptyCard.textContent = `No relief batches found for ${res.owner_id}.`;
    grid.appendChild(emptyCard);
    return;
  }
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < rescues.length;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const item = rescues[idx];
    const card = document.createElement("article");
    card.className = "stage-card stage-card-active";
    const title = document.createElement("h3");
    title.className = "stage-title";
    title.textContent =
      `${item.batch_id} (${item.summary.transport_mode})`;
    const detail = document.createElement("p");
    detail.className = "stage-detail";
    detail.textContent =
      `Responder: ${item.owner_id} | Subject: ${item.subject_token} | ` +
      `Sector: ${item.summary.terrain_sector} | ` +
      `Clusters: ${item.summary.transitivity_clusters}`;
    card.appendChild(title);
    card.appendChild(detail);
    grid.appendChild(card);
  }
}

async function handleGdprErasure() {
  const subjectId = getElement("input-subject-id").value;
  const res = await requestJson(ENDPOINTS.GDPR_ERASE, {
    body: JSON.stringify({ subject_id: subjectId }),
    headers: { "Content-Type": "application/json" },
    method: "POST"
  });
  const modalStatus = getElement("modal-status-text");
  if (res && modalStatus) {
    modalStatus.textContent =
      `Art. 17 Erasure Complete: ` +
      `DB rows=${res.erasure_report.deleted_db_rows}, ` +
      `JSON=${res.erasure_report.scrubbed_json_files}, ` +
      `Zarr=${res.erasure_report.scrubbed_zarr_stores}`;
    await refreshSystemStatus();
  }
}

async function handleBlueGreenPromote() {
  const res = await requestJson(ENDPOINTS.DEPLOY_SWITCH, {
    body: JSON.stringify({
      health_verified: true,
      version: "2.1.0"
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST"
  });
  const modalStatus = getElement("modal-status-text");
  if (res && modalStatus) {
    modalStatus.textContent =
      `Active Slot: ${res.deployment.active_slot.toUpperCase()} | ` +
      `Version: ${res.deployment.active_version}`;
    await refreshSystemStatus();
  }
}

function bindEvents() {
  const navButtons = document.querySelectorAll(".nav-tab-btn");
  for (
    let idx = UI_CONSTANTS.FIRST_INDEX;
    idx < navButtons.length;
    idx += UI_CONSTANTS.INCREMENT_ONE
  ) {
    const btn = navButtons[idx];
    btn.addEventListener("click", () => {
      switchActiveScreen(btn.getAttribute("data-target"));
    });
  }
  const enterBtn = getElement("btn-enter-command");
  const launchBtn = getElement("btn-launch-batch");
  const createRescueBtn = getElement("btn-create-rescue-flow");
  const queryRagBtn = getElement("btn-query-rag");
  const refreshPortalBtn = getElement("btn-refresh-portal");
  const openModalBtn = getElement("btn-open-gdpr-modal");
  const closeModalBtn = getElement("btn-close-gdpr-modal");
  const eraseBtn = getElement("btn-trigger-erasure");
  const switchOkBtn = getElement("btn-switch-slot-ok");
  const dismissErrBtn = getElement("btn-dismiss-error");
  if (enterBtn) {
    enterBtn.addEventListener("click", () => {
      switchActiveScreen("view-base-home");
    });
  }
  if (launchBtn) {
    launchBtn.addEventListener("click", handleLaunchBatch);
  }
  if (createRescueBtn) {
    createRescueBtn.addEventListener("click", handleLaunchBatch);
  }
  if (queryRagBtn) {
    queryRagBtn.addEventListener("click", handleQueryRag);
  }
  if (refreshPortalBtn) {
    refreshPortalBtn.addEventListener("click", handleRefreshPortal);
  }
  if (openModalBtn) {
    openModalBtn.addEventListener("click", () => {
      getElement("gdpr-modal").classList.remove("hidden");
    });
  }
  if (closeModalBtn) {
    closeModalBtn.addEventListener("click", () => {
      getElement("gdpr-modal").classList.add("hidden");
    });
  }
  if (eraseBtn) {
    eraseBtn.addEventListener("click", handleGdprErasure);
  }
  if (switchOkBtn) {
    switchOkBtn.addEventListener("click", handleBlueGreenPromote);
  }
  if (dismissErrBtn) {
    dismissErrBtn.addEventListener("click", clearErrorBoundary);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  refreshSystemStatus();
});
