const imageInput = document.getElementById("imageInput");
const pasteButton = document.getElementById("pasteButton");
const pasteInput = document.getElementById("pasteInput");
const preview = document.getElementById("preview");
const examplesGrid = document.getElementById("examplesGrid");
const examplesState = document.getElementById("examplesState");
const atomCutoffInput = document.getElementById("atomCutoffInput");
const chargeCutoffInput = document.getElementById("chargeCutoffInput");
const bondCutoffInput = document.getElementById("bondCutoffInput");
const atomNmsInput = document.getElementById("atomNmsInput");
const chargeNmsInput = document.getElementById("chargeNmsInput");
const runButton = document.getElementById("runButton");
const addAtomButton = document.getElementById("addAtomButton");
const addChargeButton = document.getElementById("addChargeButton");
const addBondButton = document.getElementById("addBondButton");
const editAtomButton = document.getElementById("editAtomButton");
const editChargeButton = document.getElementById("editChargeButton");
const editBondButton = document.getElementById("editBondButton");
const deleteDetectionButton = document.getElementById("deleteDetectionButton");
const loadDetectionHistoryButton = document.getElementById("loadDetectionHistoryButton");
const updateDetectionsButton = document.getElementById("updateDetectionsButton");
const fixButton = document.getElementById("fixButton");
const loadOriginalButton = document.getElementById("loadOriginalButton");
const loadExpandedButton = document.getElementById("loadExpandedButton");
const loadPreferredButton = document.getElementById("loadPreferredButton");
const statusNode = document.getElementById("status");
const skeletonSmiles = document.getElementById("skeletonSmiles");
const expandedSmiles = document.getElementById("expandedSmiles");
const molfileSourceState = document.getElementById("molfileSourceState");
const alignmentState = document.getElementById("alignmentState");
const alignmentMeta = document.getElementById("alignmentMeta");
const alignmentCompareSvg = document.getElementById("alignmentCompareSvg");
const molfileOutput = document.getElementById("molfileOutput");
const graphOutput = document.getElementById("graphOutput");
const detectionsOutput = document.getElementById("detectionsOutput");
const ocrFixButton = document.getElementById("ocrFixButton");
const ocrState = document.getElementById("ocrState");
const ocrBody = document.getElementById("ocrBody");
const selectionMeta = document.getElementById("selectionMeta");
const detectionState = document.getElementById("detectionState");
const acCount = document.getElementById("acCount");
const bondCount = document.getElementById("bondCount");
const acEditorSvg = document.getElementById("acEditorSvg");
const bondEditorSvg = document.getElementById("bondEditorSvg");
const editorState = document.getElementById("editorState");
const editorFallback = document.getElementById("editorFallback");
const ketcherFrame = document.getElementById("ketcherFrame");
const diagnosticsState = document.getElementById("diagnosticsState");
const diagnosticsHint = document.getElementById("diagnosticsHint");
const diagnosticsLog = document.getElementById("diagnosticsLog");

const diagnosticsEntries = [];
const MAX_DIAGNOSTICS = 60;
const KETCHER_LOAD_TIMEOUT_MS = 60000;
const HANDLE_SIZE = 8;
const editorViews = {
  ac: { svg: acEditorSvg, countNode: acCount, emptyLabel: "No atoms or charges" },
  b: { svg: bondEditorSvg, countNode: bondCount, emptyLabel: "No bonds" },
};

let ketcherReady = false;
let ketcherLoadTimer = null;
let currentPrediction = null;
let currentImageHref = null;
let currentImageFile = null;
let currentImageSourceKey = null;
let currentImageOrigin = "upload";
let exampleItems = [];
let editableDetections = { ac: [], b: [] };
let ocrDetections = [];
let selectedItem = { editorKey: null, index: null };
let dragState = null;
let activeMolfileSource = "none";
let pendingAddMode = null;
let itemUidCounter = 0;
let inferenceProfile = "legacy";

function setStatus(message) {
  statusNode.textContent = message;
}

function setDiagnosticsState(label, tone = "subtle") {
  diagnosticsState.textContent = label;
  diagnosticsState.className = "tag tag-" + tone;
}

function setDetectionState(label, tone = "subtle") {
  detectionState.textContent = label;
  detectionState.className = "tag tag-" + tone;
}

function setMolfileSourceState(label, tone = "subtle") {
  molfileSourceState.textContent = "Molfile source: " + label;
  molfileSourceState.className = "tag tag-" + tone;
}

function setAlignmentState(label, tone = "subtle") {
  alignmentState.textContent = label;
  alignmentState.className = "tag tag-" + tone;
}

function setExamplesState(label, tone = "subtle") {
  examplesState.textContent = label;
  examplesState.className = "tag tag-" + tone;
}

function stringifyDetail(detail) {
  if (detail instanceof Error) {
    return detail.name + ": " + detail.message;
  }
  if (typeof detail === "string") {
    return detail;
  }
  try {
    return JSON.stringify(detail, null, 2);
  } catch {
    return String(detail);
  }
}

function escapeXml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function logDiagnostic(level, message, detail = null) {
  const timestamp = new Date().toLocaleTimeString();
  const lines = ["[" + timestamp + "] " + level.toUpperCase() + " " + message];
  if (detail !== null && detail !== undefined) {
    lines.push(stringifyDetail(detail));
  }
  diagnosticsEntries.unshift(lines.join("\n"));
  diagnosticsLog.textContent = diagnosticsEntries.slice(0, MAX_DIAGNOSTICS).join("\n\n");
}

function updatePreview(file) {
  currentImageFile = file || null;
  if (currentImageHref) {
    URL.revokeObjectURL(currentImageHref);
    currentImageHref = null;
  }
  if (!file) {
    preview.style.display = "none";
    preview.removeAttribute("src");
    return;
  }
  currentImageHref = URL.createObjectURL(file);
  preview.src = currentImageHref;
  preview.style.display = "block";
}

function isEditableTarget(target) {
  if (!target || !(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

function focusPasteZone(active) {
  pasteButton.classList.toggle("is-active", active);
  if (!active) {
    pasteInput.value = "";
  }
}

function resetPredictionStateForNewImage() {
  currentPrediction = null;
  activeMolfileSource = "none";
  setPendingAddMode(null);
  editableDetections = { ac: [], b: [] };
  ocrDetections = [];
  selectedItem = { editorKey: null, index: null };
  skeletonSmiles.textContent = "-";
  expandedSmiles.textContent = "-";
  molfileOutput.value = "";
  graphOutput.value = "";
  renderEditors();
  renderOcrEditor();
  renderAlignmentCompare(null);
  updateMolfileSourceButtons(null);
  setMolfileSourceState("none");
  updateOcrState("No OCR", "subtle");
}

function useSelectedFile(file, sourceLabel) {
  if (!file) return;
  resetPredictionStateForNewImage();
  updatePreview(file);
  currentImageOrigin = sourceLabel === "clipboard" ? "clipboard" : sourceLabel === "example" ? "example" : "upload";
  currentImageSourceKey = sourceLabel === "example" ? "example:" + file.name : null;
  if (exampleItems.length) {
    renderExamples(exampleItems);
  }
  setStatus("Image ready from " + sourceLabel + ": " + file.name);
  logDiagnostic("info", "Prepared input image.", {
    source: sourceLabel,
    file_name: file.name,
    file_size: file.size,
    file_type: file.type || "unknown",
  });
}

function renderExamples(items) {
  if (!items.length) {
    examplesGrid.innerHTML = "<p class=\"examples-empty\">No example images found in test/.</p>";
    setExamplesState("Empty");
    return;
  }

  examplesGrid.innerHTML = items.map((item) => (
    "<button class=\"example-tile" + (currentImageSourceKey === "example:" + item.name ? " is-active" : "") + "\" type=\"button\" data-example-url=\"" + escapeXml(item.url) + "\" data-example-name=\"" + escapeXml(item.name) + "\">" +
      "<img class=\"example-thumb\" src=\"" + escapeXml(item.url) + "\" alt=\"" + escapeXml(item.name) + "\">" +
      "<span class=\"example-caption\">" + escapeXml(item.has_annotation
        ? (item.review_status === "reviewed" ? "已确认 · r" + item.revision : "导入旧标注 · 待复核")
          + " · " + item.counts.atoms + " atoms / " + item.counts.bonds + " bonds"
        : item.name) + "</span>" +
    "</button>"
  )).join("");

  examplesGrid.querySelectorAll("[data-example-url]").forEach((node) => {
    node.addEventListener("click", async () => {
      const url = node.dataset.exampleUrl;
      const name = node.dataset.exampleName;
      setStatus("Loading example image: " + name);
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error("Failed to fetch example image.");
        }
        const blob = await response.blob();
        const file = new File([blob], name, { type: blob.type || "image/png" });
        useSelectedFile(file, "example");
        currentImageSourceKey = "example:" + name;
        renderExamples(items);
        const example = items.find((item) => item.name === name);
        if (example && example.has_annotation) await openAnnotatedExample(name);
      } catch (error) {
        setStatus("Failed to load example: " + error.message);
        logDiagnostic("error", "Example image load failed.", { name: name, url: url, error: String(error) });
      }
    });
  });

  setExamplesState(String(items.length), "ok");
}

async function openAnnotatedExample(name) {
  setStatus("正在加载已保存的人工标注，请稍候…");
  const formData = new FormData();
  formData.append("filename", name);
  const response = await fetch("/api/open-example", { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Could not open saved annotation");
  if (currentImageSourceKey !== "example:" + name) return;
  currentPrediction = payload;
  editableDetections = {
    ac: normalizeHistoryItems(payload.annotation_items.ac, "atom"),
    b: normalizeHistoryItems(payload.annotation_items.b, "bond"),
  };
  // Keep imported human labels exactly as stored. Historical OCR must not
  // silently overwrite them when an example is opened.
  ocrDetections = normalizeOcrItems(payload.ocr);
  selectedItem = { editorKey: null, index: null };
  skeletonSmiles.textContent = payload.smiles.skeleton || "-";
  expandedSmiles.textContent = payload.smiles.expanded || "-";
  renderAlignmentCompare(payload);
  renderOcrEditor();
  renderEditors();
  await loadMolfileSource("preferred");
  setStatus(payload.annotation_review_status === "reviewed"
    ? "已加载最新人工确认标注 r" + payload.annotation_revision + "。修改后点击 Fix 保存新修订。"
    : "已加载转移包中的旧人工标注（尚未重新确认）。请核对并修改，点击 Fix 保存本轮确认版本。");
}

async function loadExamples() {
  setExamplesState("Loading", "warn");
  try {
    const response = await fetch("/api/examples");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load examples");
    }
    exampleItems = payload.examples || [];
    renderExamples(exampleItems);
  } catch (error) {
    examplesGrid.innerHTML = "<p class=\"examples-empty\">Failed to load examples.</p>";
    setExamplesState("Error", "error");
    logDiagnostic("error", "Example list load failed.", error);
  }
}

function buildClipboardFile(blob, sourceName = "clipboard") {
  const extension = (blob.type.split("/")[1] || "png").replace(/[^a-z0-9]/gi, "").toLowerCase() || "png";
  const name = sourceName + "_" + new Date().toISOString().replace(/[:.]/g, "-") + "." + extension;
  return new File([blob], name, { type: blob.type || "image/png" });
}

function extractPastedImageFile(event) {
  const clipboardItems = Array.from((event.clipboardData && event.clipboardData.items) || []);
  const imageItem = clipboardItems.find((item) => item.kind === "file" && item.type.startsWith("image/"));
  if (!imageItem) return null;
  const file = imageItem.getAsFile();
  if (!file) return null;
  if (file.name && file.name !== "image.png") {
    return new File([file], file.name, { type: file.type || "image/png" });
  }
  return buildClipboardFile(file);
}

function handlePaste(event, options = {}) {
  const pastedFile = extractPastedImageFile(event);
  if (!pastedFile) {
    if (options.fromPasteZone) {
      setStatus("Clipboard does not contain an image.");
    }
    return;
  }
  event.preventDefault();
  useSelectedFile(pastedFile, "clipboard");
}

async function readClipboardImage() {
  if (!navigator.clipboard || typeof navigator.clipboard.read !== "function") {
    throw new Error("This browser does not support direct clipboard image reading.");
  }

  const items = await navigator.clipboard.read();
  for (const item of items) {
    const imageType = item.types.find((type) => type.startsWith("image/"));
    if (!imageType) {
      continue;
    }
    const blob = await item.getType(imageType);
    return buildClipboardFile(blob);
  }

  throw new Error("Clipboard does not contain an image.");
}

function startKetcherLoadWatch() {
  clearTimeout(ketcherLoadTimer);
  setDiagnosticsState("Loading", "warn");
  diagnosticsHint.textContent = "Waiting for iframe load and Ketcher API initialization.";
  logDiagnostic("info", "Started loading Ketcher iframe.", { src: ketcherFrame.src });
  ketcherLoadTimer = window.setTimeout(() => {
    if (!ketcherReady) {
      setDiagnosticsState("Timeout", "error");
      diagnosticsHint.textContent = "Ketcher iframe did not expose the editor API before timeout.";
      logDiagnostic("error", "Ketcher initialization timed out.", { timeout_ms: KETCHER_LOAD_TIMEOUT_MS });
      editorState.textContent = "Editor timeout";
    }
  }, KETCHER_LOAD_TIMEOUT_MS);
}

function attachFrameDiagnostics() {
  ketcherFrame.addEventListener("load", () => {
    logDiagnostic("info", "Iframe load event fired.", { src: ketcherFrame.src });
  });

  ketcherFrame.addEventListener("error", (event) => {
    setDiagnosticsState("Load failed", "error");
    diagnosticsHint.textContent = "The iframe emitted an error before the editor initialized.";
    logDiagnostic("error", "Iframe load error event.", {
      type: event.type,
      src: ketcherFrame.src,
    });
  });
}

function simplifyKetcherDemoLayout() {
  const frameWindow = ketcherFrame.contentWindow;
  if (!frameWindow || !frameWindow.document) return;
  const frameDoc = frameWindow.document;
  const root = frameDoc.getElementById("root");
  if (!root || !root.firstElementChild) return;

  const appShell = root.firstElementChild;
  const directChildren = Array.from(appShell.children || []);
  if (directChildren.length >= 3) {
    directChildren[1].style.display = "none";
    directChildren[2].style.display = "none";
    appShell.style.display = "block";
    directChildren[0].style.width = "100%";
    directChildren[0].style.maxWidth = "100%";
    directChildren[0].style.margin = "0";
    logDiagnostic("info", "Ketcher demo side panels hidden. Editor-only layout applied.");
    return;
  }

  const textNodes = Array.from(frameDoc.querySelectorAll("div, span, h1, h2, h3, p"));
  const terminalLabel = textNodes.find((node) => node.textContent && node.textContent.trim() === "Terminal");
  if (!terminalLabel) return;

  let panel = terminalLabel;
  while (panel && panel.parentElement) {
    if (/Clear Terminal/i.test(panel.textContent || "")) {
      break;
    }
    panel = panel.parentElement;
  }

  if (!panel) return;
  panel.style.display = "none";
  panel.style.width = "0";
  panel.style.minWidth = "0";
  panel.style.maxWidth = "0";
  panel.style.padding = "0";
  panel.style.margin = "0";
  panel.style.border = "0";
  logDiagnostic("info", "Ketcher Terminal panel hidden with fallback selector.");
}

function attachKetcherWindowDiagnostics() {
  const frameWindow = ketcherFrame.contentWindow;
  if (!frameWindow || frameWindow.__i2blDiagnosticsAttached) {
    return;
  }

  const report = (level, message, detail) => {
    logDiagnostic(level, message, detail);
    if (level === "error") {
      setDiagnosticsState("Runtime error", "error");
    }
  };

  frameWindow.addEventListener("error", (event) => {
    report("error", "Ketcher window error.", {
      message: event.message,
      source: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: event.error ? String(event.error) : null,
    });
  });

  frameWindow.addEventListener("unhandledrejection", (event) => {
    report("error", "Ketcher unhandled promise rejection.", {
      reason: stringifyDetail(event.reason),
    });
  });

  frameWindow.document.addEventListener(
    "error",
    (event) => {
      const target = event.target;
      if (!target || target === frameWindow.document) {
        return;
      }
      report("error", "Ketcher resource load failure.", {
        tag: target.tagName,
        src: target.src || null,
        href: target.href || null,
      });
    },
    true,
  );

  frameWindow.__i2blDiagnosticsAttached = true;
  logDiagnostic("info", "Attached diagnostics listeners inside Ketcher iframe.");
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function createItemUid(prefix) {
  itemUidCounter += 1;
  return prefix + ":" + Date.now().toString(36) + ":" + itemUidCounter.toString(36);
}

function bboxToCenter(bbox) {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

function normalizeItems(group, source) {
  const items = [];
  const labels = (group && group.pred_classes) || [];
  labels.forEach((label, index) => {
    const bbox = deepClone(group.bbox[index]);
    items.push({
      uid: source + ":" + index,
      source: source,
      label: label,
      score: group.scores[index],
      bbox: bbox,
      center: bboxToCenter(bbox),
    });
  });
  return items;
}

function normalizeOcrItems(payload) {
  const items = (payload && payload.items) || [];
  return items.map((item, index) => ({
    uid: item.uid || ("ocr:" + index),
    detUid: item.det_uid || item.uid || ("atom:" + item.det_index),
    detIndex: Number.isFinite(Number(item.det_index)) ? Number(item.det_index) : index,
    bbox: deepClone(item.bbox || []),
    center: Array.isArray(item.center) ? deepClone(item.center) : bboxToCenter(item.bbox || [0, 0, 0, 0]),
    matchedIndex: Number.isFinite(Number(item.matched_index)) ? Number(item.matched_index) : null,
    originalText: item.original_text || "",
    originalScore: Number.isFinite(Number(item.original_score)) ? Number(item.original_score) : 0,
    scaledText: item.scaled_text || "",
    scaledScore: Number.isFinite(Number(item.scaled_score)) ? Number(item.scaled_score) : 0,
    label: item.label || "",
    score: Number.isFinite(Number(item.score)) ? Number(item.score) : 0,
  }));
}

function getOcrItemByUid(uid) {
  return ocrDetections.find((item) => item.uid === uid) || null;
}

function boxesAlmostEqual(a, b, tolerance = 0.6) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) {
    return false;
  }
  return a.every((value, index) => Math.abs(Number(value) - Number(b[index])) <= tolerance);
}

function bboxCenter(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return [0, 0];
  return [
    (Number(bbox[0]) + Number(bbox[2])) / 2,
    (Number(bbox[1]) + Number(bbox[3])) / 2,
  ];
}

function bboxIoU(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) {
    return 0;
  }
  const x1 = Math.max(Number(a[0]), Number(b[0]));
  const y1 = Math.max(Number(a[1]), Number(b[1]));
  const x2 = Math.min(Number(a[2]), Number(b[2]));
  const y2 = Math.min(Number(a[3]), Number(b[3]));
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const areaA = Math.max(0, Number(a[2]) - Number(a[0])) * Math.max(0, Number(a[3]) - Number(a[1]));
  const areaB = Math.max(0, Number(b[2]) - Number(b[0])) * Math.max(0, Number(b[3]) - Number(b[1]));
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function findAtomItemByBbox(targetBbox) {
  let exactMatch = null;
  let bestMatch = null;
  editableDetections.ac.forEach((item, index) => {
    if (item.source !== "atom") return;
    const currentBox = item.bbox;
    if (boxesAlmostEqual(currentBox, targetBbox, 0.25)) {
      if (!exactMatch) {
        exactMatch = { item: item, index: index };
      }
      return;
    }
    const iou = bboxIoU(currentBox, targetBbox);
    const centerDist = Math.hypot(
      bboxCenter(currentBox)[0] - bboxCenter(targetBbox)[0],
      bboxCenter(currentBox)[1] - bboxCenter(targetBbox)[1],
    );
    const score = iou * 100 - centerDist * 0.01;
    if (!bestMatch || score > bestMatch.score) {
      bestMatch = { item: item, index: index, score: score };
    }
  });
  return exactMatch || (bestMatch ? { item: bestMatch.item, index: bestMatch.index } : null);
}

function matchOcrDetectionsToAtoms() {
  ocrDetections.forEach((ocrItem) => {
    const matched = findAtomItemByBbox(ocrItem.bbox);
    if (matched) {
      ocrItem.matchedIndex = matched.index;
      ocrItem.detUid = matched.item.uid;
    }
  });
}

function syncOcrItemToDetections(item) {
  if (!item) return;
  const matched = findAtomItemByBbox(item.bbox);
  const detItem = matched ? matched.item : editableDetections.ac.find((candidate) => candidate.uid === item.detUid);
  if (!detItem) return;
  if (item.label) {
    detItem.label = item.label;
  }
  detItem.score = item.score;
}

function syncOcrStateToDetections() {
  ocrDetections.forEach((item) => syncOcrItemToDetections(item));
}

function updateOcrState(label, tone = "subtle") {
  ocrState.textContent = label;
  ocrState.className = "tag tag-" + tone;
}

function buildOcrPayload() {
  return {
    items: ocrDetections.map((item) => ({
      uid: item.uid,
      det_uid: item.detUid,
      det_index: item.detIndex,
      matched_index: item.matchedIndex,
      bbox: item.bbox.map((value) => Number(value.toFixed(2))),
      original_text: item.originalText,
      original_score: item.originalScore,
      scaled_text: item.scaledText,
      scaled_score: item.scaledScore,
      label: item.label,
      score: item.score,
    })),
  };
}

function rebuildDetectionPayload() {
  const atoms = editableDetections.ac.filter((item) => item.source === "atom");
  const charges = editableDetections.ac.filter((item) => item.source === "charge");
  const bonds = editableDetections.b;
  return {
    atoms: {
      bbox: atoms.map((item) => item.bbox),
      bbox_centers: atoms.map((item) => item.center),
      scores: atoms.map((item) => item.score),
      pred_classes: atoms.map((item) => item.label),
    },
    charges: {
      bbox: charges.map((item) => item.bbox),
      bbox_centers: charges.map((item) => item.center),
      scores: charges.map((item) => item.score),
      pred_classes: charges.map((item) => item.label),
    },
    bonds: {
      bbox: bonds.map((item) => item.bbox),
      bbox_centers: bonds.map((item) => item.center),
      scores: bonds.map((item) => item.score),
      pred_classes: bonds.map((item) => item.label),
    },
  };
}

function updateSelectionMeta() {
  if (!selectedItem.editorKey) {
    selectionMeta.textContent = "Nothing selected.";
    return;
  }
  const item = editableDetections[selectedItem.editorKey][selectedItem.index];
  if (!item) {
    selectionMeta.textContent = "Nothing selected.";
    return;
  }
  selectionMeta.textContent = [
    "editor: " + selectedItem.editorKey,
    "source: " + item.source,
    "label: " + item.label,
    "score: " + item.score,
    (selectedItem.editorKey === "b" ? "line endpoints: [" : "bbox: [") + item.bbox.map((v) => v.toFixed(1)).join(", ") + "]",
  ].join("\n");
}

function updateDetectionOutput() {
  detectionsOutput.value = JSON.stringify(rebuildDetectionPayload(), null, 2);
  acCount.textContent = editableDetections.ac.length + " items";
  bondCount.textContent = editableDetections.b.length + " lines";
  const total = editableDetections.ac.length + editableDetections.b.length;
  setDetectionState(total ? "Editable" : "Empty", total ? "ok" : "subtle");
  deleteDetectionButton.disabled = !selectedItem.editorKey;
  updateSelectionMeta();
}

function renderOcrEditor() {
  if (!ocrBody) {
    return;
  }

  if (!ocrDetections.length) {
    ocrBody.innerHTML = "<tr><td colspan=\"7\" class=\"ocr-empty\">No OCR results yet.</td></tr>";
    updateOcrState("No OCR", "subtle");
    return;
  }

  ocrBody.innerHTML = ocrDetections.map((item, index) => {
    const originalLabel = item.originalText || "-";
    const scaledLabel = item.scaledText || "-";
    const bbox = Array.isArray(item.bbox) ? item.bbox : [0, 0, 0, 0];
    const matchedLabel = item.matchedIndex === null || item.matchedIndex === undefined ? "-" : "ac[" + item.matchedIndex + "]";
    return (
      "<tr data-ocr-uid=\"" + escapeXml(item.uid) + "\">" +
        "<td class=\"ocr-index\">" + (index + 1) + "</td>" +
        "<td class=\"ocr-index\">" + escapeXml(matchedLabel) + "<br><span class=\"muted\">ocr " + escapeXml(item.detIndex) + "</span></td>" +
        "<td class=\"ocr-index\">[" + bbox.map((value) => Number(value || 0).toFixed(1)).join(", ") + "]</td>" +
        "<td><div class=\"ocr-text\">" + escapeXml(originalLabel) + "</div><div class=\"ocr-text muted\">score: " + Number(item.originalScore || 0).toFixed(3) + "</div></td>" +
        "<td><div class=\"ocr-text\">" + escapeXml(scaledLabel) + "</div><div class=\"ocr-text muted\">score: " + Number(item.scaledScore || 0).toFixed(3) + "</div></td>" +
        "<td><input class=\"ocr-input ocr-label-input\" data-role=\"ocr-label\" data-ocr-uid=\"" + escapeXml(item.uid) + "\" type=\"text\" value=\"" + escapeXml(item.label || "") + "\" placeholder=\"Edit label\"></td>" +
        "<td><input class=\"ocr-input ocr-score-input\" data-role=\"ocr-score\" data-ocr-uid=\"" + escapeXml(item.uid) + "\" type=\"number\" min=\"0\" max=\"1\" step=\"0.001\" value=\"" + escapeXml(Number(item.score || 0).toFixed(3)) + "\"></td>" +
      "</tr>"
    );
  }).join("");

  ocrBody.querySelectorAll("[data-role='ocr-label']").forEach((node) => {
    node.addEventListener("input", () => {
      const item = getOcrItemByUid(node.dataset.ocrUid);
      if (!item) return;
      item.label = node.value;
      syncOcrItemToDetections(item);
      updateDetectionOutput();
    });
    node.addEventListener("change", () => {
      const item = getOcrItemByUid(node.dataset.ocrUid);
      if (!item) return;
      item.label = node.value.trim();
      syncOcrItemToDetections(item);
      renderEditors();
    });
  });

  ocrBody.querySelectorAll("[data-role='ocr-score']").forEach((node) => {
    node.addEventListener("input", () => {
      const item = getOcrItemByUid(node.dataset.ocrUid);
      if (!item) return;
      const nextScore = Number(node.value);
      if (Number.isFinite(nextScore)) {
        item.score = nextScore;
        syncOcrItemToDetections(item);
      }
      updateDetectionOutput();
    });
    node.addEventListener("change", () => {
      const item = getOcrItemByUid(node.dataset.ocrUid);
      if (!item) return;
      const nextScore = Number(node.value);
      if (Number.isFinite(nextScore)) {
        item.score = Math.max(0, Math.min(1, nextScore));
        node.value = item.score.toFixed(3);
        syncOcrItemToDetections(item);
      }
      renderEditors();
    });
  });

  updateOcrState(ocrDetections.length + " items", "ok");
}

function getDefaultLabelForSource(source) {
  if (source === "atom") return "C";
  if (source === "charge") return "+";
  return "single";
}

function setPendingAddMode(mode) {
  pendingAddMode = mode;
  [
    [addAtomButton, "atom"],
    [addChargeButton, "charge"],
    [addBondButton, "bond"],
  ].forEach(([button, key]) => {
    button.classList.toggle("is-active", mode && mode.source === key);
  });
  Object.entries(editorViews).forEach(([editorKey, view]) => {
    view.svg.classList.toggle("is-add-mode", Boolean(mode && mode.editorKey === editorKey));
  });
}

function buildFixPayload(editorKey) {
  return {
    image_size: getImageSize(),
    items: editableDetections[editorKey].map((item) => ({
      uid: item.uid,
      source: item.source,
      label: item.label,
      score: item.score,
      bbox: item.bbox.map((value) => Number(value.toFixed(2))),
      center: item.center.map((value) => Number(value.toFixed(2))),
    })),
  };
}

function normalizeHistoryItems(items, defaultSource) {
  return (items || []).map((item, index) => {
    const bbox = (item.bbox || [0, 0, 0, 0]).map(Number);
    return {
      uid: item.uid || (defaultSource + ":history:" + index),
      source: item.source || defaultSource,
      label: String(item.label || ""),
      score: Number.isFinite(Number(item.score)) ? Number(item.score) : 1,
      bbox: bbox,
      center: Array.isArray(item.center) ? item.center.map(Number) : bboxToCenter(bbox),
    };
  });
}

async function loadDetectionHistory() {
  if (!currentPrediction || !currentPrediction.result_stem) {
    setStatus("Run recognition before loading image history.");
    return;
  }
  loadDetectionHistoryButton.disabled = true;
  try {
    const formData = new FormData();
    formData.append("result_stem", currentPrediction.result_stem);
    const response = await fetch("/api/load-history", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "History lookup failed");
    if (!payload.found) {
      setStatus("No saved Fix history exists for this image.");
      return;
    }
    editableDetections.ac = payload.ac.items.map((item, index) => ({
      ...normalizeHistoryItems([item], item.source || "atom")[0],
      uid: item.uid || (item.source || "atom") + ":history:" + index,
      source: item.source || "atom",
    }));
    editableDetections.b = normalizeHistoryItems(payload.b.items, "bond");
    selectedItem = { editorKey: null, index: null };
    renderEditors();
    setStatus("Loaded Fix history revision " + payload.revision + " from " + payload.source_stem + ". Click Update to rebuild Ketcher.");
    logDiagnostic("info", "Loaded saved detection history for matching image.", {
      image_sha256: payload.image_sha256,
      revision: payload.revision,
      source_stem: payload.source_stem,
      atom_charge_items: editableDetections.ac.length,
      bonds: editableDetections.b.length,
    });
  } catch (error) {
    setStatus("Failed to load detection history: " + error.message);
    logDiagnostic("error", "Detection history load failed.", error);
  } finally {
    loadDetectionHistoryButton.disabled = false;
  }
}

async function updateKetcherFromDetections() {
  if (!currentPrediction || !currentPrediction.result_stem) {
    setStatus("Run recognition before updating Ketcher from detections.");
    return;
  }
  updateDetectionsButton.disabled = true;
  setStatus("Rebuilding molecule from current atom, charge and bond edits...");
  try {
    const formData = new FormData();
    formData.append("result_stem", currentPrediction.result_stem);
    formData.append("ac_json", JSON.stringify(buildFixPayload("ac")));
    formData.append("b_json", JSON.stringify(buildFixPayload("b")));
    const response = await fetch("/api/reassemble", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Molecule rebuild failed");
    currentPrediction.molfile = payload.molfile;
    currentPrediction.graph = payload.graph;
    currentPrediction.smiles = payload.smiles;
    skeletonSmiles.textContent = payload.smiles.skeleton || "-";
    expandedSmiles.textContent = payload.smiles.expanded || "-";
    activeMolfileSource = "preferred";
    molfileOutput.value = payload.molfile.preferred || "";
    graphOutput.value = JSON.stringify(payload.graph.preferred || {}, null, 2);
    updateMolfileSourceButtons("preferred");
    setMolfileSourceState("preferred", "ok");
    await pushToKetcher(payload.molfile.preferred || "");
    setStatus("Rebuilt molecule from current detections and refreshed Ketcher.");
    logDiagnostic("info", "Updated Ketcher from current edited detections.", {
      result_stem: payload.stem,
      atoms: (payload.detection_graph.atoms || []).length,
      bonds: (payload.detection_graph.bonds || []).length,
    });
  } catch (error) {
    setStatus("Ketcher update failed: " + error.message);
    logDiagnostic("error", "Detection reassembly failed.", error);
  } finally {
    updateDetectionsButton.disabled = false;
  }
}

function selectPreferredMolfile(payload) {
  if (!payload || !payload.molfile) return "";
  return payload.molfile.preferred || payload.molfile.expanded || payload.molfile.skeleton || "";
}

function selectPreferredGraph(payload) {
  if (!payload || !payload.graph) return {};
  return payload.graph.preferred || payload.graph.expanded || payload.graph.skeleton || {};
}

function getMolfileBySource(payload, source) {
  if (!payload || !payload.molfile) return "";
  if (source === "original") return payload.molfile.skeleton || "";
  if (source === "expanded") return payload.molfile.expanded || "";
  if (source === "preferred") return selectPreferredMolfile(payload);
  return "";
}

function getGraphBySource(payload, source) {
  if (!payload || !payload.graph) return {};
  if (source === "original") return payload.graph.skeleton || {};
  if (source === "expanded") return payload.graph.expanded || {};
  if (source === "preferred") return selectPreferredGraph(payload);
  return {};
}

function updateMolfileSourceButtons(source) {
  [
    [loadOriginalButton, "original"],
    [loadExpandedButton, "expanded"],
    [loadPreferredButton, "preferred"],
  ].forEach(([button, key]) => {
    button.classList.toggle("is-active", source === key);
    button.disabled = !currentPrediction;
  });
}

async function loadMolfileSource(source) {
  if (!currentPrediction) {
    setStatus("Run recognition before switching molfile source.");
    return;
  }

  const molfile = getMolfileBySource(currentPrediction, source);
  const graph = getGraphBySource(currentPrediction, source);
  if (!molfile) {
    setStatus("No " + source + " molfile is available.");
    logDiagnostic("warn", "Requested molfile source is unavailable.", { source: source });
    return;
  }

  activeMolfileSource = source;
  molfileOutput.value = molfile;
  graphOutput.value = JSON.stringify(graph, null, 2);
  updateMolfileSourceButtons(source);
  setMolfileSourceState(source, source === "expanded" ? "warn" : "ok");
  await pushToKetcher(molfile);
  setStatus("Loaded " + source + " molfile into Ketcher.");
  logDiagnostic("info", "Loaded molfile source into Ketcher.", { source: source, length: molfile.length });
}

function getColor(item) {
  if (item.source === "atom") return "#2274a5";
  if (item.source === "charge") return "#d48a2f";
  const bondColors = {
    single: "#16856b",
    "=": "#2673c7",
    double: "#2673c7",
    "#": "#c14f45",
    triple: "#c14f45",
    ":": "#8557b5",
    aromatic: "#8557b5",
    wedge: "#d27a24",
    dash: "#008aa1",
    other: "#697875",
  };
  return bondColors[String(item.label || "").toLowerCase()] || "#16856b";
}

function getImageSize() {
  const size = currentPrediction && currentPrediction.image_size;
  if (size && size.width && size.height) {
    return size;
  }
  return { width: 1000, height: 1000 };
}

function buildBondBandPoints(endpoints, halfWidth) {
  const x1 = Number(endpoints[0]);
  const y1 = Number(endpoints[1]);
  const x2 = Number(endpoints[2]);
  const y2 = Number(endpoints[3]);
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.max(Math.hypot(dx, dy), 1e-6);
  const offsetX = (-dy / length) * halfWidth;
  const offsetY = (dx / length) * halfWidth;
  return [
    [x1 + offsetX, y1 + offsetY],
    [x2 + offsetX, y2 + offsetY],
    [x2 - offsetX, y2 - offsetY],
    [x1 - offsetX, y1 - offsetY],
  ].map((point) => point[0] + "," + point[1]).join(" ");
}

function getCoordBounds(points) {
  if (!points.length) {
    return { minX: 0, maxX: 1000, minY: 0, maxY: 1000 };
  }
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function projectCoord(point, bounds, width, height, padding) {
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX);
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY);
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;
  return {
    x: offsetX + (point.x - bounds.minX) * scale,
    y: offsetY + (bounds.maxY - point.y) * scale,
  };
}

function buildAlignmentPairs(payload) {
  if (!payload || !payload.graph || !payload.graph.skeleton || !payload.graph.preferred) {
    return [];
  }
  const originalAtoms = payload.graph.skeleton.atoms || [];
  const preferredAtoms = payload.graph.preferred.atoms || [];
  const preferredByIdx = new Map(preferredAtoms.map((atom) => [atom.idx, atom]));
  return originalAtoms
    .map((atom) => {
      const preferred = preferredByIdx.get(atom.idx);
      if (!preferred || !atom.coords || !preferred.coords) return null;
      return {
        idx: atom.idx,
        symbol: atom.atom_label || atom.symbol || "?",
        original: atom.coords,
        preferred: preferred.coords,
      };
    })
    .filter(Boolean);
}

function renderAlignmentCompare(payload) {
  const pairs = buildAlignmentPairs(payload);
  if (!pairs.length) {
    alignmentCompareSvg.innerHTML = "<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#5d6b68\" font-size=\"22\">No alignment comparison available</text>";
    alignmentMeta.textContent = "No comparison available.";
    setAlignmentState("No alignment data");
    return;
  }

  const allPoints = pairs.flatMap((pair) => [pair.original, pair.preferred]);
  const bounds = getCoordBounds(allPoints);
  const width = 1000;
  const height = 1000;
  const padding = 90;
  const layers = [];
  let totalShift = 0;
  let maxShift = 0;

  pairs.forEach((pair) => {
    const original = projectCoord(pair.original, bounds, width, height, padding);
    const preferred = projectCoord(pair.preferred, bounds, width, height, padding);
    const dx = pair.preferred.x - pair.original.x;
    const dy = pair.preferred.y - pair.original.y;
    const shift = Math.sqrt(dx * dx + dy * dy);
    totalShift += shift;
    maxShift = Math.max(maxShift, shift);

    layers.push("<line class=\"cmp-shift\" x1=\"" + original.x + "\" y1=\"" + original.y + "\" x2=\"" + preferred.x + "\" y2=\"" + preferred.y + "\"></line>");
    layers.push("<circle class=\"cmp-node-original\" cx=\"" + original.x + "\" cy=\"" + original.y + "\" r=\"7\"></circle>");
    layers.push("<circle class=\"cmp-node-preferred\" cx=\"" + preferred.x + "\" cy=\"" + preferred.y + "\" r=\"5\"></circle>");
    layers.push("<text class=\"cmp-label\" x=\"" + (preferred.x + 9) + "\" y=\"" + (preferred.y - 8) + "\">" + escapeXml(String(pair.idx) + ":" + pair.symbol) + "</text>");
  });

  alignmentCompareSvg.setAttribute("viewBox", "0 0 1000 1000");
  alignmentCompareSvg.innerHTML = layers.join("");
  const averageShift = totalShift / pairs.length;
  alignmentMeta.textContent = [
    "shared_atoms: " + pairs.length,
    "avg_shift_px: " + averageShift.toFixed(2),
    "max_shift_px: " + maxShift.toFixed(2),
    "compare: original skeleton vs preferred aligned-expanded",
  ].join("\n");
  setAlignmentState("Aligned", maxShift < 12 ? "ok" : maxShift < 28 ? "warn" : "error");
}

function renderEditor(editorKey) {
  const view = editorViews[editorKey];
  const svg = view.svg;
  const items = editableDetections[editorKey];
  const size = getImageSize();

  svg.setAttribute("viewBox", "0 0 " + size.width + " " + size.height);

  const layers = [];
  if (currentImageHref) {
    layers.push(
      "<image href=\"" + escapeXml(currentImageHref) + "\" x=\"0\" y=\"0\" width=\"" + size.width + "\" height=\"" + size.height + "\" opacity=\"0.35\" preserveAspectRatio=\"none\"></image>",
    );
  }

  if (!items.length) {
    layers.push("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#5d6b68\" font-size=\"22\">" + view.emptyLabel + "</text>");
    svg.innerHTML = layers.join("");
    attachEditorEvents(editorKey);
    return;
  }

  items.forEach((item, index) => {
    const color = getColor(item);
    const bbox = item.bbox;
    const selected = selectedItem.editorKey === editorKey && selectedItem.index === index;
    const isBond = editorKey === "b";
    const x = bbox[0];
    const y = bbox[1];
    const width = bbox[2] - bbox[0];
    const height = bbox[3] - bbox[1];

    layers.push("<g data-editor=\"" + editorKey + "\" data-index=\"" + index + "\">");
    if (isBond) {
      const midX = (bbox[0] + bbox[2]) / 2;
      const midY = (bbox[1] + bbox[3]) / 2;
      const baseHalfWidth = Math.max(3, Math.min(10, Math.max(size.width, size.height) * 0.01));
      const bandPoints = buildBondBandPoints(bbox, baseHalfWidth * (selected ? 1.25 : 1));
      const shadowPoints = buildBondBandPoints(bbox, baseHalfWidth * (selected ? 2.8 : 2.2));
      layers.push(
        "<polygon class=\"det-bond-shadow" + (selected ? " is-selected" : "") + "\" points=\"" + shadowPoints + "\" fill=\"" + color + "\"></polygon>",
      );
      layers.push(
        "<polygon class=\"det-bond-band" + (selected ? " is-selected" : "") + "\" data-role=\"bbox\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" points=\"" + bandPoints + "\" fill=\"" + color + "\" stroke=\"" + color + "\"></polygon>",
      );
      layers.push(
        "<text class=\"det-label det-bond-label\" data-role=\"label\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" x=\"" + midX + "\" y=\"" + Math.max(14, midY - 10) + "\" fill=\"" + color + "\">" + escapeXml(item.label) + "</text>",
      );
    } else {
      layers.push(
        "<rect class=\"det-rect" + (selected ? " is-selected" : "") + "\" data-role=\"bbox\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" x=\"" + x + "\" y=\"" + y + "\" width=\"" + width + "\" height=\"" + height + "\" fill=\"" + color + "\" stroke=\"" + color + "\"></rect>",
      );
      layers.push(
        "<text class=\"det-label\" data-role=\"label\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" x=\"" + (x + 4) + "\" y=\"" + Math.max(14, y - 6) + "\" fill=\"" + color + "\">" + escapeXml(item.label) + "</text>",
      );
    }

    if (selected && isBond) {
      const endpoints = [
        ["start", bbox[0], bbox[1]],
        ["end", bbox[2], bbox[3]],
      ];
      endpoints.forEach((handle) => {
        layers.push(
          "<circle class=\"det-handle det-bond-handle\" data-role=\"handle\" data-handle=\"" + handle[0] + "\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" cx=\"" + handle[1] + "\" cy=\"" + handle[2] + "\" r=\"" + (HANDLE_SIZE + 1) + "\" style=\"color:" + color + "\"></circle>",
        );
      });
    } else if (selected) {
      const handles = [
        ["nw", x, y],
        ["ne", bbox[2], y],
        ["sw", x, bbox[3]],
        ["se", bbox[2], bbox[3]],
      ];
      handles.forEach((handle) => {
        layers.push(
          "<circle class=\"det-handle handle-" + handle[0] + "\" data-role=\"handle\" data-handle=\"" + handle[0] + "\" data-editor=\"" + editorKey + "\" data-index=\"" + index + "\" cx=\"" + handle[1] + "\" cy=\"" + handle[2] + "\" r=\"" + HANDLE_SIZE + "\" style=\"color:" + color + "\"></circle>",
        );
      });
    }

    layers.push("</g>");
  });

  svg.innerHTML = layers.join("");
  attachEditorEvents(editorKey);
}

function renderEditors() {
  renderEditor("ac");
  renderEditor("b");
  updateDetectionOutput();
}

function svgPoint(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function clampBox(box) {
  const minSize = 6;
  if (box[2] - box[0] < minSize) box[2] = box[0] + minSize;
  if (box[3] - box[1] < minSize) box[3] = box[1] + minSize;
}

function updateItemGeometry(item, bbox) {
  item.bbox = bbox;
  item.center = bboxToCenter(bbox);
}

function selectItem(editorKey, index) {
  selectedItem = { editorKey: editorKey, index: index };
  setPendingAddMode(null);
  renderEditors();
}

function renameItem(editorKey, index) {
  const item = editableDetections[editorKey][index];
  if (!item) return;
  const nextLabel = window.prompt("Edit label", item.label);
  if (nextLabel === null) return;
  item.label = nextLabel.trim() || item.label;
  renderEditors();
}

function clampPointToImage(point) {
  const size = getImageSize();
  return {
    x: Math.max(0, Math.min(size.width, point.x)),
    y: Math.max(0, Math.min(size.height, point.y)),
  };
}

function buildNewBoxAtPoint(point, source) {
  const sizeHint = source === "bond" ? 64 : source === "charge" ? 36 : 48;
  const clamped = clampPointToImage(point);
  if (source === "bond") {
    const halfLength = sizeHint / 2;
    return [
      Math.max(0, clamped.x - halfLength),
      clamped.y,
      Math.min(getImageSize().width, clamped.x + halfLength),
      clamped.y,
    ];
  }
  const box = [
    clamped.x - sizeHint / 2,
    clamped.y - sizeHint / 2,
    clamped.x + sizeHint / 2,
    clamped.y + sizeHint / 2,
  ];
  const size = getImageSize();
  box[0] = Math.max(0, box[0]);
  box[1] = Math.max(0, box[1]);
  box[2] = Math.min(size.width, box[2]);
  box[3] = Math.min(size.height, box[3]);
  clampBox(box);
  return box;
}

function addDetectionItem(editorKey, source, point) {
  const label = pendingAddMode.label;
  const nextItem = {
    uid: createItemUid(editorKey + ":" + source),
    source: source,
    label: label.trim() || getDefaultLabelForSource(source),
    score: 1,
    bbox: buildNewBoxAtPoint(point, source),
    center: [point.x, point.y],
  };
  updateItemGeometry(nextItem, nextItem.bbox);
  editableDetections[editorKey].push(nextItem);
  selectedItem = { editorKey: editorKey, index: editableDetections[editorKey].length - 1 };
  if (!pendingAddMode.keepActive) setPendingAddMode(null);
  renderEditors();
  setStatus("Added new " + source + (source === "bond" ? " line." : " box.") + (pendingAddMode ? " Ctrl-click to add more; press Escape to finish." : ""));
}

function startAddMode(editorKey, source) {
  const label = window.prompt("Label for new " + source, getDefaultLabelForSource(source));
  if (label === null) return;
  setPendingAddMode({ editorKey, source, label: label.trim() || getDefaultLabelForSource(source), keepActive: false });
  setStatus("Add " + source + " mode active. Click once to add; hold Ctrl while clicking to add multiple. Escape finishes.");
}

function editSelectedLabel(source) {
  if (!selectedItem.editorKey) {
    setStatus("Select an " + source + " detection first.");
    return;
  }
  const items = editableDetections[selectedItem.editorKey];
  const item = items && items[selectedItem.index];
  if (!item || item.source !== source) {
    setStatus("Select an existing " + source + " detection first.");
    return;
  }
  const next = window.prompt("Change " + source + " label", item.label);
  if (next === null) return;
  item.label = next.trim() || item.label;
  renderEditors();
  setStatus("Updated " + source + " label. Click Fix to save this correction.");
}

function deleteSelectedItem() {
  if (!selectedItem.editorKey) {
    setStatus("Select a detection before deleting.");
    return;
  }
  const items = editableDetections[selectedItem.editorKey];
  if (!items[selectedItem.index]) {
    selectedItem = { editorKey: null, index: null };
    renderEditors();
    return;
  }
  items.splice(selectedItem.index, 1);
  selectedItem = { editorKey: null, index: null };
  renderEditors();
  setStatus("Selected detection deleted.");
}

function startDrag(editorKey, index, mode, handle, event) {
  const svg = editorViews[editorKey].svg;
  const start = svgPoint(svg, event);
  dragState = {
    editorKey: editorKey,
    index: index,
    mode: mode,
    handle: handle,
    startX: start.x,
    startY: start.y,
    startBox: deepClone(editableDetections[editorKey][index].bbox),
  };
  selectItem(editorKey, index);
  event.preventDefault();
}

function applyDrag(event) {
  if (!dragState) return;

  const editorKey = dragState.editorKey;
  const svg = editorViews[editorKey].svg;
  const point = svgPoint(svg, event);
  const dx = point.x - dragState.startX;
  const dy = point.y - dragState.startY;
  const box = deepClone(dragState.startBox);

  if (dragState.mode === "move") {
    box[0] += dx;
    box[1] += dy;
    box[2] += dx;
    box[3] += dy;
  } else if (editorKey === "b") {
    if (dragState.handle === "start") {
      box[0] += dx;
      box[1] += dy;
    } else {
      box[2] += dx;
      box[3] += dy;
    }
    const size = getImageSize();
    box[0] = Math.max(0, Math.min(size.width, box[0]));
    box[1] = Math.max(0, Math.min(size.height, box[1]));
    box[2] = Math.max(0, Math.min(size.width, box[2]));
    box[3] = Math.max(0, Math.min(size.height, box[3]));
  } else {
    if (dragState.handle.indexOf("w") !== -1) box[0] += dx;
    if (dragState.handle.indexOf("e") !== -1) box[2] += dx;
    if (dragState.handle.indexOf("n") !== -1) box[1] += dy;
    if (dragState.handle.indexOf("s") !== -1) box[3] += dy;
    if (box[0] > box[2]) [box[0], box[2]] = [box[2], box[0]];
    if (box[1] > box[3]) [box[1], box[3]] = [box[3], box[1]];
  }

  if (editorKey !== "b") {
    clampBox(box);
  }
  updateItemGeometry(editableDetections[editorKey][dragState.index], box);
  renderEditors();
}

function stopDrag() {
  dragState = null;
}

function attachEditorEvents(editorKey) {
  const svg = editorViews[editorKey].svg;
  if (!svg.dataset.boundEditorEvents) {
    svg.addEventListener("click", (event) => {
      const currentEditorKey = svg.dataset.editorKey;
      const role = event.target && event.target.dataset ? event.target.dataset.role : null;
      if (pendingAddMode && pendingAddMode.editorKey === currentEditorKey) {
        if (event.ctrlKey) {
          pendingAddMode.keepActive = true;
          event.preventDefault();
          event.stopImmediatePropagation();
          addDetectionItem(currentEditorKey, pendingAddMode.source, svgPoint(svg, event));
        } else if (role !== "bbox" && role !== "label" && role !== "handle") {
          pendingAddMode.keepActive = false;
          addDetectionItem(currentEditorKey, pendingAddMode.source, svgPoint(svg, event));
        }
        return;
      }
      if (role === "bbox" || role === "label" || role === "handle") {
        return;
      }
      selectedItem = { editorKey: null, index: null };
      renderEditors();
    });
    svg.dataset.boundEditorEvents = "true";
  }
  svg.dataset.editorKey = editorKey;
  svg.querySelectorAll("[data-role='bbox']").forEach((node) => {
    node.addEventListener("mousedown", (event) => startDrag(editorKey, Number(node.dataset.index), "move", null, event));
    node.addEventListener("click", (event) => {
      if (pendingAddMode && event.ctrlKey) return;
      selectItem(editorKey, Number(node.dataset.index));
    });
    node.addEventListener("dblclick", () => renameItem(editorKey, Number(node.dataset.index)));
  });
  svg.querySelectorAll("[data-role='label']").forEach((node) => {
    node.addEventListener("click", (event) => {
      if (pendingAddMode && event.ctrlKey) return;
      selectItem(editorKey, Number(node.dataset.index));
    });
    node.addEventListener("dblclick", () => renameItem(editorKey, Number(node.dataset.index)));
  });
  svg.querySelectorAll("[data-role='handle']").forEach((node) => {
    node.addEventListener("mousedown", (event) => startDrag(editorKey, Number(node.dataset.index), "resize", node.dataset.handle, event));
  });
}

document.addEventListener("mousemove", applyDrag);
document.addEventListener("mouseup", stopDrag);
document.addEventListener("keydown", (event) => {
  if ((event.key === "Delete" || event.key === "Backspace") && !isEditableTarget(event.target)) {
    if (selectedItem.editorKey) {
      event.preventDefault();
      deleteSelectedItem();
    }
  }
  if (event.key === "Escape" && pendingAddMode) {
    setPendingAddMode(null);
    setStatus("Add mode cancelled.");
  }
});

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  inferenceProfile = config.inference_profile || "legacy";
  if (config.inference_profile === "dual_decoder") {
    const intro = document.getElementById("appIntro");
    if (intro) intro.textContent = "Active model: the best validated shared-encoder dual-decoder bundle, with learned text boxes, bond-pair recognition, text anchors, and adapted 1×/2×/3×/4× OCR. Inspect and correct the assembled structure below.";
    document.querySelectorAll(".cutoff-grid, .nms-grid").forEach((element) => { element.hidden = true; element.style.display = "none"; });
  }
  if (!config.ketcher_available) {
    editorState.textContent = "Ketcher missing";
    ketcherFrame.hidden = true;
    editorFallback.hidden = false;
    setDiagnosticsState("Missing", "error");
    diagnosticsHint.textContent = "No Ketcher build was found for iframe mounting.";
    logDiagnostic("error", "Ketcher assets are missing.", config);
    return;
  }

  ketcherFrame.src = config.ketcher_url;
  editorState.textContent = "Loading editor";
  diagnosticsHint.textContent = "Resolved Ketcher directory: " + config.ketcher_dir;
  startKetcherLoadWatch();
  waitForKetcher();
}

async function waitForKetcher() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const frameWindow = ketcherFrame.contentWindow;
    if (frameWindow && frameWindow.ketcher && typeof frameWindow.ketcher.setMolecule === "function") {
      ketcherReady = true;
      clearTimeout(ketcherLoadTimer);
      attachKetcherWindowDiagnostics();
      simplifyKetcherDemoLayout();
      editorState.textContent = "Editor ready";
      setDiagnosticsState("Ready", "ok");
      diagnosticsHint.textContent = "Ketcher initialized successfully. Runtime errors will appear here if they occur.";
      logDiagnostic("info", "Ketcher API detected.", { methods: ["setMolecule", "getMolfile", "getKet"] });
      // An example can finish loading before the iframe. Push the currently
      // selected molecule now rather than leaving the ready editor empty.
      if (currentPrediction && currentPrediction.molfile) {
        await pushToKetcher(currentPrediction.molfile[activeMolfileSource]
          || currentPrediction.molfile.preferred || "");
      }
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
  clearTimeout(ketcherLoadTimer);
  editorState.textContent = "Editor timeout";
  setDiagnosticsState("Timeout", "error");
  diagnosticsHint.textContent = "Ketcher iframe loaded, but the API did not become ready.";
  logDiagnostic("error", "Ketcher API was not detected after polling.");
}

async function pushToKetcher(molfile) {
  if (!ketcherReady || !molfile) return;
  try {
    await ketcherFrame.contentWindow.ketcher.setMolecule(molfile);
    logDiagnostic("info", "Molfile pushed into Ketcher.", { length: molfile.length });
  } catch (error) {
    setDiagnosticsState("Set failed", "error");
    logDiagnostic("error", "setMolecule failed.", error);
    throw error;
  }
}

async function collectCurrentMolfile() {
  if (ketcherReady && ketcherFrame.contentWindow && ketcherFrame.contentWindow.ketcher) {
    try {
      const molfile = await ketcherFrame.contentWindow.ketcher.getMolfile();
      if (molfile) {
        return molfile;
      }
    } catch (error) {
      logDiagnostic("warn", "Failed to read molfile from Ketcher. Falling back to text area.", error);
    }
  }
  return molfileOutput.value || "";
}

function collectCurrentGraphJson() {
  const raw = graphOutput.value || "{}";
  try {
    return JSON.stringify(JSON.parse(raw));
  } catch (error) {
    throw new Error("Graph JSON is invalid and cannot be saved.");
  }
}

function parseCutoffValue(inputNode, fallback) {
  const value = Number(inputNode.value);
  if (Number.isFinite(value) && value >= 0 && value <= 1) {
    return value;
  }
  inputNode.value = fallback.toFixed(2);
  return fallback;
}

function parseNmsValue(inputNode, fallback) {
  const value = Number(inputNode.value);
  if (Number.isFinite(value) && value >= 0 && value <= 1) {
    return value;
  }
  inputNode.value = fallback.toFixed(2);
  return fallback;
}

async function runPrediction() {
  const file = currentImageFile;
  const atomCutoff = parseCutoffValue(atomCutoffInput, 0.5);
  const chargeCutoff = parseCutoffValue(chargeCutoffInput, 0.5);
  const bondCutoff = parseCutoffValue(bondCutoffInput, 0.4);
  const atomNms = parseNmsValue(atomNmsInput, 0.55);
  const chargeNms = parseNmsValue(chargeNmsInput, 0.55);
  if (!file) {
    setStatus("Select or paste an image first.");
    return;
  }

  runButton.disabled = true;
  setStatus(inferenceProfile === "dual_decoder"
    ? "Running the best-validation dual-decoder model with adapted multiscale OCR..."
    : "Running I2BL inference with atom " + atomCutoff.toFixed(2) + ", charge " + chargeCutoff.toFixed(2) + ", bond " + bondCutoff.toFixed(2) + ", atom NMS " + atomNms.toFixed(2) + ", charge NMS " + chargeNms.toFixed(2) + "...");
  logDiagnostic("info", "Started prediction request.", {
    inference_profile: inferenceProfile,
    file_name: file.name,
    file_size: file.size,
    atom_cutoff: atomCutoff,
    charge_cutoff: chargeCutoff,
    bond_cutoff: bondCutoff,
    atom_nms: atomNms,
    charge_nms: chargeNms,
  });

  const formData = new FormData();
  formData.append("image", file);
  formData.append("source", currentImageOrigin);
  formData.append("atom_cutoff", String(atomCutoff));
  formData.append("charge_cutoff", String(chargeCutoff));
  formData.append("bond_cutoff", String(bondCutoff));
  formData.append("atom_nms", String(atomNms));
  formData.append("charge_nms", String(chargeNms));

  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed");
    }

    currentPrediction = payload;
    editableDetections = {
      ac: normalizeItems(payload.detections.atoms, "atom").concat(normalizeItems(payload.detections.charges, "charge")),
      b: normalizeItems(payload.detections.bonds, "bond"),
    };
    ocrDetections = normalizeOcrItems(payload.ocr);
    matchOcrDetectionsToAtoms();
    syncOcrStateToDetections();
    selectedItem = { editorKey: null, index: null };

    skeletonSmiles.textContent = payload.smiles.skeleton || "-";
    expandedSmiles.textContent = payload.smiles.expanded || "-";
    renderAlignmentCompare(payload);
    renderOcrEditor();
    renderEditors();
    if (currentImageOrigin === "clipboard") {
      await loadExamples();
      currentImageSourceKey = payload.storage && payload.storage.files && payload.storage.files.pasted_example
        ? "example:" + payload.storage.files.pasted_example : null;
      renderExamples(exampleItems);
    }
    await loadMolfileSource("preferred");
    setStatus("Recognition completed. Aligned expanded molfile loaded into Ketcher.");
    logDiagnostic("info", "Prediction completed successfully.", {
      inference_profile: payload.inference_profile,
      skeleton_smiles: payload.smiles.skeleton,
      expanded_smiles: payload.smiles.expanded,
      molfile_default: "preferred",
      atom_cutoff: atomCutoff,
      charge_cutoff: chargeCutoff,
      bond_cutoff: bondCutoff,
      atom_nms: atomNms,
      charge_nms: chargeNms,
    });
  } catch (error) {
    setStatus("Failed: " + error.message);
    logDiagnostic("error", "Prediction or editor update failed.", error);
  } finally {
    runButton.disabled = false;
  }
}

async function saveFix(fixMode = "detections") {
  if (!currentPrediction || !currentPrediction.result_stem) {
    setStatus("Run recognition before saving fix data.");
    return;
  }

  const total = editableDetections.ac.length + editableDetections.b.length;
  if (!total) {
    setStatus("No detection boxes are available to save.");
    return;
  }

  fixButton.disabled = true;
  setStatus(fixMode === "detections"
    ? "Rebuilding the molecule from edited atoms, charges and bonds, then updating Ketcher..."
    : "Saving OCR corrections and current Ketcher structure...");
  logDiagnostic("info", "Started fix save request.", {
    result_stem: currentPrediction.result_stem,
    ac_items: editableDetections.ac.length,
    b_items: editableDetections.b.length,
    ocr_items: ocrDetections.length,
  });

  try {
    const molfileText = fixMode === "ocr" ? await collectCurrentMolfile() : "";
    const graphJson = fixMode === "ocr" ? collectCurrentGraphJson() : "{}";
    const formData = new FormData();
    formData.append("result_stem", currentPrediction.result_stem);
    formData.append("fix_mode", fixMode);
    formData.append("ac_json", JSON.stringify(buildFixPayload("ac")));
    formData.append("b_json", JSON.stringify(buildFixPayload("b")));
    formData.append("ocr_json", JSON.stringify(buildOcrPayload()));
    formData.append("molfile_text", fixMode === "ocr" ? molfileText : "");
    formData.append("graph_json", fixMode === "ocr" ? graphJson : "{}");

    const response = await fetch("/api/fix", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Fix save failed");
    }
    if (payload.rebuilt) {
      if (!currentPrediction.molfile) currentPrediction.molfile = {};
      if (!currentPrediction.graph) currentPrediction.graph = {};
      if (!currentPrediction.smiles) currentPrediction.smiles = {};
      currentPrediction.ocr = buildOcrPayload();
      currentPrediction.molfile.skeleton = payload.rebuilt.molfile.skeleton || currentPrediction.molfile.skeleton || "";
      currentPrediction.molfile.expanded = payload.rebuilt.molfile.expanded || currentPrediction.molfile.expanded || "";
      currentPrediction.molfile.preferred = payload.rebuilt.molfile.preferred || currentPrediction.molfile.preferred || "";
      currentPrediction.graph.skeleton = payload.rebuilt.graph.skeleton || currentPrediction.graph.skeleton || {};
      currentPrediction.graph.expanded = payload.rebuilt.graph.expanded || currentPrediction.graph.expanded || {};
      currentPrediction.graph.preferred = payload.rebuilt.graph.preferred || currentPrediction.graph.preferred || {};
      currentPrediction.smiles.skeleton = payload.rebuilt.smiles.skeleton || currentPrediction.smiles.skeleton || "-";
      currentPrediction.smiles.expanded = payload.rebuilt.smiles.expanded || currentPrediction.smiles.expanded || "-";
      skeletonSmiles.textContent = currentPrediction.smiles.skeleton || "-";
      expandedSmiles.textContent = currentPrediction.smiles.expanded || "-";
      activeMolfileSource = "preferred";
      molfileOutput.value = currentPrediction.molfile.preferred || "";
      graphOutput.value = JSON.stringify(currentPrediction.graph.preferred || {}, null, 2);
      updateMolfileSourceButtons("preferred");
      setMolfileSourceState("preferred", "ok");
      await pushToKetcher(currentPrediction.molfile.preferred || "");
    }
    setStatus((fixMode === "detections"
      ? "Rebuilt molecule from edited detections and refreshed Ketcher: "
      : "Saved OCR corrections and Ketcher structure: ") + payload.stem);
    if (payload.training_pair) {
      setStatus("人工标注已保存 · r" + payload.training_pair.revision + " · " + payload.saved_at_utc
        + "。下次打开同一图片将加载此最新版本。");
    }
    await loadExamples();
    logDiagnostic("info", "Fix data saved.", payload);
  } catch (error) {
    setStatus("Fix save failed: " + error.message);
    logDiagnostic("error", "Fix save failed.", error);
  } finally {
    fixButton.disabled = false;
  }
}

updateDetectionOutput();
renderOcrEditor();
updateMolfileSourceButtons(null);
setMolfileSourceState("none");
setAlignmentState("No alignment data");
imageInput.addEventListener("change", () => useSelectedFile(imageInput.files[0], "file picker"));
addAtomButton.addEventListener("click", () => {
  startAddMode("ac", "atom");
});
addChargeButton.addEventListener("click", () => {
  startAddMode("ac", "charge");
});
addBondButton.addEventListener("click", () => {
  startAddMode("b", "bond");
});
editAtomButton.addEventListener("click", () => editSelectedLabel("atom"));
editChargeButton.addEventListener("click", () => editSelectedLabel("charge"));
editBondButton.addEventListener("click", () => editSelectedLabel("bond"));
deleteDetectionButton.addEventListener("click", deleteSelectedItem);
pasteButton.addEventListener("click", async () => {
  focusPasteZone(true);
  setStatus("Reading clipboard image...");
  try {
    const file = await readClipboardImage();
    useSelectedFile(file, "clipboard");
    pasteInput.blur();
  } catch (error) {
    pasteInput.focus();
    setStatus("Direct clipboard read failed. Press Ctrl+V or Cmd+V. " + error.message);
    logDiagnostic("warn", "Direct clipboard image read failed.", error);
  }
});
pasteInput.addEventListener("focus", () => focusPasteZone(true));
pasteInput.addEventListener("blur", () => focusPasteZone(false));
pasteInput.addEventListener("input", () => {
  pasteInput.value = "";
});
pasteInput.addEventListener("paste", (event) => handlePaste(event, { fromPasteZone: true }));
document.addEventListener("paste", (event) => {
  if (event.defaultPrevented) return;
  if (event.target === pasteInput || event.target === pasteButton || pasteButton.contains(event.target)) {
    handlePaste(event, { fromPasteZone: true });
    return;
  }
  if (isEditableTarget(event.target)) {
    return;
  }
  handlePaste(event, { fromPasteZone: false });
});
runButton.addEventListener("click", runPrediction);
fixButton.addEventListener("click", () => saveFix("detections"));
ocrFixButton.addEventListener("click", () => saveFix("ocr"));
loadDetectionHistoryButton.addEventListener("click", loadDetectionHistory);
updateDetectionsButton.addEventListener("click", updateKetcherFromDetections);
loadOriginalButton.addEventListener("click", () => loadMolfileSource("original"));
loadExpandedButton.addEventListener("click", () => loadMolfileSource("expanded"));
loadPreferredButton.addEventListener("click", () => loadMolfileSource("preferred"));
attachFrameDiagnostics();
loadConfig();
loadExamples();
