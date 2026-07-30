/* ═══════════════════════════════════════════════════════════
   AI Character Detector — Application Logic
   ═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── Configuration ─────────────────────────────────────────
  const CONFIG = {
    maxFileSizeMB: 10,
    allowedExtensions: new Set(['jpg', 'jpeg', 'png', 'webp', 'bmp']),
    allowedMimeTypes: new Set([
      'image/jpeg', 'image/png', 'image/webp', 'image/bmp',
    ]),
    endpoints: {
      health:  '/health',
      predict: '/predict',
      gradcam: '/predict/gradcam',
    },
    classDisplayNames: {
      human:        'Real Human',
      cartoon:      'Cartoon / Anime',
      ai_generated: 'AI Generated',
    },
    classOrder: ['human', 'cartoon', 'ai_generated'],
    classCSSKeys: {
      human:        'human',
      cartoon:      'cartoon',
      ai_generated: 'ai',
    },
  };

  // ── DOM references ────────────────────────────────────────
  const dom = {
    // Header
    statusDot:         document.getElementById('statusDot'),
    statusText:        document.getElementById('statusText'),

    // Upload
    uploadZone:        document.getElementById('uploadZone'),
    fileInput:         document.getElementById('fileInput'),
    uploadPrompt:      document.getElementById('uploadPrompt'),
    previewContainer:  document.getElementById('previewContainer'),
    previewImage:      document.getElementById('previewImage'),
    previewName:       document.getElementById('previewName'),
    previewDetails:    document.getElementById('previewDetails'),
    btnRemove:         document.getElementById('btnRemove'),
    validationError:   document.getElementById('validationError'),
    validationMsg:     document.getElementById('validationMsg'),

    // Options
    gradcamCheckbox:   document.getElementById('gradcamCheckbox'),

    // Action
    btnAnalyze:        document.getElementById('btnAnalyze'),
    serverError:       document.getElementById('serverError'),
    serverErrorMsg:    document.getElementById('serverErrorMsg'),

    // Results
    resultsEmpty:      document.getElementById('resultsEmpty'),
    resultsContent:    document.getElementById('resultsContent'),
    predictionBadge:   document.getElementById('predictionBadge'),
    predictionIcon:    document.getElementById('predictionIcon'),
    predictionValue:   document.getElementById('predictionValue'),
    predictionConf:    document.getElementById('predictionConf'),
    probList:          document.getElementById('probList'),
    metaInference:     document.getElementById('metaInference'),
    metaTimestamp:      document.getElementById('metaTimestamp'),

    // Grad-CAM
    gradcamSection:    document.getElementById('gradcamSection'),
    gradcamOriginal:   document.getElementById('gradcamOriginal'),
    gradcamHeatmap:    document.getElementById('gradcamHeatmap'),
    gradcamDownload:   document.getElementById('gradcamDownload'),
    gradcamError:      document.getElementById('gradcamError'),

    // Live regions
    liveStatus:        document.getElementById('liveStatus'),
  };

  // ── State ─────────────────────────────────────────────────
  let state = {
    selectedFile: null,
    previewObjectURL: null,
    isAnalyzing: false,
    gradcamBlobURL: null,
  };

  // ── Utilities ─────────────────────────────────────────────
  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function getFileExtension(filename) {
    const parts = filename.split('.');
    return parts.length > 1 ? parts.pop().toLowerCase() : '';
  }

  function escapeHTML(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function announceToScreenReader(message) {
    if (dom.liveStatus) {
      dom.liveStatus.textContent = message;
    }
  }

  // ── API helper ────────────────────────────────────────────
  async function apiRequest(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (err) {
      throw new Error('Network error. Check that the server is running.');
    }

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        if (body.detail) {
          detail = typeof body.detail === 'string'
            ? body.detail
            : JSON.stringify(body.detail);
        }
      } catch (_) { /* use statusText */ }

      if (response.status === 422) {
        throw new Error('Validation error: ' + detail);
      }
      if (response.status === 503) {
        throw new Error('Model not available. ' + detail);
      }
      if (response.status === 500) {
        throw new Error('Server error. Please try again later.');
      }
      throw new Error(detail);
    }

    return response.json();
  }

  // ── Health check ──────────────────────────────────────────
  async function checkHealth() {
    dom.statusDot.setAttribute('data-status', 'checking');
    dom.statusText.textContent = 'Checking\u2026';

    try {
      const data = await apiRequest(CONFIG.endpoints.health);
      if (data.status === 'ok' && data.model_ready) {
        dom.statusDot.setAttribute('data-status', 'ready');
        dom.statusText.textContent = 'System ready';
      } else if (data.status === 'ok' && !data.model_ready) {
        dom.statusDot.setAttribute('data-status', 'error');
        dom.statusText.textContent = 'Model not loaded';
      } else {
        dom.statusDot.setAttribute('data-status', 'error');
        dom.statusText.textContent = 'Unavailable';
      }
    } catch (_) {
      dom.statusDot.setAttribute('data-status', 'error');
      dom.statusText.textContent = 'Unavailable';
    }
  }

  // ── File validation ───────────────────────────────────────
  function validateFile(file) {
    if (!file) {
      return 'No file selected.';
    }

    if (file.size === 0) {
      return 'The selected file is empty.';
    }

    const ext = getFileExtension(file.name);
    if (!CONFIG.allowedExtensions.has(ext)) {
      return 'Unsupported file type ".' + ext + '". Accepted formats: JPG, PNG, WebP, BMP.';
    }

    if (file.type && !CONFIG.allowedMimeTypes.has(file.type)) {
      // Some browsers may not report MIME types for BMP
      if (ext !== 'bmp') {
        return 'Unsupported file type. Accepted formats: JPG, PNG, WebP, BMP.';
      }
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > CONFIG.maxFileSizeMB) {
      return 'File too large (' + sizeMB.toFixed(1) + ' MB). Maximum: ' + CONFIG.maxFileSizeMB + ' MB.';
    }

    return null;
  }

  // ── UI state management ───────────────────────────────────
  function showValidationError(msg) {
    dom.validationMsg.textContent = msg;
    dom.validationError.classList.add('visible');
    announceToScreenReader('Error: ' + msg);
  }

  function hideValidationError() {
    dom.validationError.classList.remove('visible');
  }

  function showServerError(msg) {
    dom.serverErrorMsg.textContent = msg;
    dom.serverError.classList.add('visible');
    announceToScreenReader('Error: ' + msg);
  }

  function hideServerError() {
    dom.serverError.classList.remove('visible');
  }

  function clearPreview() {
    if (state.previewObjectURL) {
      URL.revokeObjectURL(state.previewObjectURL);
      state.previewObjectURL = null;
    }
    state.selectedFile = null;

    dom.uploadZone.classList.remove('has-file');
    dom.previewContainer.classList.remove('visible');
    dom.previewImage.removeAttribute('src');
    dom.previewImage.removeAttribute('alt');
    dom.previewName.textContent = '';
    dom.previewDetails.textContent = '';

    dom.btnAnalyze.disabled = true;
    dom.fileInput.value = '';

    hideValidationError();
    hideServerError();
  }

  function showPreview(file) {
    if (state.previewObjectURL) {
      URL.revokeObjectURL(state.previewObjectURL);
    }

    state.selectedFile = file;
    state.previewObjectURL = URL.createObjectURL(file);

    dom.previewImage.src = state.previewObjectURL;
    dom.previewImage.alt = 'Preview of ' + file.name;
    dom.previewName.textContent = file.name;

    const ext = getFileExtension(file.name).toUpperCase();
    dom.previewDetails.textContent = ext + ' \u00B7 ' + formatFileSize(file.size);

    dom.uploadZone.classList.add('has-file');
    dom.previewContainer.classList.add('visible');

    dom.btnAnalyze.disabled = false;
    hideValidationError();
    hideServerError();
  }

  function resetResults() {
    dom.resultsEmpty.style.display = '';
    dom.resultsContent.classList.remove('visible');
    clearGradCAM();
  }

  function clearGradCAM() {
    dom.gradcamSection.classList.remove('visible');
    dom.gradcamOriginal.removeAttribute('src');
    dom.gradcamHeatmap.removeAttribute('src');
    dom.gradcamError.style.display = 'none';
    dom.gradcamDownload.style.display = 'none';

    if (state.gradcamBlobURL) {
      URL.revokeObjectURL(state.gradcamBlobURL);
      state.gradcamBlobURL = null;
    }
  }

  function setLoading(loading) {
    state.isAnalyzing = loading;
    dom.btnAnalyze.classList.toggle('loading', loading);
    dom.btnAnalyze.disabled = loading;
    if (loading) {
      dom.btnAnalyze.setAttribute('aria-busy', 'true');
    } else {
      dom.btnAnalyze.removeAttribute('aria-busy');
      if (state.selectedFile) {
        dom.btnAnalyze.disabled = false;
      }
    }
  }

  // ── Class SVG icons ───────────────────────────────────────
  function getClassIcon(classKey) {
    switch (classKey) {
      case 'human':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
      case 'cartoon':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>';
      case 'ai_generated':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M1 9h3"/><path d="M1 15h3"/><path d="M20 9h3"/><path d="M20 15h3"/></svg>';
      default:
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    }
  }

  // ── Render results ────────────────────────────────────────
  function renderResults(data) {
    dom.resultsEmpty.style.display = 'none';

    // Prediction badge
    const cssKey = CONFIG.classCSSKeys[data.class_key] || '';
    dom.predictionBadge.className = 'prediction-badge ' + cssKey;
    dom.predictionIcon.innerHTML = getClassIcon(data.class_key);

    // Prediction text
    const displayName = CONFIG.classDisplayNames[data.class_key] || data.prediction;
    dom.predictionValue.textContent = displayName;
    dom.predictionValue.className = 'prediction-value ' + cssKey;

    // Confidence
    const confPct = (data.confidence * 100).toFixed(1);
    dom.predictionConf.textContent = confPct + '% confidence';

    // Probability bars
    dom.probList.innerHTML = '';
    CONFIG.classOrder.forEach(function (key) {
      var prob = (data.probabilities && data.probabilities[key]) || 0;
      var pct = (prob * 100).toFixed(1);
      var fillCSSKey = CONFIG.classCSSKeys[key] || '';
      var isPredicted = key === data.class_key;

      var row = document.createElement('div');
      row.className = 'prob-row';

      var nameSpan = document.createElement('span');
      nameSpan.className = 'prob-name' + (isPredicted ? ' is-predicted' : '');
      nameSpan.textContent = CONFIG.classDisplayNames[key] || key;

      var track = document.createElement('div');
      track.className = 'prob-track';

      var fill = document.createElement('div');
      fill.className = 'prob-fill ' + fillCSSKey + (isPredicted ? '' : ' is-muted');
      fill.setAttribute('data-width', (prob * 100).toString());

      track.appendChild(fill);

      var pctSpan = document.createElement('span');
      pctSpan.className = 'prob-pct' + (isPredicted ? ' is-predicted' : '');
      pctSpan.textContent = pct + '%';

      row.appendChild(nameSpan);
      row.appendChild(track);
      row.appendChild(pctSpan);
      dom.probList.appendChild(row);
    });

    // Animate bars
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        var fills = dom.probList.querySelectorAll('.prob-fill');
        fills.forEach(function (el) {
          el.style.width = el.getAttribute('data-width') + '%';
        });
      });
    });

    // Metadata
    if (data.inference_ms != null) {
      dom.metaInference.textContent = data.inference_ms.toFixed(0) + ' ms';
    } else {
      dom.metaInference.textContent = '\u2014';
    }

    var now = new Date();
    dom.metaTimestamp.textContent = now.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });

    dom.resultsContent.classList.add('visible');

    announceToScreenReader(
      'Analysis complete. Predicted category: ' + displayName +
      ' with ' + confPct + ' percent confidence.'
    );
  }

  // ── Render Grad-CAM ───────────────────────────────────────
  function renderGradCAM(data) {
    clearGradCAM();

    if (!data.gradcam_base64) {
      dom.gradcamSection.classList.add('visible');
      dom.gradcamError.style.display = 'block';
      dom.gradcamError.textContent = 'Grad-CAM visualization was not returned by the server.';
      return;
    }

    // Show original uploaded image
    if (state.previewObjectURL) {
      dom.gradcamOriginal.src = state.previewObjectURL;
      dom.gradcamOriginal.alt = 'Original uploaded image';
    }

    // Show heatmap
    var heatmapSrc = 'data:image/png;base64,' + data.gradcam_base64;
    dom.gradcamHeatmap.src = heatmapSrc;
    dom.gradcamHeatmap.alt = 'Grad-CAM attention heatmap';

    // Create downloadable blob
    try {
      var byteString = atob(data.gradcam_base64);
      var bytes = new Uint8Array(byteString.length);
      for (var i = 0; i < byteString.length; i++) {
        bytes[i] = byteString.charCodeAt(i);
      }
      var blob = new Blob([bytes], { type: 'image/png' });
      state.gradcamBlobURL = URL.createObjectURL(blob);
      dom.gradcamDownload.href = state.gradcamBlobURL;
      dom.gradcamDownload.download = 'gradcam_result.png';
      dom.gradcamDownload.style.display = 'inline-flex';
    } catch (_) {
      dom.gradcamDownload.style.display = 'none';
    }

    dom.gradcamSection.classList.add('visible');
  }

  // ── File handling ─────────────────────────────────────────
  function handleFile(file) {
    hideServerError();
    resetResults();

    var error = validateFile(file);
    if (error) {
      clearPreview();
      showValidationError(error);
      return;
    }

    hideValidationError();
    showPreview(file);
  }

  // ── Event: file input ─────────────────────────────────────
  dom.fileInput.addEventListener('change', function () {
    if (dom.fileInput.files && dom.fileInput.files[0]) {
      handleFile(dom.fileInput.files[0]);
    }
  });

  // ── Event: drag and drop ──────────────────────────────────
  dom.uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    e.stopPropagation();
    dom.uploadZone.classList.add('drag-over');
  });

  dom.uploadZone.addEventListener('dragleave', function (e) {
    e.preventDefault();
    e.stopPropagation();
    dom.uploadZone.classList.remove('drag-over');
  });

  dom.uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    e.stopPropagation();
    dom.uploadZone.classList.remove('drag-over');

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // ── Event: remove file ────────────────────────────────────
  dom.btnRemove.addEventListener('click', function (e) {
    e.preventDefault();
    e.stopPropagation();
    clearPreview();
    resetResults();
  });

  // Keyboard support for remove button
  dom.btnRemove.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      clearPreview();
      resetResults();
    }
  });

  // ── Event: analyze ────────────────────────────────────────
  dom.btnAnalyze.addEventListener('click', function () {
    submitAnalysis();
  });

  // Keyboard support
  dom.btnAnalyze.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      submitAnalysis();
    }
  });

  async function submitAnalysis() {
    if (!state.selectedFile || state.isAnalyzing) return;

    setLoading(true);
    hideServerError();
    resetResults();
    announceToScreenReader('Analyzing image\u2026');

    var useGradcam = dom.gradcamCheckbox.checked;
    var endpoint = useGradcam
      ? CONFIG.endpoints.gradcam
      : CONFIG.endpoints.predict;

    var formData = new FormData();
    formData.append('file', state.selectedFile);

    try {
      var data = await apiRequest(endpoint, {
        method: 'POST',
        body: formData,
      });

      renderResults(data);

      if (useGradcam) {
        renderGradCAM(data);
      }
    } catch (err) {
      showServerError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  }

  // ── Initialization ────────────────────────────────────────
  checkHealth();

  // Re-check health every 30 seconds
  setInterval(checkHealth, 30000);

})();
