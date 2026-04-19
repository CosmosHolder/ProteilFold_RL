/* ============================================================
   ProteinFold-RL — Shared Chart Helpers
   Requires Chart.js loaded before this file.
   Usage: called by dashboard.html
   ============================================================ */

/* ── Design tokens (match design-system.css) ── */
const PF = {
  blue:     '#3d9fff',
  teal:     '#00d4aa',
  red:      '#ff5f5f',
  amber:    '#ffb347',
  bg2:      '#040f1f',
  bg3:      '#061428',
  border:   'rgba(61,159,255,0.10)',
  textDim:  'rgba(180,210,240,0.45)',
  textMid:  'rgba(180,210,240,0.65)',
  mono:     "'Space Mono', monospace",
};

/* ── Global Chart.js defaults ── */
function applyChartDefaults() {
  Chart.defaults.color            = PF.textDim;
  Chart.defaults.font.family      = PF.mono;
  Chart.defaults.font.size        = 10;
  Chart.defaults.borderColor      = PF.border;
  Chart.defaults.backgroundColor  = 'transparent';
}

/* ── Smooth rolling average ── */
function rollingAvg(arr, window = 20) {
  return arr.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = arr.slice(start, i + 1);
    const avg   = slice.reduce((a, b) => a + b, 0) / slice.length;
    return parseFloat(avg.toFixed(3));
  });
}

/* ── Synthetic fallback ─────────────────────────────────────
   Used when assets/data/training_log.json is not yet present
   (i.e. training hasn't run). Mirrors the real curve shape.
   ──────────────────────────────────────────────────────── */
function _syntheticData() {
  const episodes = Array.from({ length: 500 }, (_, i) => i + 1);

  const energy = episodes.map(ep => {
    const base  = 267 - (267 - 149) * (1 - Math.exp(-ep / 120));
    const noise = (Math.random() - 0.5) * 28;
    return parseFloat(Math.max(140, base + noise).toFixed(2));
  });

  const rmsd = episodes.map(ep => {
    const base  = 7.6 - (7.6 - 1.8) * (1 - Math.exp(-ep / 140));
    const noise = (Math.random() - 0.5) * 1.4;
    return parseFloat(Math.max(1.1, base + noise).toFixed(3));
  });

  const reward = episodes.map(ep => {
    const base  = -18 + 22 * (1 - Math.exp(-ep / 100));
    const noise = (Math.random() - 0.5) * 6;
    return parseFloat((base + noise).toFixed(2));
  });

  const policyLoss = episodes.map(ep => {
    const base  = 0.45 * Math.exp(-ep / 180) + 0.04;
    const noise = (Math.random() - 0.5) * 0.04;
    return parseFloat(Math.max(0.01, base + noise).toFixed(4));
  });

  const valueLoss = episodes.map(ep => {
    const base  = 0.38 * Math.exp(-ep / 160) + 0.03;
    const noise = (Math.random() - 0.5) * 0.03;
    return parseFloat(Math.max(0.01, base + noise).toFixed(4));
  });

  const entropy = episodes.map(ep => {
    const base  = 3.8 * Math.exp(-ep / 200) + 1.2;
    const noise = (Math.random() - 0.5) * 0.3;
    return parseFloat(Math.max(0.5, base + noise).toFixed(4));
  });

  return { episodes, energy, rmsd, reward, policyLoss, valueLoss, entropy, source: 'synthetic' };
}

/* ── Parse real training_log.json into the same shape ───────
   Expected JSON format (array of episode objects):
   [
     {
       "episode": 1,
       "protein": "1L2Y",
       "stage": 1,
       "total_reward": -12.4,
       "final_energy": 241.3,
       "rmsd": 6.82,
       "steps": 50,
       "clashes": 2,
       "policy_loss": 0.312,
       "value_loss": 0.281,
       "entropy": 3.61
     },
     ...
   ]
   ──────────────────────────────────────────────────────── */
function _parseLogJson(rows) {
  const episodes   = rows.map(r => r.episode);
  const energy     = rows.map(r => parseFloat(r.final_energy));
  const rmsd       = rows.map(r => parseFloat(r.rmsd));
  const reward     = rows.map(r => parseFloat(r.total_reward));
  const policyLoss = rows.map(r => parseFloat(r.policy_loss  || 0));
  const valueLoss  = rows.map(r => parseFloat(r.value_loss   || 0));
  const entropy    = rows.map(r => parseFloat(r.entropy      || 0));
  return { episodes, energy, rmsd, reward, policyLoss, valueLoss, entropy, source: 'real' };
}

/* ── Main data loader ────────────────────────────────────────
   Fetches GET http://localhost:8000/training-log-json.
   Falls back to synthetic data if the endpoint is unreachable,
   returns an empty array, or returns malformed JSON —
   so the dashboard always renders.

   Returns a Promise that resolves to the data object.
   dashboard.html must call this as:

     loadTrainingData().then(data => {
       buildEnergyChart('chart-energy', data);
       buildRmsdChart('chart-rmsd', data);
       buildRewardChart('chart-reward', data);
       buildLossChart('chart-loss', data);
       // update data-source badge if present
       const badge = document.getElementById('data-source-badge');
       if (badge) badge.textContent = data.source === 'real'
         ? 'Real training data'
         : 'Synthetic data (training pending)';
     });
   ──────────────────────────────────────────────────────── */
async function loadTrainingData() {
  try {
    const res = await fetch('http://localhost:8000/training-log-json');

    // Server error or endpoint not found
    if (!res.ok) {
      console.warn('[charts.js] /training-log-json not found — using synthetic data');
      return _syntheticData();
    }

    const text = await res.text();

    // Empty response
    if (!text || text.trim() === '' || text.trim() === 'null') {
      console.warn('[charts.js] /training-log-json is empty — using synthetic data');
      return _syntheticData();
    }

    const rows = JSON.parse(text);

    // Not an array or zero rows
    if (!Array.isArray(rows) || rows.length === 0) {
      console.warn('[charts.js] /training-log-json has no rows — using synthetic data');
      return _syntheticData();
    }

    console.info(`[charts.js] Loaded ${rows.length} real episodes from /training-log-json`);
    return _parseLogJson(rows);

  } catch (err) {
    // JSON parse error or network error
    console.warn('[charts.js] Failed to fetch /training-log-json:', err.message, '— using synthetic data');
    return _syntheticData();
  }
}

/* ── Build Energy Chart ── */
function buildEnergyChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const smoothed = rollingAvg(data.energy, 20);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.episodes,
      datasets: [
        {
          label: 'Raw energy',
          data: data.energy,
          borderColor: 'rgba(61,159,255,0.18)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Smoothed (avg 20)',
          data: smoothed,
          borderColor: PF.blue,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: {
            target: 'origin',
            above: 'rgba(61,159,255,0.04)',
          },
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: PF.textDim,
            font: { family: PF.mono, size: 10 },
            boxWidth: 12,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#061428',
          borderColor: PF.border,
          borderWidth: 0.5,
          titleColor: PF.textMid,
          bodyColor: PF.textDim,
          titleFont: { family: PF.mono, size: 10 },
          bodyFont:  { family: PF.mono, size: 10 },
          callbacks: {
            title: items => `Episode ${items[0].label}`,
            label: item  => ` ${item.dataset.label}: ${item.parsed.y.toFixed(1)} kcal/mol`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            maxTicksLimit: 10,
            callback: v => `Ep ${v}`,
          },
        },
        y: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            callback: v => `${v} kcal`,
          },
        },
      },
    },
  });
}

/* ── Build RMSD Chart ── */
function buildRmsdChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const smoothed = rollingAvg(data.rmsd, 20);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.episodes,
      datasets: [
        {
          label: 'Raw RMSD',
          data: data.rmsd,
          borderColor: 'rgba(0,212,170,0.18)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Smoothed (avg 20)',
          data: smoothed,
          borderColor: PF.teal,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: {
            target: 'origin',
            above: 'rgba(0,212,170,0.04)',
          },
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1400, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: PF.textDim,
            font: { family: PF.mono, size: 10 },
            boxWidth: 12,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#061428',
          borderColor: PF.border,
          borderWidth: 0.5,
          titleColor: PF.textMid,
          bodyColor: PF.textDim,
          titleFont: { family: PF.mono, size: 10 },
          bodyFont:  { family: PF.mono, size: 10 },
          callbacks: {
            title: items => `Episode ${items[0].label}`,
            label: item  => ` ${item.dataset.label}: ${item.parsed.y.toFixed(3)} Å`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            maxTicksLimit: 10,
            callback: v => `Ep ${v}`,
          },
        },
        y: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            callback: v => `${v} Å`,
          },
        },
      },
    },
  });
}

/* ── Build Reward Chart ── */
function buildRewardChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const smoothed = rollingAvg(data.reward, 20);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.episodes,
      datasets: [
        {
          label: 'Raw reward',
          data: data.reward,
          borderColor: 'rgba(255,179,71,0.18)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Smoothed (avg 20)',
          data: smoothed,
          borderColor: PF.amber,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1600, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: PF.textDim,
            font: { family: PF.mono, size: 10 },
            boxWidth: 12,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#061428',
          borderColor: PF.border,
          borderWidth: 0.5,
          titleColor: PF.textMid,
          bodyColor: PF.textDim,
          titleFont: { family: PF.mono, size: 10 },
          bodyFont:  { family: PF.mono, size: 10 },
          callbacks: {
            title: items => `Episode ${items[0].label}`,
            label: item  => ` ${item.dataset.label}: ${item.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            maxTicksLimit: 10,
            callback: v => `Ep ${v}`,
          },
        },
        y: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
          },
        },
      },
    },
  });
}

/* ── Build Loss Chart (policy + value) ── */
function buildLossChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.episodes,
      datasets: [
        {
          label: 'Policy loss',
          data: data.policyLoss,
          borderColor: PF.blue,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.4,
        },
        {
          label: 'Value loss',
          data: data.valueLoss,
          borderColor: PF.red,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1800, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: PF.textDim,
            font: { family: PF.mono, size: 10 },
            boxWidth: 12,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#061428',
          borderColor: PF.border,
          borderWidth: 0.5,
          titleColor: PF.textMid,
          bodyColor: PF.textDim,
          titleFont: { family: PF.mono, size: 10 },
          bodyFont:  { family: PF.mono, size: 10 },
          callbacks: {
            title: items => `Episode ${items[0].label}`,
            label: item  => ` ${item.dataset.label}: ${item.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            maxTicksLimit: 10,
            callback: v => `Ep ${v}`,
          },
        },
        y: {
          grid:  { color: PF.border },
          ticks: {
            color: PF.textDim,
            font: { family: PF.mono, size: 9 },
            callback: v => v.toFixed(3),
          },
        },
      },
    },
  });
}