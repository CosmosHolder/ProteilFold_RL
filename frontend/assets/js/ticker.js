/* ============================================================
   ProteinFold-RL — Shared Ticker
   Usage: call initTicker() after DOM loads.
   ============================================================ */

function initTicker() {
  const track = document.getElementById('ticker-track');
  if (!track) return;

  const items = [
    { label: 'BEST RMSD',          value: '15.920 Å',           cls: 'v-teal'  },
    { label: 'ENERGY IMPROVEMENT', value: '+62.8 kcal/mol',     cls: 'v-blue'  },
    { label: 'EPISODES TRAINED',   value: '500',                cls: 'v-blue'  },
    { label: 'RMSD REDUCTION',     value: '−5.1% vs random',    cls: 'v-teal'  },
    { label: 'ARCHITECTURE',       value: 'GNN + PPO-CLIP',     cls: 'v-blue'  },
    { label: 'PROTEIN TARGET',     value: 'TRP-CAGE 1L2Y',      cls: 'v-amber' },
    { label: 'ACTION SPACE',       value: 'N × 2 × 12 DISCRETE',cls: 'v-blue'  },
    { label: 'POLICY LAYERS',      value: '4-LAYER MPNN → 256D',cls: 'v-blue'  },
    { label: 'REWARD SIGNAL',      value: 'PHYSICS-BASED',      cls: 'v-teal'  },
    { label: 'STATUS',             value: 'AGENT BEATS RANDOM ✓',cls: 'v-teal' },
    { label: 'RANDOM RMSD',        value: '16.775 Å',           cls: 'v-amber' },
    { label: 'TRAINED RMSD',       value: '15.920 Å',           cls: 'v-teal'  },
    { label: 'RANDOM ENERGY',      value: '117.258 kcal/mol',   cls: 'v-amber' },
    { label: 'TRAINED ENERGY',     value: '54.430 kcal/mol',    cls: 'v-teal'  },
    { label: 'CHECKPOINT',         value: '07 COMPLETE',        cls: 'v-blue'  },
  ];

  const html = items.map(i =>
    `<div class="ticker-item">
      ${i.label} &nbsp;
      <span class="${i.cls}">${i.value}</span>
    </div>`
  ).join('');

  /* Duplicate so the scroll loops seamlessly */
  track.innerHTML = html + html;
}