/**
 * app.js — Main application controller
 * Wires sliders ↔ number inputs, handles run/reset/sweep actions,
 * updates results panel and canvases.
 */

// ── Parameter defaults & bounds ───────────────────────────────────────────
const PARAMS = {
  c:     { default: 20,  min: 5,   max: 40,  step: 0.5, unit: 'kPa' },
  phi:   { default: 30,  min: 20,  max: 40,  step: 0.5, unit: '°'   },
  E:     { default: 50,  min: 10,  max: 100, step: 1,   unit: 'MPa' },
  gamma: { default: 18,  min: 16,  max: 22,  step: 0.1, unit: 'kN/m³'},
  beta:  { default: 40,  min: 30,  max: 50,  step: 0.5, unit: '°'   },
  H:     { default: 15,  min: 10,  max: 30,  step: 0.5, unit: 'm'   },
  ru:    { default: 0.0, min: 0,   max: 0.45,step: 0.01,unit: 'u/γh'},
  kh:    { default: 0.0, min: 0,   max: 0.30,step: 0.01,unit: 'g'   },
};

const SWEEP_RANGES = {
  c:     { min: 5,   max: 40,  label: 'Cohesion c (kPa)',       n: 28 },
  phi:   { min: 20,  max: 40,  label: 'Friction Angle φ (°)',   n: 24 },
  E:     { min: 10,  max: 100, label: "Young's Modulus E (MPa)", n: 30 },
  gamma: { min: 16,  max: 22,  label: 'Unit Weight γ (kN/m³)',  n: 20 },
  beta:  { min: 30,  max: 50,  label: 'Slope Angle β (°)',      n: 24 },
  H:     { min: 10,  max: 30,  label: 'Slope Height H (m)',     n: 24 },
  ru:    { min: 0,   max: 0.45,label: 'Pore Pressure Ratio ru', n: 20 },
  kh:    { min: 0,   max: 0.30,label: 'Seismic Coeff kh (g)',  n: 20 },
};

// ── Initialise ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  bindSliders();
  updateSliderColors();
  initHeroCanvas();
  drawHeroPreview();
  bindButtons();
  bindSweep();
  bindNav();
});

// ── Slider ↔ Number input sync ────────────────────────────────────────────
function bindSliders() {
  Object.keys(PARAMS).forEach(id => {
    const slider = document.getElementById(id);
    const numIn  = document.getElementById(id + '-num');
    if (!slider || !numIn) return;

    slider.addEventListener('input', () => {
      numIn.value = slider.value;
      updateSliderColors();
      schedulePreview();
    });
    numIn.addEventListener('input', () => {
      const v = parseFloat(numIn.value);
      const p = PARAMS[id];
      if (!isNaN(v) && v >= p.min && v <= p.max) {
        slider.value = v;
        updateSliderColors();
        schedulePreview();
      }
    });
  });
}

function updateSliderColors() {
  Object.keys(PARAMS).forEach(id => {
    const slider = document.getElementById(id);
    if (!slider) return;
    const p   = PARAMS[id];
    const pct = (slider.value - p.min) / (p.max - p.min) * 100;
    slider.style.background =
      `linear-gradient(to right, #f0a500 ${pct}%, #353c54 ${pct}%)`;
  });
}

function getValues() {
  return {
    c:     parseFloat(document.getElementById('c').value),
    phi:   parseFloat(document.getElementById('phi').value),
    E:     parseFloat(document.getElementById('E').value),
    gamma: parseFloat(document.getElementById('gamma').value),
    beta:  parseFloat(document.getElementById('beta').value),
    H:     parseFloat(document.getElementById('H').value),
    ru:    parseFloat(document.getElementById('ru') ? document.getElementById('ru').value : 0),
    kh:    parseFloat(document.getElementById('kh') ? document.getElementById('kh').value : 0),
  };
}

// ── Live slope preview while adjusting ───────────────────────────────────
let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {}, 200);
}

// ── Buttons ───────────────────────────────────────────────────────────────
function bindButtons() {
  document.getElementById('runBtn').addEventListener('click', runSimulation);
  document.getElementById('resetBtn').addEventListener('click', resetParams);
}

function runSimulation() {
  const btn     = document.getElementById('runBtn');
  const label   = btn.querySelector('.run-label');
  const spinner = document.getElementById('spinner');
  label.style.opacity = '0';
  spinner.style.display = 'inline-block';
  btn.disabled = true;

  setTimeout(() => {
    const p = getValues();
    const t0 = performance.now();
    const result = runLEM(p.c, p.phi, p.E, p.gamma, p.beta, p.H, p.ru, p.kh);
    const elapsed = (performance.now() - t0).toFixed(1);

    label.style.opacity = '1';
    spinner.style.display = 'none';
    btn.disabled = false;

    if (!result.converged) {
      showError('Simulation did not converge. Try different parameters.');
      return;
    }
    renderResults(result, p, elapsed);
  }, 80);
}

function resetParams() {
  Object.keys(PARAMS).forEach(id => {
    const slider = document.getElementById(id);
    const numIn  = document.getElementById(id + '-num');
    if (slider) slider.value = PARAMS[id].default;
    if (numIn)  numIn.value  = PARAMS[id].default;
  });
  updateSliderColors();
  document.getElementById('resultsPlaceholder').style.display = '';
  document.getElementById('resultsContent').style.display = 'none';
}

// ── Render results ────────────────────────────────────────────────────────
function renderResults(result, params, elapsed) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  const content = document.getElementById('resultsContent');
  content.style.display = '';

  // Bishop gauge
  setGauge('bishop', result.FoS_bishop);
  setGauge('janbu',  result.FoS_janbu);

  // Displacement
  document.getElementById('umax').textContent =
    result.u_max_mm !== null ? result.u_max_mm.toFixed(1) : '—';

  // Slip circle canvas
  drawSlipCircle(result, params);

  // Parameter chips
  const chips = [
    ['c', params.c, 'kPa'],
    ['φ', params.phi, '°'],
    ['E', params.E, 'MPa'],
    ['γ', params.gamma, 'kN/m³'],
    ['β', params.beta, '°'],
    ['H', params.H, 'm'],
  ];
  document.getElementById('paramSummary').innerHTML = chips
    .map(([sym, val, unit]) =>
      `<div class="param-chip"><span>${sym}</span> = ${val} ${unit}</div>`)
    .join('');

  // Animate in
  content.querySelectorAll('.gauge-card, .disp-card, .canvas-wrap').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(12px)';
    el.style.transition = `opacity .4s ease ${i * 0.08}s, transform .4s ease ${i * 0.08}s`;
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    });
  });
}

function setGauge(name, fos) {
  const card   = document.getElementById(name + 'Card');
  const valEl  = document.getElementById('fos' + name);
  const bar    = document.getElementById(name + 'Bar');
  const status = document.getElementById(name + 'Status');

  if (!card) return;

  const cls = statusClass(fos);
  card.className = 'gauge-card ' + cls;
  valEl.textContent = fos ? fos.toFixed(3) : '—';
  status.textContent = statusLabel(fos);

  // Bar: map FoS 0→3+ to 0→100%
  const pct = fos ? Math.min(100, (fos / 3) * 100) : 0;
  const color = statusColor(fos);
  setTimeout(() => {
    bar.style.width = pct + '%';
    bar.style.background = color;
    bar.style.boxShadow = `0 0 8px ${color}80`;
  }, 100);
}

function showError(msg) {
  document.getElementById('resultsPlaceholder').style.display = '';
  document.getElementById('resultsPlaceholder').innerHTML =
    `<div class="placeholder-icon" style="color:#f87171">⚠</div>
     <div class="placeholder-text" style="color:#f87171">${msg}</div>`;
  document.getElementById('resultsContent').style.display = 'none';
}

// ── Sensitivity sweep ─────────────────────────────────────────────────────
function bindSweep() {
  document.getElementById('sweepBtn').addEventListener('click', runSweep);
}

function runSweep() {
  const paramId = document.getElementById('sweepParam').value;
  const base    = getValues();
  const range   = SWEEP_RANGES[paramId];
  const btn     = document.getElementById('sweepBtn');

  btn.disabled = true;
  btn.textContent = 'Computing…';

  // Async to let UI update
  setTimeout(() => {
    const xVals = [], yBishop = [], yJanbu = [];
    const n = range.n;

    for (let i = 0; i <= n; i++) {
      const v = range.min + (range.max - range.min) * i / n;
      const p = { ...base, [paramId]: v };
      const res = runLEM(p.c, p.phi, p.E, p.gamma, p.beta, p.H);
      xVals.push(+v.toFixed(3));
      yBishop.push(res.FoS_bishop || null);
      yJanbu.push(res.FoS_janbu   || null);
    }

    document.getElementById('chartPlaceholder').style.display = 'none';
    drawSweepChart(xVals, yBishop, yJanbu, range.label);

    btn.disabled = false;
    btn.textContent = 'Run Sweep';
  }, 20);
}

// ── Smooth nav scroll ─────────────────────────────────────────────────────
function bindNav() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // Scroll spy
  const sections = ['simulator', 'theory', 'about'];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.nav-link').forEach(l => {
          l.classList.toggle('active',
            l.getAttribute('href') === '#' + entry.target.id);
        });
      }
    });
  }, { threshold: 0.4 });
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}
