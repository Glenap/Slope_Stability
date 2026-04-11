/**
 * chart.js — Minimal line chart renderer (pure Canvas, no dependencies)
 * Used for the one-parameter sensitivity sweep.
 */

/**
 * Draw a line chart on #sweepChart canvas.
 * @param {number[]} xValues  - x axis data
 * @param {number[]} yBishop  - FoS Bishop values
 * @param {number[]} yJanbu   - FoS Janbu values
 * @param {string}   xLabel   - label for x axis
 */
function drawSweepChart(xValues, yBishop, yJanbu, xLabel) {
  const canvas = document.getElementById('sweepChart');
  if (!canvas) return;

  // Proper DPR scaling
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width, H = rect.height;
  const ml = 56, mr = 24, mt = 24, mb = 48;
  const pw = W - ml - mr, ph = H - mt - mb;

  // Ranges
  const allY = [...yBishop, ...yJanbu].filter(Boolean);
  const yMin = Math.max(0, Math.min(...allY) - 0.1);
  const yMax = Math.max(...allY) + 0.1;
  const xMin = Math.min(...xValues), xMax = Math.max(...xValues);

  function sx(v) { return ml + (v - xMin) / (xMax - xMin) * pw; }
  function sy(v) { return mt + ph - (v - yMin) / (yMax - yMin) * ph; }

  // Background
  ctx.fillStyle = '#141720';
  ctx.fillRect(0, 0, W, H);

  // FoS threshold bands
  const bands = [
    { y1: 1.5, y2: yMax, color: 'rgba(74,222,128,0.04)', label: null },
    { y1: 1.0, y2: 1.5,  color: 'rgba(250,204,21,0.05)', label: null },
    { y1: yMin, y2: 1.0, color: 'rgba(248,113,113,0.07)', label: null },
  ];
  bands.forEach(b => {
    const y1c = Math.max(sy(b.y2), mt);
    const y2c = Math.min(sy(b.y1), mt + ph);
    if (y2c > y1c) {
      ctx.fillStyle = b.color;
      ctx.fillRect(ml, y1c, pw, y2c - y1c);
    }
  });

  // Grid lines
  ctx.strokeStyle = 'rgba(42,47,66,0.8)';
  ctx.lineWidth = 0.5;
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const yv = yMin + (yMax - yMin) * i / yTicks;
    const yc = sy(yv);
    ctx.beginPath(); ctx.moveTo(ml, yc); ctx.lineTo(ml + pw, yc); ctx.stroke();
    ctx.font = '9px Space Mono, monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(2), ml - 6, yc + 3);
  }
  const xTicks = Math.min(8, xValues.length);
  for (let i = 0; i <= xTicks; i++) {
    const xv = xMin + (xMax - xMin) * i / xTicks;
    const xc = sx(xv);
    ctx.beginPath(); ctx.moveTo(xc, mt); ctx.lineTo(xc, mt + ph); ctx.stroke();
    ctx.font = '9px Space Mono, monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText(xv.toFixed(1), xc, mt + ph + 16);
  }

  // Threshold lines
  [{ v: 1.5, c: '#4ade80', label: 'FoS=1.5' }, { v: 1.0, c: '#f87171', label: 'FoS=1.0' }]
    .forEach(t => {
      if (t.v >= yMin && t.v <= yMax) {
        ctx.beginPath();
        ctx.moveTo(ml, sy(t.v)); ctx.lineTo(ml + pw, sy(t.v));
        ctx.strokeStyle = t.c + '60';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '9px Space Mono, monospace';
        ctx.fillStyle = t.c + 'aa';
        ctx.textAlign = 'left';
        ctx.fillText(t.label, ml + 4, sy(t.v) - 4);
      }
    });

  // Axis borders
  ctx.strokeStyle = '#2a2f42';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph);
  ctx.stroke();

  // Axis labels
  ctx.font = '10px Space Mono, monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, ml + pw / 2, H - 6);
  ctx.save();
  ctx.translate(14, mt + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Factor of Safety', 0, 0);
  ctx.restore();

  // Draw line helper
  function drawLine(ys, color, dash = []) {
    const pts = xValues.map((x, i) => [sx(x), sy(ys[i])]).filter(([, y]) => isFinite(y));
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(...pts[i]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dots
    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawLine(yBishop, '#f0a500');
  drawLine(yJanbu,  '#2dd4bf', [5, 3]);

  // Legend
  const lx = ml + pw - 160, ly = mt + 12;
  ctx.fillStyle = 'rgba(20,23,32,0.9)';
  ctx.fillRect(lx - 4, ly - 2, 164, 36);
  ctx.strokeStyle = '#2a2f42';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(lx - 4, ly - 2, 164, 36);

  ctx.fillStyle = '#f0a500';
  ctx.fillRect(lx, ly + 4, 18, 2);
  ctx.font = '9px Space Mono, monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  ctx.fillText('Bishop', lx + 24, ly + 8);

  ctx.strokeStyle = '#2dd4bf';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(lx, ly + 20); ctx.lineTo(lx + 18, ly + 20); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('Janbu', lx + 24, ly + 24);
}
