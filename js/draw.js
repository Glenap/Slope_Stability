/**
 * draw.js — Canvas rendering for slope geometry and critical slip circle
 */

// ── Colour palette (matches CSS variables) ────────────────────────────────
const C = {
  bg:       '#141720',
  bg2:      '#1c2030',
  border:   '#2a2f42',
  amber:    '#f0a500',
  teal:     '#2dd4bf',
  red:      '#f87171',
  green:    '#4ade80',
  yellow:   '#facc15',
  soil:     '#a07850',
  soilDark: '#7a5c38',
  sky:      '#0d0f14',
  grid:     'rgba(42,47,66,0.6)',
  slipLine: '#f87171',
  slipFill: 'rgba(248,113,113,0.08)',
  centre:   '#f0a500',
  water:    'rgba(45,212,191,0.15)',
};

// ── Hero preview (animated slope sketch) ─────────────────────────────────

function drawHeroPreview() {
  const canvas = document.getElementById('slopePreview');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  let t = 0;

  function frame() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Gentle background grid
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += 30) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 30) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Parameters for preview
    const sH = 120, beta = 40, HR = sH / Math.tan(beta * Math.PI / 180);
    const ox = 60, oy = H - 40;
    const tx = ox, ty = oy;
    const cx2 = ox + HR, cy2 = oy - sH;

    // Sky gradient
    const skyGrad = ctx.createLinearGradient(0, 0, 0, oy - sH);
    skyGrad.addColorStop(0, '#0a0c10');
    skyGrad.addColorStop(1, '#141720');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, W, oy - sH);

    // Soil body
    ctx.beginPath();
    ctx.moveTo(0, oy);
    ctx.lineTo(tx, ty);
    ctx.lineTo(cx2, cy2);
    ctx.lineTo(W, cy2);
    ctx.lineTo(W, oy);
    ctx.closePath();
    const soilGrad = ctx.createLinearGradient(0, cy2, 0, oy);
    soilGrad.addColorStop(0, '#6b4c2a');
    soilGrad.addColorStop(1, '#3d2a16');
    ctx.fillStyle = soilGrad;
    ctx.fill();

    // Slope surface line
    ctx.beginPath();
    ctx.moveTo(0, oy); ctx.lineTo(tx, ty); ctx.lineTo(cx2, cy2); ctx.lineTo(W, cy2);
    ctx.strokeStyle = C.amber + '80';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Animated slip circle
    const phase = t * 0.015;
    const scx = ox + HR * 0.45 + Math.sin(phase) * 8;
    const scy = cy2 - sH * 0.6 + Math.cos(phase * 0.7) * 5;
    const sR  = sH * 1.35;

    // Filled arc (failure mass highlight)
    ctx.save();
    ctx.beginPath();
    ctx.arc(scx, scy, sR, 0, Math.PI * 2);
    ctx.closePath();
    ctx.clip();
    ctx.fillStyle = C.slipFill;
    ctx.fillRect(0, 0, W, H);
    ctx.restore();

    // Circle arc
    ctx.beginPath();
    ctx.arc(scx, scy, sR, 0, Math.PI * 2);
    ctx.strokeStyle = C.slipLine;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Centre mark
    ctx.beginPath();
    ctx.arc(scx, scy, 4, 0, Math.PI * 2);
    ctx.fillStyle = C.centre;
    ctx.fill();

    // Slices visualization
    const sliceX = Math.max(scx - sR, tx);
    const sliceXEnd = Math.min(scx + sR, cx2 + (W - cx2) * 0.5);
    const nSlices = 12;
    for (let i = 0; i < nSlices; i++) {
      const xm = sliceX + (sliceXEnd - sliceX) * (i + 0.5) / nSlices;
      const dArc = sR * sR - (xm - scx) * (xm - scx);
      if (dArc < 0) continue;
      const arcY = scy - Math.sqrt(dArc);
      let surfY;
      if (xm <= tx)       surfY = oy;
      else if (xm >= cx2) surfY = cy2;
      else surfY = oy - (xm - tx) / HR * sH;
      if (arcY > surfY) continue;
      ctx.beginPath();
      ctx.moveTo(xm, arcY);
      ctx.lineTo(xm, surfY);
      ctx.strokeStyle = 'rgba(240,165,0,0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Dimension annotation
    ctx.beginPath();
    ctx.moveTo(cx2 + 20, cy2);
    ctx.lineTo(cx2 + 20, oy);
    ctx.strokeStyle = C.teal + '80';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = '10px Space Mono, monospace';
    ctx.fillStyle = C.teal;
    ctx.fillText('H', cx2 + 25, (cy2 + oy) / 2 + 4);

    // FoS badge
    const fosVal = (1.42 + Math.sin(phase * 0.5) * 0.03).toFixed(3);
    ctx.fillStyle = C.bg2;
    ctx.fillRect(W - 100, 12, 86, 30);
    ctx.strokeStyle = C.amber + '60';
    ctx.lineWidth = 1;
    ctx.strokeRect(W - 100, 12, 86, 30);
    ctx.font = '9px Space Mono, monospace';
    ctx.fillStyle = C.amber;
    ctx.fillText('FoS (Bishop)', W - 95, 24);
    ctx.font = 'bold 12px Space Mono, monospace';
    ctx.fillText(fosVal, W - 72, 36);

    t++;
    requestAnimationFrame(frame);
  }
  frame();
}

// ── Result slip circle drawing ────────────────────────────────────────────

function drawSlipCircle(result, params) {
  const canvas = document.getElementById('slipCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const CW = canvas.width, CH = canvas.height;

  const { geo, cx, cy, R, slices, FoS_bishop } = result;
  const { H, betaDeg, toeX, crestX } = geo;

  // Compute world bounding box
  const pad = H * 0.35;
  const wxMin = toeX - pad;
  const wxMax = crestX + pad * 1.5;
  const wyMin = -H * 0.25;
  const wyMax = cy + R * 0.25 + H * 0.15;

  function toCanvas(wx, wy) {
    const margin = 24;
    const sx = margin + (wx - wxMin) / (wxMax - wxMin) * (CW - 2 * margin);
    const sy = (CH - margin) - (wy - wyMin) / (wyMax - wyMin) * (CH - 2 * margin);
    return [sx, sy];
  }

  ctx.clearRect(0, 0, CW, CH);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, CW, CH);

  // Grid
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 0.5;
  for (let wx = Math.ceil(wxMin); wx <= wxMax; wx += Math.max(1, Math.round(H / 5))) {
    const [sx] = toCanvas(wx, 0);
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, CH); ctx.stroke();
  }

  // Soil body
  const pts = [
    [wxMin, wyMin],
    [toeX,  0],
    [crestX, H],
    [wxMax,  H],
    [wxMax,  wyMin],
  ];
  ctx.beginPath();
  pts.forEach(([wx, wy], i) => {
    const [sx, sy] = toCanvas(wx, wy);
    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
  });
  ctx.closePath();
  const sg = ctx.createLinearGradient(0, 0, 0, CH);
  sg.addColorStop(0, '#5a3e22');
  sg.addColorStop(1, '#2e1c0e');
  ctx.fillStyle = sg;
  ctx.fill();

  // Slope surface
  const surfPts = [[wxMin, 0], [toeX, 0], [crestX, H], [wxMax, H]];
  ctx.beginPath();
  surfPts.forEach(([wx, wy], i) => {
    const [sx, sy] = toCanvas(wx, wy);
    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
  });
  ctx.strokeStyle = C.amber;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Slices
  if (slices) {
    slices.forEach(s => {
      const [bx, by] = toCanvas(s.xm, s.baseY);
      const [tx2, ty2] = toCanvas(s.xm, s.surfY);
      ctx.beginPath();
      ctx.moveTo(bx, by); ctx.lineTo(tx2, ty2);
      ctx.strokeStyle = 'rgba(240,165,0,0.2)';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });
  }

  // Failure mass highlight
  if (slices && slices.length > 0) {
    ctx.beginPath();
    const first = slices[0], last = slices[slices.length - 1];
    const [fx, fy] = toCanvas(first.xm, first.surfY);
    ctx.moveTo(fx, fy);
    slices.forEach(s => {
      const [sx, sy] = toCanvas(s.xm, s.surfY);
      ctx.lineTo(sx, sy);
    });
    const [lx, ly] = toCanvas(last.xm, last.baseY);
    ctx.lineTo(lx, ly);
    slices.slice().reverse().forEach(s => {
      const [sx, sy] = toCanvas(s.xm, s.baseY);
      ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(248,113,113,0.1)';
    ctx.fill();
  }

  // Slip circle
  const [ccx, ccy] = toCanvas(cx, cy);
  const [edgeX] = toCanvas(cx + R, cy);
  const rPx = Math.abs(edgeX - ccx);
  ctx.beginPath();
  ctx.arc(ccx, ccy, rPx, 0, Math.PI * 2);
  ctx.strokeStyle = C.slipLine;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Centre
  ctx.beginPath();
  ctx.arc(ccx, ccy, 5, 0, Math.PI * 2);
  ctx.fillStyle = C.centre;
  ctx.fill();

  // Labels
  ctx.font = '10px Space Mono, monospace';
  ctx.fillStyle = C.amber + 'cc';
  ctx.fillText(`H = ${params.H} m`, ...toCanvas(toeX + 2, H * 0.5));

  // FoS badge
  ctx.fillStyle = C.bg2;
  roundRect(ctx, CW - 120, 10, 108, 34, 4);
  ctx.fill();
  ctx.strokeStyle = statusColor(FoS_bishop) + '80';
  ctx.lineWidth = 1;
  roundRect(ctx, CW - 120, 10, 108, 34, 4);
  ctx.stroke();
  ctx.font = '9px Space Mono, monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('FoS (Bishop)', CW - 113, 22);
  ctx.font = 'bold 14px Space Mono, monospace';
  ctx.fillStyle = statusColor(FoS_bishop);
  ctx.fillText(FoS_bishop ? FoS_bishop.toFixed(3) : '—', CW - 88, 36);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function statusColor(fos) {
  if (!fos)       return '#94a3b8';
  if (fos >= 1.5) return C.green;
  if (fos >= 1.0) return C.yellow;
  return C.red;
}

function statusLabel(fos) {
  if (!fos)       return '—';
  if (fos >= 1.5) return 'STABLE';
  if (fos >= 1.25) return 'MARGINAL';
  if (fos >= 1.0) return 'CRITICAL';
  return 'FAILURE';
}

function statusClass(fos) {
  if (!fos) return '';
  if (fos >= 1.5) return 'stable';
  if (fos >= 1.0) return 'marginal';
  return 'critical';
}

// ── Hero canvas (particle field) ──────────────────────────────────────────

function initHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const N = 60;
  const dots = Array.from({ length: N }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: Math.random() * 1.5 + 0.5,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    opacity: Math.random() * 0.4 + 0.1,
  }));

  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dots.forEach(d => {
      d.x += d.vx; d.y += d.vy;
      if (d.x < 0) d.x = canvas.width;
      if (d.x > canvas.width) d.x = 0;
      if (d.y < 0) d.y = canvas.height;
      if (d.y > canvas.height) d.y = 0;

      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240,165,0,${d.opacity})`;
      ctx.fill();
    });

    // Connections
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        const dx = dots[i].x - dots[j].x;
        const dy = dots[i].y - dots[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y);
          ctx.lineTo(dots[j].x, dots[j].y);
          ctx.strokeStyle = `rgba(240,165,0,${0.06 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(frame);
  }
  frame();
}
