/**
 * lem.js — Limit Equilibrium Method solver (JavaScript port)
 * Bishop's Simplified + Janbu's Simplified
 * Exact same mathematics as the Python backend (01_lem_solver.py)
 */

// ── Geometry ──────────────────────────────────────────────────────────────

function buildSlopeGeometry(H, betaDeg) {
  const betaRad = betaDeg * Math.PI / 180;
  const horizontalRun = H / Math.tan(betaRad);
  return {
    toeX: 0, toeY: 0,
    crestX: horizontalRun, crestY: H,
    H, betaRad, betaDeg, horizontalRun
  };
}

function slopeSurfaceY(x, geo) {
  if (x <= geo.toeX)   return 0;
  if (x >= geo.crestX) return geo.H;
  return ((x - geo.toeX) / (geo.crestX - geo.toeX)) * geo.H;
}

// ── Slice discretisation ──────────────────────────────────────────────────

function discretizeCircle(cx, cy, R, geo, nSlices = 30) {
  const disc = R * R - cy * cy;
  if (disc <= 0) return null;
  let xLeft  = cx - Math.sqrt(disc);
  let xRight = cx + Math.sqrt(disc);
  xLeft  = Math.max(xLeft,  geo.toeX);
  xRight = Math.min(xRight, geo.crestX + geo.H / Math.tan(geo.betaRad) * 0.5);
  if (xRight - xLeft < 0.01 * geo.H) return null;

  const slices = [];
  const step = (xRight - xLeft) / nSlices;

  for (let i = 0; i < nSlices; i++) {
    const xl = xLeft + i * step;
    const xr = xl + step;
    const xm = (xl + xr) / 2;
    const w  = xr - xl;
    const dBase = R * R - (xm - cx) * (xm - cx);
    if (dBase < 0) continue;
    const baseY  = cy - Math.sqrt(dBase);
    const surfY  = slopeSurfaceY(xm, geo);
    const hSoil  = surfY - baseY;
    if (hSoil <= 0) continue;
    let sinAlpha = (xm - cx) / R;
    sinAlpha = Math.max(-0.98, Math.min(0.98, sinAlpha));
    const alpha = Math.asin(sinAlpha);
    slices.push({ xm, width: w, alpha, hSoil, baseY, surfY });
  }
  return slices.length >= 3 ? slices : null;
}

// ── Bishop's Simplified ───────────────────────────────────────────────────

function bishopFoS(slices, c, phiDeg, gamma) {
  const phi = phiDeg * Math.PI / 180;
  const tanPhi = Math.tan(phi);
  const Ws     = slices.map(s => s.hSoil * s.width * gamma);
  const alphas = slices.map(s => s.alpha);
  const bs     = slices.map(s => s.width);

  const sumDriving = Ws.reduce((acc, W, i) => acc + W * Math.sin(alphas[i]), 0);
  if (sumDriving <= 0) return null;

  let fos = 1.5;
  for (let iter = 0; iter < 200; iter++) {
    let num = 0;
    for (let i = 0; i < slices.length; i++) {
      let mAlpha = Math.cos(alphas[i]) + Math.sin(alphas[i]) * tanPhi / fos;
      if (Math.abs(mAlpha) < 1e-6) mAlpha = 1e-6;
      num += (c * bs[i] + Ws[i] * tanPhi) / mAlpha;
    }
    const fosNew = num / sumDriving;
    if (Math.abs(fosNew - fos) < 1e-5) { fos = fosNew; break; }
    fos = 0.5 * fos + 0.5 * fosNew;
  }
  return (fos > 0.2 && fos < 10) ? +fos.toFixed(4) : null;
}

// ── Janbu's Simplified ────────────────────────────────────────────────────

function janbuFoS(slices, c, phiDeg, gamma) {
  const phi = phiDeg * Math.PI / 180;
  const tanPhi = Math.tan(phi);
  const Ws     = slices.map(s => s.hSoil * s.width * gamma);
  const alphas = slices.map(s => s.alpha);
  const bs     = slices.map(s => s.width);

  const sumDriving = Ws.reduce((acc, W, i) => acc + W * Math.tan(alphas[i]), 0);
  if (Math.abs(sumDriving) < 1e-6) return null;

  let fos = 1.5;
  for (let iter = 0; iter < 200; iter++) {
    let num = 0;
    for (let i = 0; i < slices.length; i++) {
      let nAlpha = Math.cos(alphas[i]) ** 2 * (1 + Math.tan(alphas[i]) * tanPhi / fos);
      if (Math.abs(nAlpha) < 1e-6) nAlpha = 1e-6;
      num += (c * bs[i] + Ws[i] * tanPhi) / nAlpha;
    }
    const fosNew = num / sumDriving;
    if (Math.abs(fosNew - fos) < 1e-5) { fos = fosNew; break; }
    fos = 0.5 * fos + 0.5 * fosNew;
  }
  const hAvg = slices.reduce((a, s) => a + s.hSoil, 0) / slices.length;
  const xVals = slices.map(s => s.xm);
  const L = Math.max(...xVals) - Math.min(...xVals) || 1;
  const f0 = 1 + 0.031 * (hAvg / L);
  const fc = fos * f0;
  return (fc > 0.2 && fc < 10) ? +fc.toFixed(4) : null;
}

// ── Critical slip circle search ───────────────────────────────────────────

function findCriticalFoS(c, phi, gamma, H, beta, nGrid = 12, nRandom = 150) {
  const geo = buildSlopeGeometry(H, beta);
  const cxMin = geo.toeX - 0.5 * H;
  const cxMax = geo.crestX + 0.5 * H;
  const cyMin = H * 0.5;
  const cyMax = H * 3.0;

  let bestFoS = Infinity;
  let bestCx = cxMin, bestCy = cyMin, bestR = H * 1.5;
  let bestSlices = null;

  // Grid search
  for (let gi = 0; gi < nGrid; gi++) {
    const cx = cxMin + (cxMax - cxMin) * gi / (nGrid - 1);
    for (let gj = 0; gj < nGrid; gj++) {
      const cy = cyMin + (cyMax - cyMin) * gj / (nGrid - 1);
      const R0 = Math.sqrt((cx - geo.toeX) ** 2 + cy * cy);
      for (let rk = 0; rk < 4; rk++) {
        const R = R0 * (0.95 + rk * 0.15);
        const slices = discretizeCircle(cx, cy, R, geo, 30);
        if (!slices) continue;
        const fb = bishopFoS(slices, c, phi, gamma);
        if (fb && fb < bestFoS) {
          bestFoS = fb; bestCx = cx; bestCy = cy; bestR = R; bestSlices = slices;
        }
      }
    }
  }

  // Random refinement
  const spread = H * 0.4;
  const rng = mulberry32(42);
  for (let k = 0; k < nRandom; k++) {
    const cx = bestCx + (rng() - 0.5) * 2 * spread;
    const cy = Math.max(bestCy + (rng() - 0.5) * spread, H * 0.3);
    const R  = Math.max(
      Math.sqrt((cx - geo.toeX) ** 2 + cy * cy) * (0.9 + rng() * 0.3),
      H * 0.8
    );
    const slices = discretizeCircle(cx, cy, R, geo, 30);
    if (!slices) continue;
    const fb = bishopFoS(slices, c, phi, gamma);
    if (fb && fb < bestFoS) {
      bestFoS = fb; bestCx = cx; bestCy = cy; bestR = R; bestSlices = slices;
    }
  }

  const fosJ = bestSlices ? janbuFoS(bestSlices, c, phi, gamma) : null;
  return {
    FoS_bishop: bestFoS === Infinity ? null : bestFoS,
    FoS_janbu:  fosJ,
    cx: bestCx, cy: bestCy, R: bestR,
    geo, slices: bestSlices
  };
}

// ── Displacement estimate ─────────────────────────────────────────────────

function estimateDisplacement(FoS, H, E_MPa, gamma) {
  const E_kPa = E_MPa * 1000;
  const u_base = (gamma * H * H) / E_kPa * 1000; // mm
  let amp;
  if (FoS <= 1.0)      amp = 10;
  else if (FoS < 1.1)  amp = 5 + (1.1 - FoS) / 0.1 * 5;
  else if (FoS < 2.0)  amp = 1 + 4 / Math.pow(FoS - 0.9, 1.2);
  else                 amp = 1;
  return +(u_base * amp).toFixed(3);
}

// ── Seeded PRNG (Mulberry32) ──────────────────────────────────────────────
function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ── Main entry ────────────────────────────────────────────────────────────

function runLEM(c, phi, E, gamma, beta, H) {
  const result = findCriticalFoS(c, phi, gamma, H, beta, 12, 150);
  const u = result.FoS_bishop
    ? estimateDisplacement(result.FoS_bishop, H, E, gamma)
    : null;
  return {
    FoS_bishop: result.FoS_bishop,
    FoS_janbu:  result.FoS_janbu,
    u_max_mm:   u,
    cx: result.cx, cy: result.cy, R: result.R,
    geo: result.geo, slices: result.slices,
    converged: !!result.FoS_bishop
  };
}
