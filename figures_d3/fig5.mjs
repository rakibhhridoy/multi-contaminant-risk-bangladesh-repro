// Figure 5 - spatial distribution of groundwater risk, with contours.
//
// Drawn as filled contour bands rather than a pixel raster: the bands are true
// vector paths, so the figure stays small, prints sharp at any size, and the
// isolines the reviewer asked for fall out of the same computation. Contour
// lines are stroked in Paper 1's reserved contour blue, which is the one cool
// mark on the warm ramp and stays legible over every band.
import { readFileSync } from 'fs';
import { canvas, save, tag, note, colorbar, d3, W2 } from './svgkit.mjs';
import { warmScale, CONTOUR, INK, GREY, RULE, PAPER, FS } from './palette.mjs';
import { contours as d3contours } from 'd3-contour';

const D = JSON.parse(readFileSync('data/fig5.json'));
const { nx, ny, bbox } = D;

// Grid cells outside the national boundary are null. Extend the field into them
// by iterated neighbour-averaging before contouring, so isolines meet the coast
// at a natural angle instead of bunching against a hard edge; everything beyond
// the boundary is then clipped away.
function fillOutside(v) {
  const a = v.slice();
  for (let pass = 0; pass < 40; pass++) {
    let remaining = 0;
    const b = a.slice();
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (a[k] !== null) continue;
      let s = 0, n = 0;
      for (const [di, dj] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const w = a[jj * nx + ii];
        if (w !== null) { s += w; n++; }
      }
      if (n) b[k] = s / n; else remaining++;
    }
    for (let k = 0; k < a.length; k++) a[k] = b[k];
    if (!remaining) break;
  }
  const finite = a.filter(x => x !== null && isFinite(x));
  const fallback = d3.mean(finite);
  return a.map(x => (x === null || !isFinite(x)) ? fallback : x);
}

const PANELS = [
  { key: 'asx',  letter: 'a', title: 'Wells above WHO arsenic guideline',
    cbar: 'fraction of wells > 10 µg/L', fmt: d3.format('.0%') },
  { key: 'hi',   letter: 'b', title: 'Cumulative hazard index',
    cbar: 'hazard index', fmt: d3.format('.1f') },
  { key: 'vuln', letter: 'c', title: 'Climate vulnerability',
    cbar: 'HI × normalised seasonal sensitivity', fmt: d3.format('.1f') },
];

const PW = 168, PH = 210, GAP = 12, M = { t: 24, l: 12, b: 46 };
const H = M.t + PH + M.b;
const { body, svg } = canvas(W2, H);

// geographic -> panel screen, preserving aspect
const lonR = bbox.lon_max - bbox.lon_min, latR = bbox.lat_max - bbox.lat_min;
const sc = Math.min(PW / lonR, PH / latR);
const mapW = lonR * sc, mapH = latR * sc;
const padX = (PW - mapW) / 2, padY = (PH - mapH) / 2;
const X = d3.scaleLinear().domain([bbox.lon_min, bbox.lon_max]).range([padX, padX + mapW]);
const Y = d3.scaleLinear().domain([bbox.lat_min, bbox.lat_max]).range([padY + mapH, padY]);
// grid index -> screen (d3-contour returns coordinates in grid units)
const gx = i => X(bbox.lon_min + (i / (nx - 1)) * lonR);   // full-grid helper (outline)
const gy = j => Y(bbox.lat_min + (j / (ny - 1)) * latR);
const outlinePath = ring => 'M' + ring.map(([lo, la]) => `${X(lo).toFixed(2)},${Y(la).toFixed(2)}`).join('L') + 'Z';

PANELS.forEach((P, pi) => {
  const p = D.panels[P.key];
  const g = svg.append('g').attr('transform', `translate(${M.l + pi * (PW + GAP)},${M.t})`);
  const cid = `clip-${P.key}`;
  const cp = g.append('defs').append('clipPath').attr('id', cid);
  D.outline.forEach(r => cp.append('path').attr('d', outlinePath(r)));

  // IDW with k=10, p=2 puts a small bullseye around every well. Contour bands
  // make that texture far more visible than a raster does, and it is sampling
  // noise rather than signal, so the field is Gaussian-blurred before tracing.
  // The blur is applied to the boundary-extended field so the coast is unaffected.
  const Z0 = fillOutside(p.values);
  d3.blur2({ data: Z0, width: nx, height: ny }, 3.2);
  // Contour on a downsampled field. The blur has already removed everything
  // finer than the sampling network can support, so tracing at full 200x200
  // only multiplies vertices: it produced a 72 MB PDF, far past what a journal
  // will accept, for no visible gain.
  const DS = 2, cx = Math.floor(nx / DS), cy = Math.floor(ny / DS);
  const Z = new Float64Array(cx * cy);
  for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++)
    Z[j * cx + i] = Z0[(j * DS) * nx + (i * DS)];
  // Scale to the field as drawn. Smoothing damps the extremes, so the exported
  // percentiles of the raw surface would promise a range the map never reaches
  // and leave the deep end of the ramp unused. Computed over cells inside the
  // national boundary only, since everything else is clipped away.
  const insideVals = Array.from(Z0).filter((_, k) => p.values[k] !== null).sort(d3.ascending);
  const lo = d3.quantile(insideVals, 0.02), hi = d3.quantile(insideVals, 0.98);
  const NB = 8;
  const bands = d3.range(NB + 1).map(i => lo + (hi - lo) * i / NB);
  const inner = g.append('g').attr('clip-path', `url(#${cid})`);

  // ---- filled bands ----
  const path = d3.geoPath();
  const round2 = d => d.replace(/-?\d+\.\d{3,}/g, m => (+m).toFixed(2));
  const sx = gx(DS) - gx(0), sy = gy(0) - gy(DS);
  const gridToScreen = `translate(${gx(0)},${gy(ny - 1)}) scale(${sx},${sy})`;
  const toScreen = ([gxx, gyy]) => [gx(0) + gxx * sx, gy(ny - 1) + gyy * sy];

  d3contours().size([cx, cy]).thresholds(bands)(Z).forEach((c, i) => {
    const dpath = path(c);
    if (dpath) inner.append('path').attr('d', round2(dpath))
      .attr('fill', warmScale(i / (NB - 1))).attr('transform', gridToScreen);
  });

  // ---- isolines, every second band, each carrying one value label ----
  // A label may only sit where it is clear of the drawn coastline. Testing the
  // interpolation mask was not enough: the mask is the true boundary while the
  // outline drawn on the page is a simplified version of it, and near narrow
  // spurs the two differ enough that labels still straddled the coast. So the
  // test is against the polygon actually drawn, in screen space, and it checks
  // the whole label box rather than just its centre.
  const ringsScreen = D.outline.map(r => r.map(([lo_, la_]) => [X(lo_), Y(la_)]));
  const inPoly = (px, py) => ringsScreen.some(r => {
    let hit = false;
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  });
  const labelFits = (px, py) =>
    [[0, 0], [-16, 0], [16, 0], [0, -7], [0, 7]].every(([dx, dy]) => inPoly(px + dx, py + dy));

  const labelled = bands.filter((_, i) => i % 2 === 1);
  const placed = [];
  d3contours().size([cx, cy]).thresholds(labelled)(Z).forEach(c => {
    const dpath = path(c);
    if (!dpath) return;
    inner.append('path').attr('d', round2(dpath)).attr('fill', 'none')
      .attr('stroke', CONTOUR)
      // divide by the SAME factor the group is scaled by, or the line lands at
      // twice its intended width
      .attr('stroke-width', 0.75 / sx)
      .attr('stroke-opacity', 0.8).attr('stroke-linejoin', 'round')
      .attr('transform', gridToScreen);

    // one label per isoline: its longest ring, at the flattest point that fits
    const rings = c.coordinates.flat();
    if (!rings.length) return;
    const ring = rings.reduce((a, b) => (b.length > a.length ? b : a));
    if (ring.length < 24) return;
    let best = null;
    for (let i = 6; i < ring.length - 6; i += 2) {
      const [ax, ay] = ring[i - 5], [bx2, by2] = ring[i + 5];
      const slope = Math.abs((by2 - ay) / ((bx2 - ax) || 1e-6));
      const [px, py] = toScreen(ring[i]);
      if (placed.some(q => Math.hypot(q.px - px, q.py - py) < 28)) continue;
      if (!labelFits(px, py)) continue;
      if (!best || slope < best.slope) best = { slope, px, py };
    }
    if (!best) return;
    placed.push(best);
    g.append('text').attr('x', best.px).attr('y', best.py + 2)
      .attr('text-anchor', 'middle').attr('font-size', 6.2)
      .attr('fill', CONTOUR).attr('stroke', PAPER).attr('stroke-width', 2.2)
      .attr('paint-order', 'stroke').attr('font-weight', 600)
      .text(P.fmt(c.value));
  });

  // sampling network, faint, so the reader can see where the surface is supported
  if (pi === 0) inner.selectAll('circle.w').data(D.wells).join('circle')
    .attr('cx', d => X(d.x)).attr('cy', d => Y(d.y)).attr('r', 0.7)
    .attr('fill', INK).attr('opacity', 0.30);

  // national boundary on top
  D.outline.forEach(r => g.append('path').attr('d', outlinePath(r))
    .attr('fill', 'none').attr('stroke', INK).attr('stroke-width', 0.9));

  g.append('text').attr('x', PW / 2).attr('y', -8).attr('text-anchor', 'middle')
    .attr('font-size', 8.5).attr('font-weight', 700).text(P.title);
  tag(g, P.letter, 2, 4);

  colorbar(g, { x: padX + 6, y: PH + 14, w: mapW - 12, h: 7,
                vmin: lo, vmax: hi, ramp: warmScale, label: P.cbar, fmt: P.fmt });
});

// one shared explanation of the isolines
note(svg, 'isolines every second band', M.l + 4, H - 6, { size: 7.5, fill: CONTOUR });
note(svg, 'grey dots in (a) mark the 988 sampling locations', M.l + 150, H - 6, { size: 7.5, fill: GREY });
console.log(save(body, 'fig5'));
