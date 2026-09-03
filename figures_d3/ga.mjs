// Graphical abstract - three-stage triptych, built from the same JSON the
// figures use, so it cannot drift from the paper's numbers.
//
// Elsevier wants a graphical abstract at least 1328 x 531 px; this is authored
// at 664 x 300 pt and exported at high dpi, giving well over that.
import { readFileSync } from 'fs';
import { canvas, save, note, colorbar, pill, textWidth, d3 } from './svgkit.mjs';
import { warmScale, WARMR, WARM, WARM_MID, WARM_DEEP, TEAL, TEAL_DEEP, TEAL_LIGHT,
         CONTOUR, INK, GREY, MUTE, PAPER, FONT } from './palette.mjs';
import { contours as d3contours } from 'd3-contour';

const F1 = JSON.parse(readFileSync('data/fig1.json'));
const F3 = JSON.parse(readFileSync('data/fig3.json'));
const F4 = JSON.parse(readFileSync('data/fig4.json'));
const F5 = JSON.parse(readFileSync('data/fig5.json'));
const S3 = JSON.parse(readFileSync('data/s3.json'));

const W = 664, H = 268;
const { body, svg } = canvas(W, H);
const COLW = (W - 40) / 3, CX = i => 20 + i * COLW;

// ---------- header ----------
svg.append('text').attr('x', W / 2).attr('y', 26).attr('text-anchor', 'middle')
  .attr('font-size', 15).attr('font-weight', 700).attr('fill', INK)
  .text('Cumulative multi-contaminant groundwater exposure in Bangladesh');
svg.append('text').attr('x', W / 2).attr('y', 42).attr('text-anchor', 'middle')
  .attr('font-size', 9.5).attr('font-style', 'italic').attr('fill', GREY)
  .text('A low-cost phosphate indicator ranks wells for testing; cumulative burden is twice arsenic alone');
svg.append('line').attr('x1', 20).attr('x2', W - 20).attr('y1', 50).attr('y2', 50)
  .attr('stroke', '#e3e8f0').attr('stroke-width', 1);

const stage = (i, n, title, col) => {
  const g = svg.append('g').attr('transform', `translate(${CX(i)},64)`);
  g.append('circle').attr('cx', 7).attr('cy', 0).attr('r', 7).attr('fill', col);
  g.append('text').attr('x', 7).attr('y', 3).attr('text-anchor', 'middle')
    .attr('font-size', 8).attr('font-weight', 700).attr('fill', PAPER).text(n);
  g.append('text').attr('x', 20).attr('y', 3.5).attr('font-size', 11)
    .attr('font-weight', 700).attr('fill', col).text(title);
  return g;
};
const arrow = x => svg.append('path')
  .attr('d', `M${x},170 l14,0 m-4,-4 l4,4 l-4,4`)
  .attr('stroke', INK).attr('stroke-width', 1.6).attr('fill', 'none');

// ================= 1. cheap indicator =================
{
  const g = stage(0, '1', 'Cheap indicator', TEAL_DEEP);
  const px = 8, py = 26, pw = COLW - 46, ph = 92;
  const p = g.append('g').attr('transform', `translate(${px},${py})`);
  const x = d3.scaleLinear().domain([0.2, 5]).range([0, pw]);
  const y = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
  p.append('rect').attr('x', x(1.5)).attr('width', x(2) - x(1.5)).attr('height', ph)
    .attr('fill', WARM).attr('opacity', 0.10);
  [0, 0.5, 1].forEach(v => p.append('line').attr('x1', 0).attr('x2', pw)
    .attr('y1', y(v)).attr('y2', y(v)).attr('stroke', '#ececec').attr('stroke-width', 0.6));
  p.append('path').attr('d', d3.line().x(d => x(d.thr)).y(d => y(d.sens))
    .curve(d3.curveMonotoneX)(F1.sweep)).attr('fill', 'none')
    .attr('stroke', WARM_DEEP).attr('stroke-width', 2.2);
  const s15 = F1.table.find(r => r.PO4_threshold_mgL === 1.5 && r.As_mode === 'WHO 10').sensitivity;
  const pct = v => Math.round(v * 100 + 1e-9);
  const sens2 = (Math.round(s15 * 100 + 1e-9) / 100).toFixed(2);   // 0.57, not 0.56
  p.append('circle').attr('cx', x(1.5)).attr('cy', y(s15)).attr('r', 3.4)
    .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.3);
  p.append('text').attr('x', x(1.5) + 6).attr('y', y(s15) - 5).attr('font-size', 8)
    .attr('font-weight', 700).attr('fill', WARM_DEEP).text(`${pct(s15)}% flagged`);
  p.append('line').attr('y2', ph).attr('stroke', '#bbb').attr('stroke-width', 0.8);
  p.append('line').attr('x1', 0).attr('x2', pw).attr('y1', ph).attr('y2', ph)
    .attr('stroke', '#bbb').attr('stroke-width', 0.8);
  // inside the plot, not in a left gutter that the canvas edge would clip
  note(p, 'wells above WHO As limit', 4, 10, { size: 6.8, fill: GREY });
  note(p, 'dissolved phosphate (mg/L) →', pw, ph + 11, { anchor: 'end', size: 7, fill: GREY });
  // phosphate test strip, drawn rather than iconised
  const st = g.append('g').attr('transform', `translate(${px + pw + 14},${py + 6})`);
  st.append('rect').attr('width', 11).attr('height', 46).attr('rx', 2)
    .attr('fill', PAPER).attr('stroke', TEAL_DEEP).attr('stroke-width', 1.1);
  [0, 1, 2, 3].forEach(i => st.append('rect').attr('x', 2).attr('y', 4 + i * 10)
    .attr('width', 7).attr('height', 8).attr('rx', 1)
    .attr('fill', d3.interpolateRgb(TEAL_LIGHT, TEAL_DEEP)(i / 3)));
  note(st, 'PO₄', 5.5, 56, { anchor: 'middle', size: 7, fill: TEAL_DEEP });
  note(st, 'strip', 5.5, 64, { anchor: 'middle', size: 7, fill: TEAL_DEEP });

  g.append('text').attr('x', 6).attr('y', 140).attr('font-size', 8).attr('fill', INK)
    .text('PO₄ ranks wells for confirmatory');
  g.append('text').attr('x', 6).attr('y', 150).attr('font-size', 8).attr('fill', INK)
    .text(`arsenic testing (sensitivity ${sens2})`);
  pill(g, { text: '≈ 1/10 the cost of an arsenic assay', x: (COLW - 34) / 2, y: 158,
            align: 'centre', fill: TEAL, textFill: TEAL_DEEP, maxW: COLW - 12 });
}
arrow(CX(1) - 22);

// ================= 2. doubled burden =================
{
  const g = stage(1, '2', 'Doubled burden', WARM_DEEP);
  const nat = F3.aggregation.find(r => r.phys_zone === 'NATIONAL');
  const asOnly = nat.DALY_as_only, multi = nat.DALY_dose_add;
  const mn = F3.contributions.find(c => c.contaminant === 'Mn2+').pct;
  const bx = 34, by = 26, bh = 96, bw = 34, gap = 34;
  const p = g.append('g').attr('transform', `translate(${bx},${by})`);
  const y = d3.scaleLinear().domain([0, multi * 1.14]).range([bh, 0]);
  note(p, 'annual DALYs', 0, -6, { size: 7, fill: GREY });
  p.append('line').attr('x1', -4).attr('x2', bw * 2 + gap + 6).attr('y1', bh).attr('y2', bh)
    .attr('stroke', '#bbb').attr('stroke-width', 0.8);
  // arsenic-only
  p.append('rect').attr('x', 0).attr('y', y(asOnly)).attr('width', bw).attr('height', bh - y(asOnly))
    .attr('fill', TEAL_DEEP);
  p.append('text').attr('x', bw / 2).attr('y', y(asOnly) - 5).attr('text-anchor', 'middle')
    .attr('font-size', 10).attr('font-weight', 700).attr('fill', TEAL_DEEP)
    .text(`${Math.round(asOnly / 1000)}k`);
  note(p, 'As-only', bw / 2, bh + 11, { anchor: 'middle', size: 7.5, fill: INK });
  // multi-contaminant, with the manganese share marked inside
  const mx = bw + gap;
  p.append('rect').attr('x', mx).attr('y', y(multi)).attr('width', bw).attr('height', bh - y(multi))
    .attr('fill', WARM);
  const mnH = (bh - y(multi)) * mn / 100;
  p.append('rect').attr('x', mx).attr('y', bh - mnH).attr('width', bw).attr('height', mnH)
    .attr('fill', WARM_DEEP);
  p.append('text').attr('x', mx + bw / 2).attr('y', bh - mnH / 2 + 3).attr('text-anchor', 'middle')
    .attr('font-size', 7.5).attr('font-weight', 700).attr('fill', PAPER)
    .text(`Mn ${mn.toFixed(0)}%`);
  p.append('text').attr('x', mx + bw / 2).attr('y', y(multi) - 5).attr('text-anchor', 'middle')
    .attr('font-size', 10).attr('font-weight', 700).attr('fill', WARM_DEEP)
    .text(`${Math.round(multi / 1000)}k`);
  note(p, 'multi', mx + bw / 2, bh + 11, { anchor: 'middle', size: 7.5, fill: INK });

  g.append('text').attr('x', COLW / 2 - 20).attr('y', 148).attr('text-anchor', 'middle')
    .attr('font-size', 9.5).attr('font-weight', 700).attr('fill', INK)
    .text(`${(multi / asOnly).toFixed(1)}× higher than arsenic-only`);
  pill(g, { text: 'manganese rivals arsenic, yet unmonitored', x: (COLW - 34) / 2, y: 158,
            align: 'centre', fill: WARM, textFill: WARM_DEEP, maxW: COLW - 12 });
}
arrow(CX(2) - 22);

// ================= 3. climate + cheap fix =================
{
  const g = stage(2, '3', 'Climate + cheap fix', WARM_MID);
  const { nx, ny, bbox } = F5, p5 = F5.panels.asx;
  const mw = 92, mh = 100, mx0 = 2, my0 = 20;
  const lonR = bbox.lon_max - bbox.lon_min, latR = bbox.lat_max - bbox.lat_min;
  const sc = Math.min(mw / lonR, mh / latR);
  const X = d3.scaleLinear().domain([bbox.lon_min, bbox.lon_max])
    .range([mx0 + (mw - lonR * sc) / 2, mx0 + (mw - lonR * sc) / 2 + lonR * sc]);
  const Y = d3.scaleLinear().domain([bbox.lat_min, bbox.lat_max])
    .range([my0 + (mh - latR * sc) / 2 + latR * sc, my0 + (mh - latR * sc) / 2]);
  const ring = r => 'M' + r.map(([lo, la]) => `${X(lo).toFixed(1)},${Y(la).toFixed(1)}`).join('L') + 'Z';
  const cp = g.append('defs').append('clipPath').attr('id', 'ga-clip');
  F5.outline.forEach(r => cp.append('path').attr('d', ring(r)));

  // same treatment as Figure 5: extend past the coast, smooth, contour
  const A = p5.values.slice();
  for (let pass = 0; pass < 40; pass++) {
    let left = 0; const B = A.slice();
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i; if (A[k] !== null) continue;
      let s = 0, n = 0;
      for (const [di, dj] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const w = A[jj * nx + ii]; if (w !== null) { s += w; n++; }
      }
      if (n) B[k] = s / n; else left++;
    }
    for (let k = 0; k < A.length; k++) A[k] = B[k];
    if (!left) break;
  }
  const mean = d3.mean(A.filter(v => v !== null));
  const Z0 = A.map(v => (v === null || !isFinite(v)) ? mean : v);
  d3.blur2({ data: Z0, width: nx, height: ny }, 3.2);
  const DS = 2, cx = Math.floor(nx / DS), cy = Math.floor(ny / DS);
  const Z = new Float64Array(cx * cy);
  for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) Z[j * cx + i] = Z0[(j * DS) * nx + i * DS];
  const inside = Array.from(Z0).filter((_, k) => p5.values[k] !== null).sort(d3.ascending);
  const lo = d3.quantile(inside, 0.02), hi = d3.quantile(inside, 0.98);
  const NB = 7, bands = d3.range(NB + 1).map(i => lo + (hi - lo) * i / NB);
  const inner = g.append('g').attr('clip-path', 'url(#ga-clip)');
  const gpath = d3.geoPath();
  const sx = X(bbox.lon_min + DS / (nx - 1) * lonR) - X(bbox.lon_min);
  const sy = Y(bbox.lat_min) - Y(bbox.lat_min + DS / (ny - 1) * latR);
  const tf = `translate(${X(bbox.lon_min)},${Y(bbox.lat_max)}) scale(${sx},${sy})`;
  d3contours().size([cx, cy]).thresholds(bands)(Z).forEach((c, i) => {
    const d = gpath(c);
    if (d) inner.append('path').attr('d', d).attr('fill', warmScale(i / (NB - 1))).attr('transform', tf);
  });
  d3contours().size([cx, cy]).thresholds(bands.filter((_, i) => i % 2 === 1))(Z).forEach(c => {
    const d = gpath(c);
    if (d) inner.append('path').attr('d', d).attr('fill', 'none').attr('stroke', CONTOUR)
      .attr('stroke-width', 0.6 / sx).attr('stroke-opacity', 0.75).attr('transform', tf);
  });
  F5.outline.forEach(r => g.append('path').attr('d', ring(r)).attr('fill', 'none')
    .attr('stroke', INK).attr('stroke-width', 0.7));
  colorbar(g, { x: mx0 + 8, y: my0 + mh + 4, w: mw - 16, h: 5, vmin: lo, vmax: hi,
                ramp: warmScale, label: 'wells > 10 µg/L As, IDW', fmt: d3.format('.0%') });

  // climate callout
  const cxr = mx0 + mw + 18;
  const th = g.append('g').attr('transform', `translate(${cxr + 16},${my0 + 6})`);
  th.append('rect').attr('x', -3.5).attr('width', 7).attr('height', 26).attr('rx', 3.5)
    .attr('fill', PAPER).attr('stroke', WARM_DEEP).attr('stroke-width', 1.2);
  th.append('rect').attr('x', -1.6).attr('y', 8).attr('width', 3.2).attr('height', 18).attr('fill', WARM_DEEP);
  th.append('circle').attr('cy', 30).attr('r', 5).attr('fill', WARM_DEEP);
  th.append('path').attr('d', 'M12,4 l0,22 m-4,-18 l4,-4 l4,4')
    .attr('stroke', WARM_DEEP).attr('stroke-width', 1.4).attr('fill', 'none');
  const nCross = F4.crossings.filter(d => d.base != null && d.base <= F4.who && d.med > F4.who).length;
  g.append('text').attr('x', cxr).attr('y', my0 + 50).attr('font-size', 8.5)
    .attr('font-weight', 700).attr('fill', WARM_DEEP)
    .text(`${nCross === 1 ? '1 zone' : `${nCross} zones`} crosses the`);
  g.append('text').attr('x', cxr).attr('y', my0 + 60).attr('font-size', 8.5)
    .attr('fill', WARM_DEEP).text('WHO As limit by 2050');
  note(g, '(CMIP6, SSP5-8.5)', cxr, my0 + 70, { size: 6.8, fill: GREY });

  // intervention box
  const s4 = S3.scenarios.find(s => s.scenario === 'S4_seasonal_switch');
  const s1 = S3.scenarios.find(s => s.scenario === 'S1_deepen_wells');
  const boxLines = ['Seasonal well-switching',
                    `averts ${Math.round(s4.annual_DALY_averted / 1000)}k DALYs/yr at ~$${s4.ICER_usd_per_daly.toFixed(0)} each`,
                    `${(s1.ICER_usd_per_daly / s4.ICER_usd_per_daly).toFixed(0)}× cheaper than well-deepening`];
  const boxW = Math.min(COLW - 12,
    Math.max(COLW - 34, 24 + Math.max(...boxLines.map((t, i) => textWidth(t, i === 0 ? 8.5 : 7.8, i === 0))) + 12));
  const box = g.append('g').attr('transform', `translate(2,144)`);
  box.append('rect').attr('width', boxW).attr('height', 42).attr('rx', 5)
    .attr('fill', TEAL).attr('opacity', 0.10).attr('stroke', TEAL_DEEP).attr('stroke-width', 0.9);
  box.append('circle').attr('cx', 13).attr('cy', 13).attr('r', 6).attr('fill', TEAL_DEEP);
  box.append('path').attr('d', 'M10,13 l2.2,2.4 l4.2,-5').attr('stroke', PAPER)
    .attr('stroke-width', 1.5).attr('fill', 'none').attr('stroke-linecap', 'round');
  box.append('text').attr('x', 24).attr('y', 16).attr('font-size', 8.5)
    .attr('font-weight', 700).attr('fill', TEAL_DEEP).text('Seasonal well-switching');
  box.append('text').attr('x', 24).attr('y', 27).attr('font-size', 7.8).attr('fill', INK)
    .text(`averts ${Math.round(s4.annual_DALY_averted / 1000)}k DALYs/yr at ~$${s4.ICER_usd_per_daly.toFixed(0)} each`);
  box.append('text').attr('x', 24).attr('y', 36).attr('font-size', 7.2).attr('fill', GREY)
    .text(`${(s1.ICER_usd_per_daly / s4.ICER_usd_per_daly).toFixed(0)}× cheaper than well-deepening`);
}
console.log(save(body, 'ga'));
