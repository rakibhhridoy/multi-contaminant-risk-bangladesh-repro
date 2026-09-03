// Figure 2 - concentration regimes and phosphate control.
// The dynamical-systems framing was withdrawn in this revision, so the figure
// presents what the cross-section supports: two concentration populations,
// the empirical phosphate relationship, and conditional co-occurrence between
// high-concentration regimes. Nothing here asserts a transition in time.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { CONTAM, WARM, WARM_DEEP, WARM_MID, TEAL, TEAL_DEEP, INK, GREY, MUTE,
         PAPER, warmScale, FS } from './palette.mjs';

const D = JSON.parse(readFileSync('data/fig2.json'));
// Derived width. Panel (d) carries a right-hand axis, so the right margin is
// wide enough to hold its ticks and rotated label.
const GX = 48, GY = 56, M = { t: 30, l: 50, r: 48 };
const PW = Math.floor((W2 - M.l - GX - M.r) / 2), PH = 148;
const H = M.t + PH * 2 + GY + 34;
const { body, svg } = canvas(W2, H);
const panel = (r, c) => svg.append('g')
  .attr('transform', `translate(${M.l + c * (PW + GX)},${M.t + r * (PH + GY)})`);
const LAB = { 'As': 'As', 'Mn2+': 'Mn', 'Fe2+': 'Fe', 'Cr3+': 'Cr', 'PO43-': 'PO₄', 'NO3-': 'NO₃' };

// ---------------- (a) concentration distributions, log10 ----------------
{
  const g = panel(0, 0);
  const keys = D.contaminants.filter(k => D.density[k]);
  const xs = d3.extent(keys.flatMap(k => d3.extent(D.density[k].x)));
  const ymax = d3.max(keys, k => d3.max(D.density[k].y));
  const x = d3.scaleLinear().domain(xs).range([0, PW]);
  const y = d3.scaleLinear().domain([0, ymax * 1.06]).range([PH, 0]);
  axisY(g, y, { ticks: 4, w: PW, label: 'density' });
  const decades = d3.range(Math.ceil(xs[0]), Math.floor(xs[1]) + 1);
  axisX(g, x, PH, { values: decades, fmt: v => `10${sup(v)}`, label: 'concentration (mg/L, log scale)' });
  keys.forEach(k => {
    const dd = D.density[k];
    const pts = dd.x.map((v, i) => [v, dd.y[i]]);
    g.append('path').attr('d', d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveBasis)(pts))
      .attr('fill', 'none').attr('stroke', CONTAM[k]).attr('stroke-width', 1.8);
  });
  const lg = g.append('g').attr('transform', 'translate(6,4)');
  keys.forEach((k, i) => {
    lg.append('line').attr('x1', 0).attr('x2', 13).attr('y1', i * 11).attr('y2', i * 11)
      .attr('stroke', CONTAM[k]).attr('stroke-width', 1.8);
    lg.append('text').attr('x', 17).attr('y', i * 11 + 3).attr('font-size', 7.5).text(LAB[k]);
  });
  tag(g, 'a');
}
function sup(v) {
  const S = { '-': '\u207b', '0': '\u2070', '1': '\u00b9', '2': '\u00b2', '3': '\u00b3',
              '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
              '8': '\u2078', '9': '\u2079' };
  return String(v).split('').map(ch => S[ch] ?? ch).join('');
}

// ---------------- (b) arsenic by depth, with the GMM antimodes ----------------
// The text cites this panel for the deep-zone antimodes, so it reports the
// distribution by depth interval and marks each fitted mode-separation threshold.
{
  const g = panel(0, 1);
  const DZ = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep'];
  const rows = DZ.map(z => D.depth_as.find(d => d.depth === z)).filter(Boolean);
  const anti = new Map(D.antimodes.filter(a => a.contaminant === 'As').map(a => [a.depth, a.saddle]));
  const x = d3.scaleLog().domain([0.05, 900]).range([0, PW]).clamp(true);
  const y = d3.scaleBand().domain(rows.map(d => d.depth)).range([0, PH]).padding(0.42);
  axisX(g, x, PH, { values: [0.1, 1, 10, 100], fmt: d3.format('~g'), label: 'arsenic (µg/L, log scale)' });
  g.append('line').attr('x1', x(D.who)).attr('x2', x(D.who)).attr('y1', 0).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.9).attr('stroke-dasharray', '4,2');
  note(g, `WHO ${D.who}`, x(D.who) + 3, PH - 4, { size: 7, fill: INK });
  rows.forEach(d => {
    const yy = y(d.depth), h = y.bandwidth(), mid = yy + h / 2;
    const [p5, q1, med, q3, p95] = d.q;
    g.append('line').attr('x1', x(p5)).attr('x2', x(p95)).attr('y1', mid).attr('y2', mid)
      .attr('stroke', GREY).attr('stroke-width', 1);
    g.append('rect').attr('x', x(q1)).attr('width', Math.max(1, x(q3) - x(q1)))
      .attr('y', yy).attr('height', h)
      .attr('fill', warmScale(d.frac_over)).attr('stroke', INK).attr('stroke-width', 0.5);
    g.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', yy).attr('y2', yy + h)
      .attr('stroke', INK).attr('stroke-width', 1.4);
    const a = anti.get(d.depth);
    if (a) {
      g.append('line').attr('x1', x(a)).attr('x2', x(a)).attr('y1', yy - 3).attr('y2', yy + h + 3)
        .attr('stroke', TEAL_DEEP).attr('stroke-width', 1.6);
      g.append('text').attr('x', x(a)).attr('y', yy - 6).attr('text-anchor', 'middle')
        .attr('font-size', 6.6).attr('fill', TEAL_DEEP).text(a < 1 ? a.toFixed(2) : a.toFixed(1));
    }
    g.append('text').attr('x', -5).attr('y', mid + 2.6).attr('text-anchor', 'end')
      .attr('font-size', 7.4).text(d.depth.replace('Medium_Deep', 'Med-deep'));
    g.append('text').attr('x', PW - 2).attr('y', mid - h / 2 - 2).attr('text-anchor', 'end')
      .attr('font-size', 6.6).attr('fill', GREY).text(`${(d.frac_over * 100).toFixed(0)}% > WHO`);
  });
  note(g, 'box = IQR with median; whisker = 5th–95th; teal tick = GMM antimode',
       PW / 2, PH + 40, { anchor: 'middle', size: 7, fill: GREY });
  tag(g, 'b');
}

// ---------------- (c) conditional co-occurrence ----------------
{
  const g = panel(1, 0);
  const keys = D.contaminants;
  const n = keys.length, cell = Math.min(PW, PH) / n;
  const off = (PW - cell * n) / 2;
  const P = new Map(D.pairs.map(d => [`${d.src}|${d.dst}`, d.p]));
  const SIG = new Map(D.pairs.map(d => [`${d.src}|${d.dst}`, d.sig]));
  // conditional probabilities top out near 0.5, so scale the ramp to the
  // observed range instead of to 1, where every cell would read as pale
  const pmax = d3.max(D.pairs, d => d.p) || 1;
  keys.forEach((a, i) => keys.forEach((b, j) => {
    const v = P.get(`${a}|${b}`);
    const gx = off + j * cell, gy = i * cell;
    g.append('rect').attr('x', gx).attr('y', gy).attr('width', cell - 1).attr('height', cell - 1)
      .attr('fill', a === b ? '#f2f2f2' : warmScale((v ?? 0) / pmax)).attr('rx', 1);
    if (a !== b && v != null)
      g.append('text').attr('x', gx + cell / 2 - 0.5).attr('y', gy + cell / 2 + 2.5)
        .attr('text-anchor', 'middle').attr('font-size', 6.4)
        .attr('fill', v / pmax > 0.62 ? PAPER : INK)
        .attr('font-weight', SIG.get(`${a}|${b}`) ? 700 : 400)
        .text(d3.format('.2f')(v));
  }));
  keys.forEach((k, i) => {
    g.append('text').attr('x', off - 4).attr('y', i * cell + cell / 2 + 2.5)
      .attr('text-anchor', 'end').attr('font-size', 7.5).text(LAB[k]);
    g.append('text').attr('x', off + i * cell + cell / 2).attr('y', PH + 10)
      .attr('text-anchor', 'middle').attr('font-size', 7.5).text(LAB[k]);
  });
  note(g, 'P( column high | row high ); bold = significant', PW / 2, PH + 24,
       { anchor: 'middle', size: 7.5, fill: GREY });
  tag(g, 'c');
}

// ---------------- (d) phosphate-conditioned arsenic ----------------
{
  const g = panel(1, 1);
  const b = D.po4_bins;
  const x = d3.scaleLinear().domain(d3.extent(b, d => d.x)).nice().range([0, PW]);
  const yL = d3.scaleLinear().domain([0, 0.8]).range([PH, 0]);
  const yR = d3.scaleLinear().domain([0, d3.max(b, d => d.q3) * 1.05]).range([PH, 0]);
  axisY(g, yL, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'wells above WHO guideline' });
  axisX(g, x, PH, { ticks: 5, label: 'dissolved phosphate (mg/L)' });
  g.append('rect').attr('x', x(D.saddle[0])).attr('width', x(D.saddle[1]) - x(D.saddle[0]))
    .attr('y', 0).attr('height', PH).attr('fill', WARM).attr('opacity', 0.08);
  // median arsenic with IQR, on the right axis, in teal
  g.append('path').attr('d', d3.area().x(d => x(d.x)).y0(d => yR(d.q1)).y1(d => yR(d.q3))
    .curve(d3.curveMonotoneX)(b)).attr('fill', TEAL).attr('opacity', 0.16);
  g.append('path').attr('d', d3.line().x(d => x(d.x)).y(d => yR(d.med)).curve(d3.curveMonotoneX)(b))
    .attr('fill', 'none').attr('stroke', TEAL_DEEP).attr('stroke-width', 1.6).attr('stroke-dasharray', '4,2');
  // the fraction above the guideline, the panel's primary series
  g.append('path').attr('d', d3.line().x(d => x(d.x)).y(d => yL(d.frac)).curve(d3.curveMonotoneX)(b))
    .attr('fill', 'none').attr('stroke', WARM_DEEP).attr('stroke-width', 2.2);
  g.selectAll('circle.f').data(b).join('circle').attr('class', 'f')
    .attr('cx', d => x(d.x)).attr('cy', d => yL(d.frac)).attr('r', 2.6)
    .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1);
  // right-hand axis for the median
  const ax = g.append('g').attr('transform', `translate(${PW},0)`);
  ax.append('line').attr('y2', PH).attr('stroke', TEAL_DEEP).attr('stroke-width', 0.8);
  yR.ticks(4).forEach(v => {
    ax.append('line').attr('x2', 3).attr('y1', yR(v)).attr('y2', yR(v)).attr('stroke', TEAL_DEEP).attr('stroke-width', 0.8);
    ax.append('text').attr('x', 6).attr('y', yR(v) + 3).attr('font-size', FS.tick).attr('fill', TEAL_DEEP).text(v);
  });
  ax.append('text').attr('transform', `translate(30,${PH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
    .attr('font-size', FS.label).attr('fill', TEAL_DEEP).text('median arsenic (µg/L)');
  const lg = g.append('g').attr('transform', 'translate(6,4)');
  [['fraction > WHO', WARM_DEEP, null], ['median As (IQR)', TEAL_DEEP, '4,2']].forEach((d, i) => {
    lg.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 11).attr('y2', i * 11)
      .attr('stroke', d[1]).attr('stroke-width', 2).attr('stroke-dasharray', d[2]);
    lg.append('text').attr('x', 18).attr('y', i * 11 + 3).attr('font-size', 7.5).text(d[0]);
  });
  tag(g, 'd');
}
console.log(save(body, 'fig2'));
