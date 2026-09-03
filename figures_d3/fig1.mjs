// Figure 1 - phosphate as a prioritising indicator for arsenic screening.
//
// Replaces the schematic cost-comparison graphic. Panel (a) is the screening
// performance actually measured in the data, so the figure now states the
// operating characteristic that reviewer R1.3 objected to seeing omitted:
// at the 1.5 mg/L cut-off the rule flags 57% of high-arsenic wells and misses
// 43%. Panel (b) keeps the cost argument, which is what makes a moderate screen
// worth deploying at all.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, GRID, RULE, FS, PAPER } from './palette.mjs';

const D = JSON.parse(readFileSync('data/fig1.json'));
const H = 274, M = { t: 30, r: 14, b: 68, l: 52 };
const { body, svg } = canvas(W2, H);

// ---------- panel (a): sensitivity / specificity against the cut-off ----------
const aw = 268, ah = H - M.t - M.b;
const A = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
const x = d3.scaleLinear().domain([0.2, 5]).range([0, aw]);
const y = d3.scaleLinear().domain([0, 1]).range([ah, 0]);

axisY(A, y, { values: [0, .25, .5, .57, .75, 1], w: aw,
              fmt: v => `${Math.round(v * 100)}%`, label: 'proportion' });
axisX(A, x, ah, { ticks: 6, label: '' });
note(A, 'dissolved phosphate cut-off (mg/L)', aw / 2, ah + 40, { anchor: 'middle', size: FS.label, fill: INK });

// the 1.5-2.0 mg/L band the paper recommends
A.append('rect').attr('x', x(1.5)).attr('y', 0).attr('width', x(2) - x(1.5))
  .attr('height', ah).attr('fill', WARM).attr('opacity', 0.07);
note(A, 'recommended band', (x(1.5) + x(2)) / 2, -6, { anchor: 'middle', size: 7.5, fill: WARM_DEEP });

const line = k => d3.line().x(d => x(d.thr)).y(d => y(d[k])).curve(d3.curveMonotoneX)(D.sweep);

// missed fraction shaded under the sensitivity curve: the reviewer's point, drawn
A.append('path').attr('d', d3.area().x(d => x(d.thr)).y0(d => y(d.sens)).y1(y(1))
  .curve(d3.curveMonotoneX)(D.sweep))
  .attr('fill', MUTE).attr('opacity', 0.30);

A.append('path').attr('d', line('spec')).attr('fill', 'none')
  .attr('stroke', TEAL).attr('stroke-width', 2).attr('stroke-dasharray', '4,2.5');
A.append('path').attr('d', line('sens')).attr('fill', 'none')
  .attr('stroke', WARM_DEEP).attr('stroke-width', 2.4);

// mark the two operating points quoted in the text
const s15 = D.table.find(r => r.PO4_threshold_mgL === 1.5 && r.As_mode === 'WHO 10').sensitivity;
// 0.565*100 is 56.49999999999999 in IEEE754, which rounds to 56 and would print
// 44% missed where the manuscript says 43%. Nudge before rounding.
const pct = v => Math.round(v * 100 + 1e-9);
const missedPct = 100 - pct(s15);   // 100 - 57 = 43, matching the text

// Operating points come from the SI table (Table S12), not from the sweep, so the
// figure reports exactly the sensitivities quoted in the manuscript.
D.marks.forEach(mk => {
  const row = D.table.find(r => r.PO4_threshold_mgL === mk.thr && r.As_mode === 'WHO 10');
  const p = { thr: mk.thr, sens: row.sensitivity, missed: 1 - row.sensitivity };
  A.append('line').attr('x1', x(p.thr)).attr('x2', x(p.thr)).attr('y1', y(p.sens)).attr('y2', ah)
    .attr('stroke', INK).attr('stroke-width', 0.7).attr('stroke-dasharray', '2,2');
  A.append('circle').attr('cx', x(p.thr)).attr('cy', y(p.sens)).attr('r', 3.6)
    .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.4);
  // Both labels sit ABOVE the marker, stacked. Putting the threshold below it
  // placed the text directly on the descending sensitivity curve.
  A.append('text').attr('x', x(p.thr) + 6).attr('y', y(p.sens) - 17).attr('font-size', 8)
    .attr('font-weight', 700).attr('fill', WARM_DEEP)
    .text(`${pct(p.sens)}%`);
  A.append('text').attr('x', x(p.thr) + 6).attr('y', y(p.sens) - 8).attr('font-size', 7)
    .attr('fill', GREY).text(`at ${p.thr}`);
});

// Right-aligned and anchored to the plot's right edge, so the block grows leftward
// into empty space and can never run past the panel however long the text is.
const annX = aw - 6;
A.append('text').attr('x', annX).attr('y', y(0.88)).attr('text-anchor', 'end')
  .attr('font-size', 8.5).attr('font-weight', 700).attr('fill', '#6b6b6b')
  .text(`${missedPct}% missed at 1.5 mg/L`);
note(A, 'a negative result does not clear a well', annX, y(0.88) + 11,
     { anchor: 'end', size: 7.5, fill: GREY, italic: true });

// legend below the axis, clear of both curves
const lg = A.append('g').attr('transform', `translate(0,${ah + 50})`);
let lx = 0;
[['sensitivity (wells flagged)', WARM_DEEP, null], ['specificity', TEAL, '4,2.5'],
 ['missed', MUTE, null]].forEach(d => {
  if (d[0] === 'missed') {
    lg.append('rect').attr('x', lx).attr('y', -4).attr('width', 16).attr('height', 7)
      .attr('fill', MUTE).attr('opacity', 0.45);
  } else {
    lg.append('line').attr('x1', lx).attr('x2', lx + 16).attr('stroke', d[1])
      .attr('stroke-width', 2.2).attr('stroke-dasharray', d[2]);
  }
  lg.append('text').attr('x', lx + 21).attr('y', 3).attr('font-size', 8).text(d[0]);
  lx += 26 + d[0].length * 4.3;
});
tag(A, 'a');

// ---------- panel (b): why a moderate screen is still worth deploying ----------
const bx = M.l + aw + 72;
const B = svg.append('g').attr('transform', `translate(${bx},${M.t})`);
const bw = W2 - bx - M.r - 6;
const cy = d3.scaleLog().domain([0.2, 20]).range([ah, 0]);
axisY(B, cy, { values: [0.3, 1, 3, 10], w: bw, fmt: d3.format('~g'),
               label: 'USD per test (log)' });

const bars = [
  { k: 'arsenic', lab: 'arsenic assay', rng: D.cost.arsenic, col: WARM },
  { k: 'phosphate', lab: 'phosphate strip', rng: D.cost.phosphate, col: TEAL },
];
const bandW = 40, gap = 28;
bars.forEach((b, i) => {
  const cx = 16 + i * (bandW + gap);
  B.append('rect').attr('x', cx).attr('y', cy(b.rng[1]))
    .attr('width', bandW).attr('height', cy(b.rng[0]) - cy(b.rng[1]))
    .attr('fill', b.col).attr('opacity', 0.85).attr('rx', 1.5);
  B.append('text').attr('x', cx + bandW / 2).attr('y', cy(b.rng[1]) - 6)
    .attr('text-anchor', 'middle').attr('font-size', 8).attr('font-weight', 700)
    .attr('fill', b.col)
    .text(`$${b.rng[0]}–${b.rng[1]}`);
  note(B, b.lab, cx + bandW / 2, ah + 14, { anchor: 'middle', size: 8, fill: INK });
});
// the ratio, drawn as the comparison it is
const x1 = 16 + bandW, x2 = 16 + bandW + gap;
B.append('path').attr('d', `M${x1 + 4},${cy(7)} L${x2 - 4},${cy(0.6)}`)
  .attr('stroke', GREY).attr('stroke-width', 0.9).attr('stroke-dasharray', '3,2');
B.append('text').attr('x', (x1 + x2) / 2).attr('y', cy(2.6))
  .attr('text-anchor', 'middle').attr('font-size', 9).attr('font-weight', 700)
  .attr('fill', INK).attr('stroke', PAPER).attr('stroke-width', 3).attr('paint-order', 'stroke')
  .text('~10×');
note(B, 'cheaper', (x1 + x2) / 2, cy(2.6) + 10, { anchor: 'middle', size: 7.5, fill: GREY });
tag(B, 'b');

console.log(save(body, 'fig1'));
