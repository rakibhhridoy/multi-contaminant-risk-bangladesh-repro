// S8 - one-way (tornado) sensitivity of the national DALY burden.
import { readFileSync } from 'fs';
import { canvas, save, axisX, note, d3, W2 } from './svgkit.mjs';
import { WARM, TEAL_DEEP, INK, GREY, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s8.json'));
// Parameter names arrive with LaTeX in them ("Hill $P_{\\max}$"), which would
// print raw. Strip the maths and spell the symbol out.
const tidy = t => t.replace(/\$P_\{\\max\}\$/g, 'Pmax').replace(/\$([A-Za-z])\$/g, '$1')
                   .replace(/\\max/g, 'max').replace(/[${}\\]/g, '').trim();
const R = D.rows.slice().sort((a, b) => b.swing_pct - a.swing_pct)
             .map(d => ({ ...d, label: tidy(d.parameter) }));
const PW = 330, PH = 24 * R.length + 16, M = { t: 30, l: 150 };
const { body, svg } = canvas(W2, PH + M.t + 62);
const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
const lo = d3.min(R, d => Math.min(d.DALY_at_low, d.DALY_at_high));
const hi = d3.max(R, d => Math.max(d.DALY_at_low, d.DALY_at_high));
const pad = (hi - lo) * 0.12;
const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, PW]);
const y = d3.scaleBand().domain(R.map(d => d.label)).range([0, PH]).padding(0.42);
axisX(g, x, PH, { ticks: 5, fmt: v => `${(v / 1000).toFixed(0)}k`, label: 'national annual DALYs' });
g.append('line').attr('x1', x(D.baseline)).attr('x2', x(D.baseline)).attr('y2', PH)
  .attr('stroke', INK).attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
note(g, `baseline ${d3.format(',.0f')(D.baseline)}`, x(D.baseline), -8, { anchor: 'middle', size: 7.4, fill: INK });
R.forEach(d => {
  const yy = y(d.label), h = y.bandwidth();
  const a = Math.min(d.DALY_at_low, d.DALY_at_high), b = Math.max(d.DALY_at_low, d.DALY_at_high);
  g.append('rect').attr('x', x(a)).attr('width', x(b) - x(a)).attr('y', yy).attr('height', h)
    .attr('fill', WARM).attr('opacity', 0.85).attr('rx', 1.5);
  g.append('text').attr('x', -6).attr('y', yy + h / 2 + 2.6).attr('text-anchor', 'end')
    .attr('font-size', 7.6).text(d.label);
  g.append('text').attr('x', x(b) + 5).attr('y', yy + h / 2 + 2.6).attr('font-size', 7)
    .attr('fill', GREY).text(`swing ${d.swing_pct.toFixed(1)}%`);
  g.append('text').attr('x', x(a) + 4).attr('y', yy + h / 2 + 2.6)
    .attr('font-size', 6.4).attr('fill', PAPER)
    .text(`${d.low_value}–${d.high_value}`);
});
note(g, 'each parameter varied across its plausible range with the others held at their central values',
     0, PH + 40, { size: 7.2, fill: GREY });
console.log(save(body, 's8'));
