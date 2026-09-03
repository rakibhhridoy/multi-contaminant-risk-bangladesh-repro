// S7 - external validation of the phosphate-arsenic coupling.
import { readFileSync } from 'fs';
import { canvas, save, axisX, note, d3, W2 } from './svgkit.mjs';
import { WARM_DEEP, TEAL_DEEP, INK, GREY, PAPER, MUTE } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s7.json'));
const rows = [
  { name: 'Bangladesh (this study)', n: D.bangladesh.n, rho: D.bangladesh.rho, primary: true },
  ...D.cohorts.map(c => ({ name: c.cohort, n: c.n, rho: c.spearman_As_PO4, p: c.spearman_p })),
];
const PW = 300, PH = 26 * rows.length + 14, M = { t: 30, l: 232 };
const { body, svg } = canvas(W2, PH + M.t + 62);
const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
const x = d3.scaleLinear().domain([-0.15, 0.6]).range([0, PW]);
const y = d3.scaleBand().domain(rows.map(d => d.name)).range([0, PH]).padding(0.45);
axisX(g, x, PH, { ticks: 5, fmt: d3.format('.1f'), label: "Spearman's ρ, arsenic against phosphate" });
g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y2', PH)
  .attr('stroke', INK).attr('stroke-width', 1);
rows.forEach(d => {
  const yy = y(d.name), h = y.bandwidth();
  const col = d.rho > 0.15 ? WARM_DEEP : d.rho > 0 ? MUTE : TEAL_DEEP;
  g.append('rect').attr('x', Math.min(x(0), x(d.rho))).attr('width', Math.abs(x(d.rho) - x(0)))
    .attr('y', yy).attr('height', h).attr('fill', col).attr('opacity', d.primary ? 1 : 0.85).attr('rx', 1.5);
  const nm = d.name.replace(/"/g, '').split('(')[0].trim();
  const sub = (d.name.match(/\(([^)]*)\)/) || [, ''])[1];
  g.append('text').attr('x', -6).attr('y', yy + h / 2 + (sub ? -1 : 2.6)).attr('text-anchor', 'end')
    .attr('font-size', 7.6).attr('font-weight', d.primary ? 700 : 400).text(nm);
  if (sub) g.append('text').attr('x', -6).attr('y', yy + h / 2 + 8).attr('text-anchor', 'end')
    .attr('font-size', 6.4).attr('fill', GREY).text(sub);
  // always label to the right of zero, so a negative bar's text cannot run back
  // into the row label on the left
  g.append('text').attr('x', Math.max(x(d.rho), x(0)) + 6).attr('y', yy + h / 2 + 2.6)
    .attr('font-size', 7).attr('fill', INK).text(`ρ = ${d.rho.toFixed(2)}   n = ${d.n}`);
});
note(g, 'the coupling reproduces in a freshwater reductive delta and is absent in the saline tidal plain,',
     0, PH + 40, { size: 7.2, fill: GREY });
note(g, 'where salinity-driven mobilisation dominates and wells already sit above the phosphate saddle',
     0, PH + 51, { size: 7.2, fill: GREY });
console.log(save(body, 's7'));
