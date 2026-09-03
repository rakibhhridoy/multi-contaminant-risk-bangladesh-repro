// S10 - between-campaign shift in high-concentration-mode membership, paired wells.
import { readFileSync } from 'fs';
import { canvas, save, axisY, note, d3, W2 } from './svgkit.mjs';
import { CONTAM, WARM_DEEP, TEAL_DEEP, INK, GREY, PAPER, MUTE } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s10.json'));
const LAB = { 'As': 'arsenic', 'Mn2+': 'manganese', 'Fe2+': 'iron', 'Cr3+': 'chromium',
              'PO43-': 'phosphate', 'NO3-': 'nitrate' };
const PW = 340, PH = 170, M = { t: 30, l: 74 };
const { body, svg } = canvas(W2, PH + M.t + 64);
const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
const x = d3.scaleBand().domain(D.rows.map(d => d.contaminant)).range([0, PW]).padding(0.45);
const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'wells in the high-concentration mode' });
D.rows.forEach(d => {
  const cx = x(d.contaminant) + x.bandwidth() / 2, w = Math.min(30, x.bandwidth() / 2.2);
  g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d.dry)).attr('y2', y(d.wet))
    .attr('stroke', MUTE).attr('stroke-width', 2);
  g.append('circle').attr('cx', cx - w / 2).attr('cy', y(d.dry)).attr('r', 4)
    .attr('fill', TEAL_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.2);
  g.append('circle').attr('cx', cx + w / 2).attr('cy', y(d.wet)).attr('r', 4)
    .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.2);
  const dd = (d.wet - d.dry) * 100;
  g.append('text').attr('x', cx).attr('y', Math.min(y(d.dry), y(d.wet)) - 8)
    .attr('text-anchor', 'middle').attr('font-size', 7).attr('font-weight', 700)
    .attr('fill', dd >= 0 ? WARM_DEEP : TEAL_DEEP)
    .text(`${dd > 0 ? '+' : ''}${dd.toFixed(0)} pp`);
  g.append('text').attr('x', cx).attr('y', PH + 12).attr('text-anchor', 'middle')
    .attr('font-size', 7.4).text(LAB[d.contaminant] || d.contaminant);
  g.append('text').attr('x', cx).attr('y', PH + 21).attr('text-anchor', 'middle')
    .attr('font-size', 6.4).attr('fill', GREY)
    .text(`> ${d.thr < 1 ? d.thr.toFixed(2) : d.thr.toFixed(1)}`);
});
const lg = g.append('g').attr('transform', `translate(${PW - 96},2)`);
[['dry campaign', TEAL_DEEP], ['wet campaign', WARM_DEEP]].forEach((d, i) => {
  lg.append('circle').attr('cx', 5).attr('cy', i * 11).attr('r', 3.6).attr('fill', d[1]);
  lg.append('text').attr('x', 13).attr('y', i * 11 + 3).attr('font-size', 7.4).text(d[0]);
});
note(g, `${D.n_paired} wells sampled in both campaigns; a fixed pooled-data threshold per contaminant`,
     0, PH + 42, { size: 7.2, fill: GREY });
note(g, 'the campaigns differ in calendar year and in analytical method, so this is a between-campaign contrast',
     0, PH + 53, { size: 7.2, fill: GREY, italic: true });
console.log(save(body, 's10'));
