// S11 - age-sex stratified cumulative hazard index.
import { readFileSync } from 'fs';
import { canvas, save, axisX, note, d3, W2 } from './svgkit.mjs';
import { warmScale, WARM_DEEP, INK, GREY, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s11.json'));
const PW = 330, PH = 150, M = { t: 30, l: 108 };
const { body, svg } = canvas(W2, PH + M.t + 62);
const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
const x = d3.scaleLog().domain([0.05, d3.max(D.groups, d => d.q[4]) * 1.2]).range([0, PW]);
const y = d3.scaleBand().domain(D.groups.map(d => d.group)).range([0, PH]).padding(0.42);
axisX(g, x, PH, { values: [0.1, 1, 10, 100], fmt: d3.format('~g'), label: 'cumulative hazard index (log scale)' });
g.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y2', PH)
  .attr('stroke', INK).attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
note(g, 'HI = 1', x(1) + 3, -6, { size: 7.2, fill: INK });
D.groups.forEach(d => {
  const yy = y(d.group), h = y.bandwidth(), mid = yy + h / 2;
  const [p5, q1, med, q3, p95] = d.q;
  g.append('line').attr('x1', x(p5)).attr('x2', x(p95)).attr('y1', mid).attr('y2', mid)
    .attr('stroke', GREY).attr('stroke-width', 1);
  g.append('rect').attr('x', x(q1)).attr('width', x(q3) - x(q1)).attr('y', yy).attr('height', h)
    .attr('fill', warmScale(d.exceed)).attr('stroke', INK).attr('stroke-width', 0.5);
  g.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', yy).attr('y2', yy + h)
    .attr('stroke', INK).attr('stroke-width', 1.4);
  g.append('text').attr('x', -6).attr('y', mid + 2.6).attr('text-anchor', 'end')
    .attr('font-size', 7.8).text(d.group);
  g.append('text').attr('x', -6).attr('y', mid + 11).attr('text-anchor', 'end')
    .attr('font-size', 6.4).attr('fill', GREY).text(`${d.ir} L/d, ${d.bw} kg`);
  g.append('text').attr('x', x(p95) + 6).attr('y', mid + 2.6).attr('font-size', 7)
    .attr('fill', WARM_DEEP).text(`${(d.exceed * 100).toFixed(0)}% above HI = 1`);
});
note(g, 'box = IQR with median; whisker = 5th–95th percentile; fill encodes the exceedance fraction',
     0, PH + 40, { size: 7.2, fill: GREY });
note(g, 'young children carry the highest index, because ingestion rate relative to body weight is highest',
     0, PH + 51, { size: 7.2, fill: GREY, italic: true });
console.log(save(body, 's11'));
