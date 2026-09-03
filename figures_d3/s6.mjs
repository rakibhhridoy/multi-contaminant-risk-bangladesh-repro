// S6 - competitive surface-complexation model of phosphate-driven arsenate desorption.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s6.json'));
const PW = 205, PH = 155, GX = 82, M = { t: 30, l: 58 };
const { body, svg } = canvas(W2, PH + M.t + 60);
// (a) desorbed fraction against phosphate
{
  const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
  const x = d3.scaleLog().domain(d3.extent(D.scm, d => d.PO4_mg_L)).range([0, PW]);
  const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'arsenate fraction' });
  axisX(g, x, PH, { values: [0.01, 0.1, 1, 10], fmt: d3.format('~g'), label: 'dissolved phosphate (mg/L, log)' });
  g.append('rect').attr('x', x(D.saddle[0])).attr('width', x(D.saddle[1]) - x(D.saddle[0]))
    .attr('y', 0).attr('height', PH).attr('fill', WARM).attr('opacity', 0.09);
  note(g, 'saddle band', (x(D.saddle[0]) + x(D.saddle[1])) / 2, -5, { anchor: 'middle', size: 7, fill: WARM_DEEP });
  const line = k => d3.line().x(d => x(d.PO4_mg_L)).y(d => y(d[k])).curve(d3.curveMonotoneX)(D.scm);
  g.append('path').attr('d', line('As_sorbed_frac')).attr('fill', 'none')
    .attr('stroke', TEAL_DEEP).attr('stroke-width', 2).attr('stroke-dasharray', '4,2.5');
  g.append('path').attr('d', line('As_desorbed_frac')).attr('fill', 'none')
    .attr('stroke', WARM_DEEP).attr('stroke-width', 2.4);
  const lg = g.append('g').attr('transform', 'translate(6,4)');
  [['desorbed (mobilised)', WARM_DEEP, null], ['sorbed to HFO', TEAL_DEEP, '4,2.5']].forEach((d, i) => {
    lg.append('line').attr('x1', 0).attr('x2', 15).attr('y1', i * 11).attr('y2', i * 11)
      .attr('stroke', d[1]).attr('stroke-width', 2.1).attr('stroke-dasharray', d[2]);
    lg.append('text').attr('x', 20).attr('y', i * 11 + 3).attr('font-size', 7.4).text(d[0]);
  });
  tag(g, 'a');
}
// (b) Relative enrichment of aqueous arsenic, as the caption describes.
// The PHREEQC titration table is degenerate here (non-zero sorbed mass at only
// one of its forty rows), so the panel is drawn from the surface-complexation
// model instead, which is the quantity the caption names.
{
  const g = svg.append('g').attr('transform', `translate(${M.l + PW + GX},${M.t})`);
  const x = d3.scaleLog().domain(d3.extent(D.scm, d => d.PO4_mg_L)).range([0, PW]);
  const y = d3.scaleLinear().domain([0.9, d3.max(D.scm, d => d.As_aq_relative) * 1.06]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: v => `${v.toFixed(1)}×`, label: 'aqueous arsenic, relative to baseline' });
  axisX(g, x, PH, { values: [0.01, 0.1, 1, 10], fmt: d3.format('~g'), label: 'dissolved phosphate (mg/L, log)' });
  g.append('rect').attr('x', x(D.saddle[0])).attr('width', x(D.saddle[1]) - x(D.saddle[0]))
    .attr('y', 0).attr('height', PH).attr('fill', WARM).attr('opacity', 0.09);
  note(g, 'saddle band', (x(D.saddle[0]) + x(D.saddle[1])) / 2, -5, { anchor: 'middle', size: 7, fill: WARM_DEEP });
  g.append('path').attr('d', d3.line().x(d => x(d.PO4_mg_L)).y(d => y(d.As_aq_relative))
    .curve(d3.curveMonotoneX)(D.scm))
    .attr('fill', 'none').attr('stroke', WARM_DEEP).attr('stroke-width', 2.4);
  // mark the enrichment across the saddle band, which is the panel's point
  D.saddle.forEach(t => {
    const r = D.scm.reduce((a, b) => Math.abs(b.PO4_mg_L - t) < Math.abs(a.PO4_mg_L - t) ? b : a);
    g.append('circle').attr('cx', x(r.PO4_mg_L)).attr('cy', y(r.As_aq_relative)).attr('r', 3.2)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.2);
    g.append('text').attr('x', x(r.PO4_mg_L) + 6).attr('y', y(r.As_aq_relative) + 3)
      .attr('font-size', 7).attr('fill', WARM_DEEP).text(`${r.As_aq_relative.toFixed(1)}×`);
  });
  note(g, 'inventory-independent: depends on the affinity ratio, not the absolute',
       6, PH + 40, { size: 7.2, fill: GREY });
  note(g, 'arsenic inventory or site density', 6, PH + 51, { size: 7.2, fill: GREY });
  tag(g, 'b');
}
console.log(save(body, 's6'));
