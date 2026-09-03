// S13 - the multi/arsenic-only DALY ratio as a continuous function of the
// manganese disability weight.
//
// R2.1 asks why an arsenic-calibrated dose-response is applied to a summed
// multi-contaminant hazard index. Part of the answer is that the fully
// contaminant-specific alternative does not settle the question: its answer is
// almost entirely determined by one unmeasured choice, how manganese-attributable
// neurodevelopmental impairment is valued. This panel sweeps that weight so the
// bracketing can be seen rather than asserted, and shows dose-addition sitting
// inside the resulting range rather than at its top.
import { readFileSync } from 'fs';
import { canvas, save, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER } from './palette.mjs';

const D = JSON.parse(readFileSync('data/s13.json'));
const PW = 300, PH = 190, M = { t: 30, l: 70, r: 174 };
const { body, svg } = canvas(W2, PH + M.t + 74);
const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

const x = d3.scaleLinear().domain([0, 0.4]).range([0, PW]);
const y = d3.scaleLinear().domain([1, d3.max(D.sweep, d => d.ratio) * 1.06]).range([PH, 0]);
axisY(g, y, { ticks: 5, w: PW, fmt: v => `${v.toFixed(1)}×`,
              label: 'multi-contaminant / arsenic-only DALYs' });
axisX(g, x, PH, { ticks: 5, fmt: d3.format('.2f'),
                  label: 'GBD disability weight assigned to manganese neurodevelopmental impairment' });

// the two additive rules, which do not depend on this weight at all
[[D.dose_add, 'dose-addition (central estimate)', WARM_DEEP, null],
 [D.resp_add, 'response-addition', TEAL_DEEP, '4,2.5']].forEach(([v, lab, col, dash]) => {
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(v)).attr('y2', y(v))
    .attr('stroke', col).attr('stroke-width', 1.4).attr('stroke-dasharray', dash);
  g.append('text').attr('x', PW + 6).attr('y', y(v) + 3).attr('font-size', 7.4)
    .attr('fill', col).attr('font-weight', 700).text(`${v.toFixed(2)}×`);
  g.append('text').attr('x', PW + 32).attr('y', y(v) + 3).attr('font-size', 7.0)
    .attr('fill', col).text(lab);
});

// the sweep itself
g.append('path').attr('d', d3.line().x(d => x(d.dw)).y(d => y(d.ratio))
  .curve(d3.curveMonotoneX)(D.sweep))
  .attr('fill', 'none').attr('stroke', INK).attr('stroke-width', 2.4);

// where the contaminant-specific model overtakes dose-addition
if (D.crossover != null) {
  g.append('rect').attr('x', 0).attr('width', x(D.crossover)).attr('y', 0).attr('height', PH)
    .attr('fill', WARM).attr('opacity', 0.06);
  g.append('line').attr('x1', x(D.crossover)).attr('x2', x(D.crossover)).attr('y2', PH)
    .attr('stroke', GREY).attr('stroke-width', 0.9).attr('stroke-dasharray', '2,2');
  note(g, `dose-addition is the higher estimate below ${D.crossover.toFixed(2)}`,
       4, 12, { size: 7, fill: GREY });
}

// GBD anchors
D.anchors.forEach(a => {
  const pt = D.sweep.reduce((p, q) => Math.abs(q.dw - a.dw) < Math.abs(p.dw - a.dw) ? q : p);
  const used = a.note === 'used here';
  g.append('line').attr('x1', x(a.dw)).attr('x2', x(a.dw))
    .attr('y1', y(pt.ratio)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.7).attr('stroke-dasharray', '2,2');
  g.append('circle').attr('cx', x(a.dw)).attr('cy', y(pt.ratio)).attr('r', used ? 4.2 : 3)
    .attr('fill', used ? WARM_DEEP : PAPER).attr('stroke', used ? PAPER : INK)
    .attr('stroke-width', used ? 1.4 : 1.1);
    // labels go left of the curve, which rises to the right, so they sit in empty space
  g.append('text').attr('x', x(a.dw) - 7).attr('y', y(pt.ratio) - 7)
    .attr('text-anchor', 'end')
    .attr('font-size', 7.6).attr('font-weight', used ? 700 : 400)
    .attr('fill', used ? WARM_DEEP : INK).text(`${pt.ratio.toFixed(2)}×`);
});

// annotate the two ends in words
const lo = D.sweep[0], hi = D.sweep[D.sweep.length - 1];
const key = g.append('g').attr('transform', `translate(${PW + 6},${PH - 26})`);
key.append('text').attr('x', 0).attr('y', -12).attr('font-size', 6.8)
  .attr('font-weight', 700).attr('fill', INK).text('GBD 2019 candidate endpoints');
D.anchors.forEach((a, i) => {
  key.append('text').attr('x', 0).attr('y', i * 16).attr('font-size', 6.8)
    .attr('font-weight', a.note === 'used here' ? 700 : 400)
    .attr('fill', a.note === 'used here' ? WARM_DEEP : INK).text(a.dw.toFixed(3));
  key.append('text').attr('x', 22).attr('y', i * 16).attr('font-size', 6.6).attr('fill', GREY)
    .text(a.label.length > 30 ? a.label.slice(0, 29) + '…' : a.label);
  if (a.note) key.append('text').attr('x', 22).attr('y', i * 16 + 7.5)
    .attr('font-size', 6.2).attr('fill', WARM_DEEP).attr('font-style', 'italic').text(a.note);
});
note(svg, `Across the full plausible range of the weight the ratio spans ${lo.ratio.toFixed(2)}× to ${hi.ratio.toFixed(2)}×, and the`,
     M.l, PH + M.t + 44, { size: 7.4, fill: INK });
note(svg, 'dose-addition central estimate sits inside it, not at its upper end.',
     M.l, PH + M.t + 55, { size: 7.4, fill: INK });
console.log(save(body, 's13'));
