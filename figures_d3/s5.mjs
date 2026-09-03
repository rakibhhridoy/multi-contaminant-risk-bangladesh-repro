// S5 - partial-information test. Blocked validation is primary; random folds shown for comparison.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s5.json'));
const PW = 180, PH = 150, M = { t: 30, l: 58 }, GX = 74;
const { body, svg } = canvas(W2, PH + M.t + 62);

// (a) AUC before and after adding phosphate, both schemes
{
  const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
  const rows = [...D.blocked.map(d => ({ ...d, scheme: 'blocked' })),
                ...D.random.map(d => ({ ...d, scheme: 'random' }))];
  const x = d3.scaleBand().domain(rows.map(d => `${d.scheme}|${d.thr}`)).range([0, PW]).padding(0.42);
  const y = d3.scaleLinear().domain([0.5, 0.85]).range([PH, 0]);
  axisY(g, y, { ticks: 4, w: PW, fmt: d3.format('.2f'), label: 'ROC-AUC' });
  rows.forEach(d => {
    const cx = x(`${d.scheme}|${d.thr}`) + x.bandwidth() / 2;
    const col = d.scheme === 'blocked' ? WARM_DEEP : MUTE;
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d.base)).attr('y2', y(d.full))
      .attr('stroke', col).attr('stroke-width', 2);
    g.append('circle').attr('cx', cx).attr('cy', y(d.base)).attr('r', 3)
      .attr('fill', PAPER).attr('stroke', col).attr('stroke-width', 1.6);
    g.append('circle').attr('cx', cx).attr('cy', y(d.full)).attr('r', 3.4).attr('fill', col);
    g.append('text').attr('x', cx).attr('y', y(d.full) - 6).attr('text-anchor', 'middle')
      .attr('font-size', 7).attr('fill', col).text(`+${d.d.toFixed(3)}`);
    g.append('text').attr('x', cx).attr('y', PH + 11).attr('text-anchor', 'middle')
      .attr('font-size', 7).text(d.thr.replace('WHO 10', 'As>10').replace('saddle 22', 'As>22'));
    g.append('text').attr('x', cx).attr('y', PH + 21).attr('text-anchor', 'middle')
      .attr('font-size', 6.6).attr('fill', GREY).text(d.scheme);
  });
  const lg = g.append('g').attr('transform', 'translate(4,2)');
  [['base redox proxies', PAPER, WARM_DEEP], ['+ dissolved phosphate', WARM_DEEP, WARM_DEEP]].forEach((d, i) => {
    lg.append('circle').attr('cx', 5).attr('cy', i * 11).attr('r', 3.2)
      .attr('fill', d[1]).attr('stroke', d[2]).attr('stroke-width', 1.4);
    lg.append('text').attr('x', 13).attr('y', i * 11 + 3).attr('font-size', 7.4).text(d[0]);
  });
  tag(g, 'a');
}
// (b) block-bootstrap dAUC with interval
{
  const g = svg.append('g').attr('transform', `translate(${M.l + PW + GX},${M.t})`);
  const y = d3.scaleBand().domain(D.blocked.map(d => d.thr)).range([0, PH]).padding(0.62);
  const x = d3.scaleLinear().domain([0, 0.15]).range([0, PW]);
  axisX(g, x, PH, { ticks: 4, fmt: d3.format('.2f'), label: 'ΔAUC on adding phosphate' });
  g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.9);
  D.blocked.forEach(d => {
    const cy = y(d.thr) + y.bandwidth() / 2;
    g.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi)).attr('y1', cy).attr('y2', cy)
      .attr('stroke', WARM_DEEP).attr('stroke-width', 2);
    [d.lo, d.hi].forEach(v => g.append('line').attr('x1', x(v)).attr('x2', x(v))
      .attr('y1', cy - 5).attr('y2', cy + 5).attr('stroke', WARM_DEEP).attr('stroke-width', 1.4));
    g.append('circle').attr('cx', x(d.d)).attr('cy', cy).attr('r', 4)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.4);
    g.append('text').attr('x', -5).attr('y', cy + 2.6).attr('text-anchor', 'end').attr('font-size', 7.4)
      .text(d.thr.replace('WHO 10', 'As > 10 µg/L').replace('saddle 22', 'As > 22 µg/L'));
    g.append('text').attr('x', x(d.d)).attr('y', cy - 10).attr('text-anchor', 'middle')
      .attr('font-size', 6.8).attr('fill', GREY).text(`${d.pgt0}% of replicates > 0`);
  });
  note(g, '95% uncertainty interval, block bootstrap over 0·5° grid blocks',
       0, PH + 34, { size: 7.2, fill: GREY });
  tag(g, 'b');
}
console.log(save(body, 's5'));
