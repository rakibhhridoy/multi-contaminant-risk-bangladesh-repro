// S9 - spatial cross-validation of the IDW exposure surface.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { WARM, WARM_DEEP, TEAL_DEEP, INK, GREY, PAPER, MUTE } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s9.json'));
const short = z => z.replace(/_/g, ' ').replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
                    .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');
const PW = 200, PH = 155, GX = 96, M = { t: 30, l: 62 };
const { body, svg } = canvas(W2, PH + M.t + 60);
// (a) observed against predicted zone median HI
{
  const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
  const mx = d3.max(D.zone, d => Math.max(d.obs_med, d.pred_med)) * 1.12;
  const x = d3.scaleLinear().domain([0, mx]).range([0, PW]);
  const y = d3.scaleLinear().domain([0, mx]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'predicted median HI' });
  axisX(g, x, PH, { ticks: 5, label: 'observed median HI' });
  g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(mx)).attr('y2', y(mx))
    .attr('stroke', GREY).attr('stroke-width', 0.9).attr('stroke-dasharray', '4,2');
  D.zone.forEach(d => {
    g.append('circle').attr('cx', x(d.obs_med)).attr('cy', y(d.pred_med)).attr('r', 4)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.2);
    g.append('text').attr('x', x(d.obs_med) + 6).attr('y', y(d.pred_med) + 2.6)
      .attr('font-size', 6.6).attr('fill', INK).text(short(d.zone));
  });
  note(g, '1:1', x(mx * 0.86), y(mx * 0.94), { size: 7, fill: GREY });
  const mae = d3.mean(D.zone, d => d.abs_pct_err);
  note(g, `mean absolute error ${mae.toFixed(1)}% across the seven zones`, 4, PH + 40, { size: 7.2, fill: GREY });
  tag(g, 'a');
}
// (b) Skill by validation scheme, reported as rank correlation.
// R2 is negative for every point-level scheme, i.e. the surface predicts an
// individual well's hazard index worse than the mean does. Plotting negative R2
// bars would read as a broken chart, so rank correlation carries the panel and
// R2 is stated alongside. The contrast is the point of the figure: the surface
// has little point-level skill but recovers the per-zone median well, which is
// the quantity the burden calculation actually consumes.
{
  const g = svg.append('g').attr('transform', `translate(${M.l + PW + GX},${M.t})`);
  const R = D.cv.filter(d => d.space !== 'log10_HI');
  const nm = s => s.replace('random_10fold', 'random 10-fold, per well')
                   .replace('spatial_block_0.5deg', 'spatial block 0·5°, per well')
                   .replace('zone_median_recovery_blockCV', 'zone median, block CV');
  const y = d3.scaleBand().domain(R.map(d => d.scheme)).range([0, PH]).padding(0.42);
  const x = d3.scaleLinear().domain([0, 1]).range([0, PW]);
  axisX(g, x, PH, { ticks: 5, fmt: d3.format('.1f'), label: "Spearman's ρ, observed against predicted" });
  R.forEach(d => {
    const isZone = /zone_median/.test(d.scheme);
    g.append('rect').attr('x', 0).attr('y', y(d.scheme)).attr('height', y.bandwidth())
      .attr('width', Math.max(1, x(d.spearman))).attr('fill', isZone ? WARM_DEEP : MUTE).attr('rx', 1.5);
    g.append('text').attr('x', -6).attr('y', y(d.scheme) + y.bandwidth() / 2 + 2.6)
      .attr('text-anchor', 'end').attr('font-size', 7.2).text(nm(d.scheme));
    g.append('text').attr('x', x(d.spearman) + 4).attr('y', y(d.scheme) + y.bandwidth() / 2 + 2.6)
      .attr('font-size', 7).attr('fill', GREY)
      .text(`ρ ${d.spearman.toFixed(2)}` + (d.R2 != null ? `,  R² ${d.R2.toFixed(2)}` : ''));
  });
  note(g, 'point-level R² is at or below zero: the surface does not predict an individual well,',
       0, PH + 40, { size: 7.2, fill: GREY });
  note(g, 'but it recovers the per-zone median that the burden calculation consumes',
       0, PH + 51, { size: 7.2, fill: GREY });
  tag(g, 'b');
}
console.log(save(body, 's9'));
