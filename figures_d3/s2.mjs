// S2 - Monte Carlo uncertainty propagation, six panels.
import { readFileSync } from 'fs';
import { save, tag, axisX, axisY, note, d3 } from './svgkit.mjs';
import { grid6, shortZone } from './sikit.mjs';
import { warmScale, WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s2.json'));
const { body, svg, panel, PW, PH } = grid6({ ph: 116 });
const SSP = { 'SSP2-4.5': TEAL_DEEP, 'SSP5-8.5': WARM_DEEP };
const nat = D.national;

// (a) fan chart through 2050
{
  const g = panel(0, 0);
  const yrs = [...new Set(nat.map(d => d.year))].sort();
  const x = d3.scaleLinear().domain(d3.extent(yrs)).range([0, PW]);
  const y = d3.scaleLinear().domain([0, d3.max(nat, d => d['HI_p97.5']) * 1.05]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'projected HI' });
  axisX(g, x, PH, { values: yrs, fmt: d3.format('d'), label: '' });
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(1)).attr('y2', y(1))
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2');
  Object.keys(SSP).forEach(s => {
    const r = nat.filter(d => d.ssp === s).sort((a, b) => a.year - b.year);
    g.append('path').attr('d', d3.area().x(d => x(d.year)).y0(d => y(d['HI_p2.5'])).y1(d => y(d['HI_p97.5']))
      .curve(d3.curveMonotoneX)(r)).attr('fill', SSP[s]).attr('opacity', 0.16);
    g.append('path').attr('d', d3.line().x(d => x(d.year)).y(d => y(d.HI_median)).curve(d3.curveMonotoneX)(r))
      .attr('fill', 'none').attr('stroke', SSP[s]).attr('stroke-width', 2);
  });
  const lg = g.append('g').attr('transform', 'translate(4,2)');
  Object.entries(SSP).forEach(([s, c], i) => {
    lg.append('line').attr('x1', 0).attr('x2', 13).attr('y1', i * 10).attr('y2', i * 10)
      .attr('stroke', c).attr('stroke-width', 2);
    lg.append('text').attr('x', 17).attr('y', i * 10 + 3).attr('font-size', 6.8).text(s);
  });
  note(g, 'band = 95% uncertainty interval', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'a');
}
// (b) HI distribution at 2050, from the percentile summary
{
  const g = panel(0, 1);
  const r = nat.filter(d => d.year === 2050);
  const x = d3.scaleLinear().domain([0, d3.max(r, d => d['HI_p97.5']) * 1.05]).range([0, PW]);
  const y = d3.scaleBand().domain(r.map(d => d.ssp)).range([0, PH]).padding(0.55);
  axisX(g, x, PH, { ticks: 5, label: 'projected HI at 2050' });
  g.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.9).attr('stroke-dasharray', '3,2');
  r.forEach(d => {
    const yy = y(d.ssp), h = y.bandwidth(), c = SSP[d.ssp] || WARM;
    g.append('line').attr('x1', x(d['HI_p2.5'])).attr('x2', x(d['HI_p97.5']))
      .attr('y1', yy + h / 2).attr('y2', yy + h / 2).attr('stroke', c).attr('stroke-width', 1.2);
    g.append('rect').attr('x', x(d.HI_p25)).attr('width', x(d.HI_p75) - x(d.HI_p25))
      .attr('y', yy).attr('height', h).attr('fill', c).attr('opacity', 0.35).attr('rx', 1);
    g.append('line').attr('x1', x(d.HI_median)).attr('x2', x(d.HI_median))
      .attr('y1', yy).attr('y2', yy + h).attr('stroke', c).attr('stroke-width', 2);
    g.append('text').attr('x', -4).attr('y', yy + h / 2 + 2.4).attr('text-anchor', 'end')
      .attr('font-size', 6.8).text(d.ssp);
    g.append('text').attr('x', x(d['HI_p97.5']) + 4).attr('y', yy + h / 2 + 2.4)
      .attr('font-size', 6.4).attr('fill', GREY).text(`P(HI>1) = ${d.HI_exceed_pct.toFixed(0)}%`);
  });
  note(g, 'box = IQR, line = median, whisker = 95% interval', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'b');
}
// (c) variance decomposition
{
  const g = panel(0, 2);
  const V = D.variance;
  const y = d3.scaleBand().domain(V.map(d => d.source)).range([0, PH]).padding(0.34);
  const x = d3.scaleLinear().domain([0, 50]).range([0, PW]);
  axisX(g, x, PH, { ticks: 3, fmt: v => `${v}%`, label: 'variance reduction when fixed' });
  V.forEach(d => {
    const cool = /climate|exposure/.test(d.source);
    g.append('rect').attr('x', 0).attr('y', y(d.source)).attr('height', y.bandwidth())
      .attr('width', Math.max(1.5, x(d.pct))).attr('fill', cool ? TEAL : WARM).attr('rx', 1);
    g.append('text').attr('x', 0).attr('y', y(d.source) - 3).attr('font-size', 6.6).text(d.source);
    g.append('text').attr('x', Math.max(1.5, x(d.pct)) + 3).attr('y', y(d.source) + y.bandwidth() / 2 + 2.4)
      .attr('font-size', 6.4).attr('fill', GREY).text(`${d.lt ? '<' : ''}${d.pct}%`);
  });
  tag(g, 'c');
}
// (d) zone-level uncertainty at 2050, SSP5-8.5
{
  const g = panel(1, 0);
  const r = D.zones.filter(d => d.year === 2050 && d.ssp === 'SSP5-8.5')
                   .sort((a, b) => b.HI_median - a.HI_median);
  const x = d3.scaleLinear().domain([0, d3.max(r, d => d['HI_p97.5']) * 1.05]).range([0, PW]);
  const y = d3.scaleBand().domain(r.map(d => d.phys_zone)).range([0, PH]).padding(0.42);
  axisX(g, x, PH, { ticks: 4, label: 'projected HI, SSP5-8.5' });
  g.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2');
  r.forEach(d => {
    const cy = y(d.phys_zone) + y.bandwidth() / 2;
    g.append('line').attr('x1', x(d['HI_p2.5'])).attr('x2', x(d['HI_p97.5'])).attr('y1', cy).attr('y2', cy)
      .attr('stroke', MUTE).attr('stroke-width', 1.4);
    g.append('circle').attr('cx', x(d.HI_median)).attr('cy', cy).attr('r', 2.8)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1);
    g.append('text').attr('x', -4).attr('y', cy + 2.4).attr('text-anchor', 'end')
      .attr('font-size', 6.4).text(shortZone(d.phys_zone));
  });
  tag(g, 'd');
}
// (e) pathway overlay at 2050
{
  const g = panel(1, 1);
  const r = nat.filter(d => d.year === 2050);
  const x = d3.scaleBand().domain(r.map(d => d.ssp)).range([0, PW]).padding(0.5);
  const y = d3.scaleLinear().domain([0, d3.max(r, d => d['HI_p97.5']) * 1.05]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'projected HI at 2050' });
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(1)).attr('y2', y(1))
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2');
  r.forEach(d => {
    const cx = x(d.ssp) + x.bandwidth() / 2, c = SSP[d.ssp] || WARM;
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d['HI_p2.5'])).attr('y2', y(d['HI_p97.5']))
      .attr('stroke', c).attr('stroke-width', 2);
    g.append('rect').attr('x', cx - 12).attr('width', 24).attr('y', y(d.HI_p75))
      .attr('height', y(d.HI_p25) - y(d.HI_p75)).attr('fill', c).attr('opacity', 0.3).attr('rx', 1);
    g.append('circle').attr('cx', cx).attr('cy', y(d.HI_median)).attr('r', 3.4)
      .attr('fill', c).attr('stroke', PAPER).attr('stroke-width', 1.2);
    g.append('text').attr('x', cx).attr('y', PH + 11).attr('text-anchor', 'middle')
      .attr('font-size', 6.8).text(d.ssp);
    g.append('text').attr('x', cx).attr('y', y(d.HI_median) - 7).attr('text-anchor', 'middle')
      .attr('font-size', 6.6).attr('font-weight', 700).attr('fill', c).text(d.HI_median.toFixed(2));
  });
  note(g, 'both pathways exceed HI = 1 throughout', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'e');
}
// (f) depth-stratified hazard index
{
  const g = panel(1, 2);
  const order = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep'];
  const r = order.map(z => D.depth.find(d => d.depth_zone === z)).filter(Boolean);
  const x = d3.scaleLinear().domain([0, d3.max(r, d => d.hi) * 1.05]).range([0, PW]);
  const y = d3.scaleBand().domain(r.map(d => d.depth_zone)).range([0, PH]).padding(0.42);
  axisX(g, x, PH, { ticks: 4, label: 'cumulative HI' });
  g.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2');
  r.forEach(d => {
    const cy = y(d.depth_zone) + y.bandwidth() / 2;
    g.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi)).attr('y1', cy).attr('y2', cy)
      .attr('stroke', MUTE).attr('stroke-width', 1.4);
    g.append('circle').attr('cx', x(d.med)).attr('cy', cy).attr('r', 3)
      .attr('fill', warmScale(d.exceed / 100)).attr('stroke', INK).attr('stroke-width', 0.7);
    g.append('text').attr('x', -4).attr('y', cy + 2.4).attr('text-anchor', 'end')
      .attr('font-size', 6.6).text(d.depth_zone.replace('Medium_Deep', 'Med-deep'));
    g.append('text').attr('x', x(d.hi) + 3).attr('y', cy + 2.4).attr('font-size', 6.2)
      .attr('fill', GREY).text(`${d.exceed.toFixed(0)}% > 1`);
  });
  note(g, 'shallow aquifers carry the highest index', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'f');
}
console.log(save(body, 's2'));
