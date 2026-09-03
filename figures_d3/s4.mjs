// S4 - GRACE / GRACE-FO satellite gravimetry, three panels.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { shortZone } from './sikit.mjs';
import { warmScale, WARM, WARM_DEEP, TEAL, TEAL_DEEP, TEAL_LIGHT, INK, GREY, MUTE, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s4.json'));
// derived, not hard-coded: three panels plus gutters must fit inside W2
const GX = 50, PH = 132, M = { t: 28, l: 56, r: 16 };
const PW = Math.floor((W2 - M.l - 2 * GX - M.r) / 3);
const { body, svg } = canvas(W2, PH + M.t + 68);
const panel = c => svg.append('g').attr('transform', `translate(${M.l + c * (PW + GX)},${M.t})`);
const zones = [...new Set(D.grace_zone.map(d => d.phys_zone))];
const zc = d3.scaleOrdinal().domain(zones)
  .range([WARM_DEEP, WARM, '#f2946a', TEAL_DEEP, TEAL, TEAL_LIGHT, MUTE]);
const parse = s => new Date(String(s).slice(0, 10));

// (A) GRACEDADM groundwater storage percentile
{
  const g = panel(0);
  const ts = D.grace_ts.map(d => ({ ...d, t: parse(d.date) })).filter(d => !isNaN(d.t));
  const x = d3.scaleTime().domain(d3.extent(ts, d => d.t)).range([0, PW]);
  const y = d3.scaleLinear().domain([0, 100]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'storage percentile' });
  axisX(g, x, PH, { ticks: 4, fmt: d3.timeFormat('%Y'), label: '' });
  zones.forEach(z => {
    const r = ts.filter(d => d.phys_zone === z).sort((a, b) => a.t - b.t);
    if (r.length < 5) return;
    g.append('path').attr('d', d3.line().x(d => x(d.t)).y(d => y(d.gws_percentile_mean))
      .curve(d3.curveBasis)(r))
      .attr('fill', 'none').attr('stroke', zc(z)).attr('stroke-width', 1).attr('opacity', 0.85);
  });
  note(g, 'GRACEDADM, 0·25° weekly, 2018–2024', 0, PH + 26, { size: 6.8, fill: GREY });
  tag(g, 'A');
}
// (B) TELLUS liquid-water-equivalent anomaly
{
  const g = panel(1);
  const ts = D.tellus_ts.map(d => ({ ...d, t: parse(d.date) })).filter(d => !isNaN(d.t));
  const x = d3.scaleTime().domain(d3.extent(ts, d => d.t)).range([0, PW]);
  const y = d3.scaleLinear().domain(d3.extent(ts, d => d.lwe_cm)).nice().range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'LWE anomaly (cm)' });
  axisX(g, x, PH, { ticks: 4, fmt: d3.timeFormat('%Y'), label: '' });
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', INK).attr('stroke-width', 0.7).attr('stroke-dasharray', '3,2');
  zones.forEach(z => {
    const r = ts.filter(d => d.phys_zone === z).sort((a, b) => a.t - b.t);
    if (r.length < 5) return;
    g.append('path').attr('d', d3.line().x(d => x(d.t)).y(d => y(d.lwe_cm)).curve(d3.curveBasis)(r))
      .attr('fill', 'none').attr('stroke', zc(z)).attr('stroke-width', 0.9).attr('opacity', 0.8);
  });
  note(g, 'TELLUS JPL RL06.3 mascon, 2002–2026', 0, PH + 26, { size: 6.8, fill: GREY });
  tag(g, 'B');
}
// (C) storage trend against zone median arsenic
{
  const g = panel(2);
  const r = D.tellus_as;
  const x = d3.scaleLinear().domain(d3.extent(r, d => d.trend_cm_per_yr)).nice().range([0, PW]);
  const y = d3.scaleLinear().domain([0, d3.max(r, d => d.as_median) * 1.15]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, label: 'zone median arsenic (µg/L)' });
  axisX(g, x, PH, { ticks: 4, fmt: d3.format('.1f'), label: 'LWE trend (cm/yr)' });
  g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.7).attr('stroke-dasharray', '3,2');
  r.forEach(d => {
    g.append('circle').attr('cx', x(d.trend_cm_per_yr)).attr('cy', y(d.as_median)).attr('r', 3.6)
      .attr('fill', zc(d.phys_zone)).attr('stroke', PAPER).attr('stroke-width', 1);
    g.append('text').attr('x', x(d.trend_cm_per_yr) + 5).attr('y', y(d.as_median) + 2.4)
      .attr('font-size', 6).attr('fill', INK).text(shortZone(d.phys_zone));
  });
  const grid = D.grid.find(d => /well/i.test(String(d.level)) && /TELLUS/i.test(String(d.product)));
  if (grid) note(g, `well-level ρ = ${grid.spearman_rho.toFixed(2)}, n = ${grid.n}`, 2, 9, { size: 6.6, fill: INK });
  note(g, 'depletion does not track arsenic across zones', 0, PH + 26, { size: 6.8, fill: GREY });
  tag(g, 'C');
}
console.log(save(body, 's4'));
