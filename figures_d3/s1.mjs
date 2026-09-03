// S1 - copula joint exceedance analysis, six panels as the caption describes.
import { readFileSync } from 'fs';
import { save, tag, axisX, axisY, note, d3 } from './svgkit.mjs';
import { grid6, shortZone } from './sikit.mjs';
import { CONTAM, warmScale, WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s1.json'));
const LAB = { 'As': 'As', 'Mn2+': 'Mn', 'Fe2+': 'Fe', 'Cr3+': 'Cr' };
const { body, svg, panel, PW, PH } = grid6({ ph: 116 });

// (a) As against Fe with copula dependence annotated
{
  const g = panel(0, 0);
  const x = d3.scaleLog().domain([0.05, 900]).range([0, PW]).clamp(true);
  const y = d3.scaleLog().domain([0.005, 40]).range([PH, 0]).clamp(true);
  axisY(g, y, { values: [0.01, 1, 10], w: PW, fmt: d3.format('~g'), label: 'iron (mg/L)' });
  axisX(g, x, PH, { values: [0.1, 10, 100], fmt: d3.format('~g'), label: 'arsenic (µg/L)' });
  g.selectAll('circle').data(D.scatter).join('circle')
    .attr('cx', d => x(d.x)).attr('cy', d => y(d.y)).attr('r', 1.4)
    .attr('fill', WARM_DEEP).attr('opacity', 0.34);
  const tau = D.pairs.find(p => p.pair === 'As-Fe2+');
  if (tau) note(g, `Frank, τ = ${tau.kendall_tau.toFixed(3)}`, 4, 9, { size: 7, fill: INK });
  tag(g, 'a');
}
// (b) individual and joint exceedance against current WHO values
{
  const g = panel(0, 1);
  const rows = [...D.individual.map(d => ({ k: LAB[d.c], v: d.frac, warm: true })),
                { k: '≥2', v: D.ge2, warm: true, hi: true }];
  const x = d3.scaleBand().domain(rows.map(d => d.k)).range([0, PW]).padding(0.32);
  const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'samples exceeding' });
  rows.forEach(d => {
    g.append('rect').attr('x', x(d.k)).attr('width', x.bandwidth())
      .attr('y', y(d.v)).attr('height', PH - y(d.v))
      .attr('fill', d.hi ? WARM_DEEP : warmScale(d.v)).attr('rx', 1);
    g.append('text').attr('x', x(d.k) + x.bandwidth() / 2).attr('y', y(d.v) - 3)
      .attr('text-anchor', 'middle').attr('font-size', 6.6).text(`${(d.v * 100).toFixed(0)}%`);
    g.append('text').attr('x', x(d.k) + x.bandwidth() / 2).attr('y', PH + 11)
      .attr('text-anchor', 'middle').attr('font-size', 7).text(d.k);
  });
  note(g, 'current WHO health-based values', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'b');
}
// (c) copula family selection
{
  const g = panel(0, 2);
  const fam = d3.rollups(D.pairs, v => v.length, d => d.best_copula).sort((a, b) => b[1] - a[1]);
  const tot = d3.sum(fam, d => d[1]);
  const x = d3.scaleBand().domain(fam.map(d => d[0])).range([0, PW]).padding(0.42);
  const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'of pooled pairs' });
  fam.forEach(([f, n]) => {
    g.append('rect').attr('x', x(f)).attr('width', x.bandwidth())
      .attr('y', y(n / tot)).attr('height', PH - y(n / tot))
      .attr('fill', f === 'Frank' ? WARM : TEAL).attr('rx', 1);
    g.append('text').attr('x', x(f) + x.bandwidth() / 2).attr('y', y(n / tot) - 3)
      .attr('text-anchor', 'middle').attr('font-size', 6.8).text(`${n}/${tot}`);
    g.append('text').attr('x', x(f) + x.bandwidth() / 2).attr('y', PH + 11)
      .attr('text-anchor', 'middle').attr('font-size', 7).text(f);
  });
  note(g, 'symmetric dependence, no strong tail', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'c');
}
// (d) distribution of simultaneous exceedance counts
{
  const g = panel(1, 0);
  const x = d3.scaleBand().domain(D.counts.map(d => String(d.k))).range([0, PW]).padding(0.3);
  const y = d3.scaleLinear().domain([0, d3.max(D.counts, d => d.frac) * 1.15]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'of samples' });
  D.counts.forEach(d => {
    g.append('rect').attr('x', x(String(d.k))).attr('width', x.bandwidth())
      .attr('y', y(d.frac)).attr('height', PH - y(d.frac))
      .attr('fill', d.k >= 2 ? WARM_DEEP : MUTE).attr('rx', 1);
    g.append('text').attr('x', x(String(d.k)) + x.bandwidth() / 2).attr('y', y(d.frac) - 3)
      .attr('text-anchor', 'middle').attr('font-size', 6.6).text(`${(d.frac * 100).toFixed(0)}%`);
    g.append('text').attr('x', x(String(d.k)) + x.bandwidth() / 2).attr('y', PH + 11)
      .attr('text-anchor', 'middle').attr('font-size', 7).text(d.k);
  });
  note(g, 'contaminants exceeding simultaneously', 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'd');
}
// (e) Kendall tau matrix
{
  const g = panel(1, 1);
  const C = ['As', 'Mn2+', 'Fe2+', 'Cr3+'];
  const M = new Map();
  D.pairs.forEach(p => { const [a, b] = p.pair.split('-'); M.set(`${a}|${b}`, p.kendall_tau); M.set(`${b}|${a}`, p.kendall_tau); });
  const cell = Math.min(PW, PH) / C.length, off = (PW - cell * C.length) / 2;
  const mx = d3.max(D.pairs, d => Math.abs(d.kendall_tau));
  C.forEach((a, i) => C.forEach((b, j) => {
    const v = a === b ? null : M.get(`${a}|${b}`);
    g.append('rect').attr('x', off + j * cell).attr('y', i * cell)
      .attr('width', cell - 1).attr('height', cell - 1)
      .attr('fill', v == null ? '#f2f2f2' : warmScale(Math.abs(v) / mx)).attr('rx', 1);
    if (v != null) g.append('text').attr('x', off + j * cell + cell / 2).attr('y', i * cell + cell / 2 + 2.4)
      .attr('text-anchor', 'middle').attr('font-size', 6.2)
      .attr('fill', Math.abs(v) / mx > 0.62 ? PAPER : INK).text(v.toFixed(3));
  }));
  C.forEach((k, i) => {
    g.append('text').attr('x', off - 4).attr('y', i * cell + cell / 2 + 2.4).attr('text-anchor', 'end')
      .attr('font-size', 7).text(LAB[k]);
    g.append('text').attr('x', off + i * cell + cell / 2).attr('y', PH + 10)
      .attr('text-anchor', 'middle').attr('font-size', 7).text(LAB[k]);
  });
  note(g, "Kendall's τ", 0, PH + 24, { size: 6.8, fill: GREY });
  tag(g, 'e');
}
// (f) upper tail dependence
{
  const g = panel(1, 2);
  const R = D.tail.map(d => ({ pair: d.pair.replace(/2\+|3\+/g, ''), u: d['lambda_U_0.95'] }))
                  .sort((a, b) => b.u - a.u);
  const y = d3.scaleBand().domain(R.map(d => d.pair)).range([0, PH]).padding(0.32);
  const x = d3.scaleLinear().domain([0, d3.max(R, d => d.u) * 1.2 || 0.1]).range([0, PW]);
  axisX(g, x, PH, { ticks: 3, fmt: d3.format('.2f'), label: 'λ_U at the 95th quantile' });
  R.forEach(d => {
    g.append('rect').attr('x', 0).attr('y', y(d.pair)).attr('height', y.bandwidth())
      .attr('width', Math.max(0.6, x(d.u))).attr('fill', WARM).attr('rx', 1);
    g.append('text').attr('x', -4).attr('y', y(d.pair) + y.bandwidth() / 2 + 2.4)
      .attr('text-anchor', 'end').attr('font-size', 6.6).text(d.pair);
    g.append('text').attr('x', Math.max(0.6, x(d.u)) + 3).attr('y', y(d.pair) + y.bandwidth() / 2 + 2.4)
      .attr('font-size', 6.4).attr('fill', GREY).text(d.u.toFixed(3));
  });
  tag(g, 'f');
}
console.log(save(body, 's1'));
