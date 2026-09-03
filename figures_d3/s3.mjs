// S3 - counterfactual intervention analysis, six panels.
import { readFileSync } from 'fs';
import { save, tag, axisX, axisY, note, d3 } from './svgkit.mjs';
import { grid6, shortZone, shortScen } from './sikit.mjs';
import { warmScale, WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s3.json'));
const { body, svg, panel, PW, PH } = grid6({ ph: 116, left: 60 });
const SC = D.scenarios.slice().sort((a, b) => b.annual_DALY_averted - a.annual_DALY_averted);
const nm = s => shortScen(s);

// (a) DALYs averted
{
  const g = panel(0, 0);
  const x = d3.scaleBand().domain(SC.map(d => nm(d.scenario))).range([0, PW]).padding(0.32);
  const y = d3.scaleLinear().domain([0, d3.max(SC, d => d.annual_DALY_averted) * 1.16]).range([PH, 0]);
  axisY(g, y, { ticks: 4, w: PW, fmt: v => `${(v / 1000).toFixed(0)}k`, label: 'annual DALYs averted' });
  SC.forEach(d => {
    g.append('rect').attr('x', x(nm(d.scenario))).attr('width', x.bandwidth())
      .attr('y', y(d.annual_DALY_averted)).attr('height', PH - y(d.annual_DALY_averted))
      .attr('fill', warmScale(d.pct_reduction / 100)).attr('rx', 1);
    g.append('text').attr('x', x(nm(d.scenario)) + x.bandwidth() / 2).attr('y', y(d.annual_DALY_averted) - 3)
      .attr('text-anchor', 'middle').attr('font-size', 6.2).text(`${d.pct_reduction.toFixed(0)}%`);
    g.append('text').attr('x', x(nm(d.scenario)) + x.bandwidth() / 2).attr('y', PH + 11)
      .attr('text-anchor', 'middle').attr('font-size', 7).text(nm(d.scenario));
  });
  note(g, 'label = share of baseline burden averted', 0, PH + 24, { size: 6.6, fill: GREY });
  tag(g, 'a');
}
// (b) equity heatmap
{
  const g = panel(0, 1);
  const zones = [...new Set(D.equity.map(d => d.phys_zone))];
  const scens = [...new Set(D.equity.map(d => d.scenario))].sort();
  const cw = PW / scens.length, ch = PH / zones.length;
  const M = new Map(D.equity.map(d => [`${d.phys_zone}|${d.scenario}`, d.pct_reduction]));
  zones.forEach((z, i) => scens.forEach((s, j) => {
    const v = M.get(`${z}|${s}`) ?? 0;
    g.append('rect').attr('x', j * cw).attr('y', i * ch).attr('width', cw - 1).attr('height', ch - 1)
      .attr('fill', warmScale(v / 100)).attr('rx', 1);
    g.append('text').attr('x', j * cw + cw / 2).attr('y', i * ch + ch / 2 + 2.2)
      .attr('text-anchor', 'middle').attr('font-size', 5.6)
      .attr('fill', v / 100 > 0.62 ? PAPER : INK).text(v.toFixed(0));
  }));
  zones.forEach((z, i) => g.append('text').attr('x', -4).attr('y', i * ch + ch / 2 + 2.2)
    .attr('text-anchor', 'end').attr('font-size', 6).text(shortZone(z)));
  scens.forEach((s, j) => g.append('text').attr('x', j * cw + cw / 2).attr('y', PH + 10)
    .attr('text-anchor', 'middle').attr('font-size', 6.6).text(nm(s)));
  note(g, '% burden reduction by zone', 0, PH + 24, { size: 6.6, fill: GREY });
  tag(g, 'b');
}
// (c) spread of zone-level reduction per scenario
{
  const g = panel(0, 2);
  const scens = [...new Set(D.equity.map(d => d.scenario))].sort();
  const x = d3.scaleBand().domain(scens.map(nm)).range([0, PW]).padding(0.42);
  const y = d3.scaleLinear().domain([0, 100]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: v => `${v}%`, label: 'zone burden reduction' });
  scens.forEach(s => {
    const v = D.equity.filter(d => d.scenario === s).map(d => d.pct_reduction).sort(d3.ascending);
    if (!v.length) return;
    const cx = x(nm(s)) + x.bandwidth() / 2, w = x.bandwidth();
    const q1 = d3.quantile(v, .25), md = d3.median(v), q3 = d3.quantile(v, .75);
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(v[0])).attr('y2', y(v[v.length - 1]))
      .attr('stroke', MUTE).attr('stroke-width', 1.2);
    g.append('rect').attr('x', cx - w / 2).attr('width', w).attr('y', y(q3)).attr('height', Math.max(1, y(q1) - y(q3)))
      .attr('fill', WARM).attr('opacity', 0.55).attr('rx', 1);
    g.append('line').attr('x1', cx - w / 2).attr('x2', cx + w / 2).attr('y1', y(md)).attr('y2', y(md))
      .attr('stroke', WARM_DEEP).attr('stroke-width', 1.8);
    g.append('text').attr('x', cx).attr('y', PH + 11).attr('text-anchor', 'middle')
      .attr('font-size', 7).text(nm(s));
  });
  note(g, 'box = IQR across the seven zones', 0, PH + 24, { size: 6.6, fill: GREY });
  tag(g, 'c');
}
// (d) cost-effectiveness acceptability curves
{
  const g = panel(1, 0);
  // Every scenario saturates below $900, so a linear axis to $3,000 compresses
  // the whole curve into the left edge. Log scale shows where each one turns.
  const x = d3.scaleLog().domain([10, 3000]).range([0, PW]);
  const y = d3.scaleLinear().domain([0, 1]).range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: d3.format('.0%'), label: 'probability cost-effective' });
  axisX(g, x, PH, { values: [10, 100, 1000, 3000], fmt: v => v >= 1000 ? `${v / 1000}k` : String(v), label: 'willingness to pay ($/DALY)' });
  g.append('line').attr('x1', x(D.gdp)).attr('x2', x(D.gdp)).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.9).attr('stroke-dasharray', '3,2');
  note(g, 'GDP per capita', x(D.gdp) - 3, 9, { anchor: 'end', size: 6.4, fill: INK });
  const cols = [WARM_DEEP, WARM, TEAL_DEEP, TEAL, MUTE];
  D.ceac.sort((a, b) => a.scenario.localeCompare(b.scenario)).forEach((c, i) => {
    g.append('path').attr('d', d3.line().x(d => x(Math.max(10, d.w))).y(d => y(d.p))(c.curve.filter(d => d.w >= 10)))
      .attr('fill', 'none').attr('stroke', cols[i % cols.length]).attr('stroke-width', 1.6);
  });
  const lg = g.append('g').attr('transform', `translate(${PW - 30},4)`);
  D.ceac.forEach((c, i) => {
    lg.append('line').attr('x1', 0).attr('x2', 10).attr('y1', i * 9).attr('y2', i * 9)
      .attr('stroke', cols[i % cols.length]).attr('stroke-width', 1.8);
    lg.append('text').attr('x', 13).attr('y', i * 9 + 2.6).attr('font-size', 6.2).text(nm(c.scenario));
  });
  tag(g, 'd');
}
// (e) ICER frontier
{
  const g = panel(1, 1);
  const x = d3.scaleLinear().domain([0, d3.max(SC, d => d.annual_DALY_averted) * 1.12]).range([0, PW]);
  // The WHO-CHOICE threshold ($2,500) is far above the highest ICER ($852). If the
  // domain stops at the data, the reference line lands off-panel and its label
  // prints into the panel above. Include the threshold: that every scenario sits
  // well below it is the panel's point.
  const y = d3.scaleLinear().domain([0, D.gdp * 1.08]).range([PH, 0]);
  axisY(g, y, { values: [0, 500, 1000, 1500, 2000, 2500], fmt: v => `$${v}`, w: PW, label: 'cost per DALY averted' });
  axisX(g, x, PH, { ticks: 4, fmt: v => `${(v / 1000).toFixed(0)}k`, label: 'annual DALYs averted' });
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(D.gdp)).attr('y2', y(D.gdp))
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2');
  note(g, `WHO-CHOICE $${D.gdp}`, PW - 2, y(D.gdp) - 4, { anchor: 'end', size: 6.4, fill: INK });
  SC.forEach(d => {
    g.append('circle').attr('cx', x(d.annual_DALY_averted)).attr('cy', y(d.ICER_usd_per_daly)).attr('r', 3.6)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.2);
    g.append('text').attr('x', x(d.annual_DALY_averted)).attr('y', y(d.ICER_usd_per_daly) - 7)
      .attr('text-anchor', 'middle').attr('font-size', 6.6).attr('font-weight', 700).text(nm(d.scenario));
    g.append('text').attr('x', x(d.annual_DALY_averted)).attr('y', y(d.ICER_usd_per_daly) + 11)
      .attr('text-anchor', 'middle').attr('font-size', 6).attr('fill', GREY)
      .text(`$${d.ICER_usd_per_daly.toFixed(0)}`);
  });
  note(g, 'lower and further right is better', 0, PH + 30, { size: 6.6, fill: GREY });
  tag(g, 'e');
}
// (f) where the most effective scenario acts
{
  const g = panel(1, 2);
  const r = D.equity.filter(d => d.scenario === 'S3_multi_treatment')
                    .sort((a, b) => b.pct_reduction - a.pct_reduction);
  const y = d3.scaleBand().domain(r.map(d => d.phys_zone)).range([0, PH]).padding(0.34);
  const x = d3.scaleLinear().domain([0, 100]).range([0, PW]);
  axisX(g, x, PH, { ticks: 4, fmt: v => `${v}%`, label: 'S3 burden reduction' });
  r.forEach(d => {
    g.append('rect').attr('x', 0).attr('y', y(d.phys_zone)).attr('height', y.bandwidth())
      .attr('width', Math.max(1, x(d.pct_reduction))).attr('fill', warmScale(d.pct_reduction / 100)).attr('rx', 1);
    g.append('text').attr('x', -4).attr('y', y(d.phys_zone) + y.bandwidth() / 2 + 2.2)
      .attr('text-anchor', 'end').attr('font-size', 6).text(shortZone(d.phys_zone));
    g.append('text').attr('x', Math.max(1, x(d.pct_reduction)) + 3).attr('y', y(d.phys_zone) + y.bandwidth() / 2 + 2.2)
      .attr('font-size', 6).attr('fill', GREY).text(`${d.pct_reduction.toFixed(0)}%`);
  });
  tag(g, 'f');
}
console.log(save(body, 's3'));
