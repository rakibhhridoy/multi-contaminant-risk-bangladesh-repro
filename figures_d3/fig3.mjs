// Figure 3 - multi-contaminant health risk and burden under gridded population weighting.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { CONTAM, WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER, FS } from './palette.mjs';

const D = JSON.parse(readFileSync('data/fig3.json'));
// Panel width is derived, not guessed: two columns, the left label gutter, the
// inter-panel gap and a right margin must all fit inside W2.
const GX = 58, GY = 52, M = { t: 26, l: 62, r: 16 };
const PW = Math.floor((W2 - M.l - GX - M.r) / 2), PH = 150;
const H = M.t + PH * 2 + GY + 52;   // room for the bottom-centred note in (d)
const { body, svg } = canvas(W2, H);
const panel = (r, c) => svg.append('g')
  .attr('transform', `translate(${M.l + c * (PW + GX)},${M.t + r * (PH + GY)})`);
const short = z => z.replace(/_/g, ' ').replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
                    .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');
const Z = D.zones.slice().sort((a, b) => b.median_HI_multi - a.median_HI_multi);

// (a) median HI by zone, arsenic-only against multi-contaminant
{
  const g = panel(0, 0);
  const y = d3.scaleBand().domain(Z.map(d => d.phys_zone)).range([0, PH]).padding(0.28);
  const x = d3.scaleLinear().domain([0, d3.max(Z, d => d.median_HI_multi) * 1.08]).range([0, PW]);
  axisX(g, x, PH, { ticks: 5, label: 'median hazard index' });
  Z.forEach(d => {
    const yy = y(d.phys_zone), h = y.bandwidth();
    g.append('rect').attr('x', 0).attr('y', yy).attr('width', x(d.median_HI_multi)).attr('height', h)
      .attr('fill', WARM).attr('opacity', 0.9);
    g.append('rect').attr('x', 0).attr('y', yy + h * 0.26).attr('width', x(d.median_HI_as_only))
      .attr('height', h * 0.48).attr('fill', TEAL_DEEP);
    g.append('text').attr('x', -5).attr('y', yy + h / 2 + 2.6).attr('text-anchor', 'end')
      .attr('font-size', 7.4).text(short(d.phys_zone));
  });
  g.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y1', 0).attr('y2', PH)
    .attr('stroke', INK).attr('stroke-width', 0.9).attr('stroke-dasharray', '3,2');
  note(g, 'HI = 1', x(1) + 3, -5, { size: 7, fill: INK });
  const lg = g.append('g').attr('transform', `translate(${PW - 96},${PH - 22})`);
  [['multi-contaminant', WARM], ['arsenic only', TEAL_DEEP]].forEach((d, i) => {
    lg.append('rect').attr('y', i * 10 - 5).attr('width', 12).attr('height', 6).attr('fill', d[1]);
    lg.append('text').attr('x', 16).attr('y', i * 10).attr('font-size', 7.4).text(d[0]);
  });
  tag(g, 'a');
}

// (b) per-sample multi against arsenic-only HI
{
  const g = panel(0, 1);
  // Clamping at 0.02 piled every low-arsenic well into a solid stripe on the
  // left edge. Widen the domain so the low tail spreads out instead.
  const x = d3.scaleLog().domain([0.002, 60]).range([0, PW]).clamp(true);
  const y = d3.scaleLog().domain([0.002, 60]).range([PH, 0]).clamp(true);
  axisY(g, y, { values: [0.01, 0.1, 1, 10], w: PW, fmt: d3.format('~g'), label: 'multi-contaminant HI' });
  axisX(g, x, PH, { values: [0.01, 0.1, 1, 10], fmt: d3.format('~g'), label: 'arsenic-only HI' });
  g.append('line').attr('x1', x(0.002)).attr('y1', y(0.002)).attr('x2', x(60)).attr('y2', y(60))
    .attr('stroke', GREY).attr('stroke-width', 0.9).attr('stroke-dasharray', '4,2');
  note(g, '1:1', x(30), y(46), { size: 7, fill: GREY });
  [[1, 'x'], [1, 'y']].forEach(([v, ax]) => g.append('line')
    .attr('x1', ax === 'x' ? x(v) : 0).attr('x2', ax === 'x' ? x(v) : PW)
    .attr('y1', ax === 'y' ? y(v) : 0).attr('y2', ax === 'y' ? y(v) : PH)
    .attr('stroke', INK).attr('stroke-width', 0.6).attr('stroke-dasharray', '2,2').attr('opacity', 0.6));
  g.selectAll('circle').data(D.scatter).join('circle')
    .attr('cx', d => x(d.a)).attr('cy', d => y(d.m)).attr('r', 1.5)
    .attr('fill', d => d.m > 1 ? WARM_DEEP : TEAL).attr('opacity', 0.4);
  tag(g, 'b');
}

// (c) annual burden by zone, multi-contaminant against arsenic-only.
// The text uses this panel to say that arsenic-only surveillance recovers a
// larger share of the burden in the highest-contamination zones, so both series
// must be present.
{
  const g = panel(1, 0);
  const S = D.zones.slice().sort((a, b) => b.annual_DALY_multi - a.annual_DALY_multi);
  const y = d3.scaleBand().domain(S.map(d => d.phys_zone)).range([0, PH]).padding(0.3);
  const x = d3.scaleLinear().domain([0, d3.max(S, d => d.annual_DALY_multi) * 1.24]).range([0, PW]);
  axisX(g, x, PH, { ticks: 4, fmt: v => v >= 1000 ? `${v / 1000}k` : v,
                    label: 'annual DALYs' });
  S.forEach(d => {
    const yy = y(d.phys_zone), h = y.bandwidth();
    g.append('rect').attr('x', 0).attr('y', yy).attr('height', h)
      .attr('width', x(d.annual_DALY_multi)).attr('fill', WARM).attr('opacity', 0.9);
    g.append('rect').attr('x', 0).attr('y', yy + h * 0.26).attr('height', h * 0.48)
      .attr('width', x(d.annual_DALY_as_only)).attr('fill', TEAL_DEEP);
    g.append('text').attr('x', -5).attr('y', yy + h / 2 + 2.6).attr('text-anchor', 'end')
      .attr('font-size', 7.4).text(short(d.phys_zone));
    const share = d.annual_DALY_as_only / d.annual_DALY_multi;
    g.append('text').attr('x', x(d.annual_DALY_multi) + 3).attr('y', yy + h / 2 + 2.6)
      .attr('font-size', 6.8).attr('fill', GREY)
      .text(`As-only ${(share * 100).toFixed(0)}%`);
  });
  const lg = g.append('g').attr('transform', `translate(${PW - 92},${PH - 20})`);
  [['multi-contaminant', WARM], ['arsenic only', TEAL_DEEP]].forEach((d, i) => {
    lg.append('rect').attr('y', i * 10 - 5).attr('width', 12).attr('height', 6).attr('fill', d[1]);
    lg.append('text').attr('x', 16).attr('y', i * 10).attr('font-size', 7.2).text(d[0]);
  });
  tag(g, 'c');
}

// (d) contaminant contributions to the cumulative index
{
  const g = panel(1, 1);
  const C = D.contributions.slice().sort((a, b) => b.pct - a.pct);
  const y = d3.scaleBand().domain(C.map(d => d.contaminant)).range([0, PH]).padding(0.3);
  const x = d3.scaleLinear().domain([0, d3.max(C, d => d.pct) * 1.14]).range([0, PW]);
  axisX(g, x, PH, { ticks: 5, fmt: v => `${v}%`, label: 'contribution to cumulative HI' });
  const LAB = { 'As': 'arsenic', 'Mn2+': 'manganese', 'Fe2+': 'iron', 'Cu2+': 'copper',
                'NO3-': 'nitrate', 'Al3+': 'aluminium', 'Cr3+': 'chromium' };
  C.forEach(d => {
    g.append('rect').attr('x', 0).attr('y', y(d.contaminant)).attr('height', y.bandwidth())
      .attr('width', x(d.pct)).attr('fill', CONTAM[d.contaminant] || MUTE);
    g.append('text').attr('x', -5).attr('y', y(d.contaminant) + y.bandwidth() / 2 + 2.6)
      .attr('text-anchor', 'end').attr('font-size', 7.4).text(LAB[d.contaminant] || d.contaminant);
    g.append('text').attr('x', x(d.pct) + 3).attr('y', y(d.contaminant) + y.bandwidth() / 2 + 2.6)
      .attr('font-size', 7).attr('fill', GREY).text(`${d.pct.toFixed(1)}%`);
  });
  note(g, 'manganese is unmonitored nationally', PW / 2, PH + 38,
       { anchor: 'middle', size: 7.2, fill: WARM_DEEP, italic: true });
  tag(g, 'd');
}
console.log(save(body, 'fig3'));
