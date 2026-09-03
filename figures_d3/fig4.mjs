// Figure 4 - climate scenario. Sensitivity coefficients are the within-well
// paired differences over the 810 locations sampled in both campaigns, so this
// figure reports the corrected calibration, not the unpaired contrast.
import { readFileSync } from 'fs';
import { canvas, save, tag, axisX, axisY, note, d3, W2 } from './svgkit.mjs';
import { CONTAM, WARM, WARM_DEEP, WARM_MID, TEAL, TEAL_DEEP, TEAL_LIGHT,
         INK, GREY, MUTE, PAPER, warmScale, FS } from './palette.mjs';

const D = JSON.parse(readFileSync('data/fig4.json'));
const GX = 60, GY = 54, M = { t: 26, l: 66, r: 16 };
const PW = Math.floor((W2 - M.l - GX - M.r) / 2), PH = 150;
const H = M.t + PH * 2 + GY + 30;
const { body, svg } = canvas(W2, H);
const panel = (r, c) => svg.append('g')
  .attr('transform', `translate(${M.l + c * (PW + GX)},${M.t + r * (PH + GY)})`);
const short = z => z.replace(/_/g, ' ').replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
                    .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');
const LAB = { 'As': 'As', 'Mn2+': 'Mn', 'Fe2+': 'Fe', 'Cr3+': 'Cr', 'NO3-': 'NO₃' };

// (a) paired seasonal sensitivity for arsenic, by zone and depth
{
  const g = panel(0, 0);
  const S = D.sensitivity.filter(d => d.contaminant === 'As' && d.sensitivity_per_pct != null);
  const zones = [...new Set(S.map(d => d.phys_zone))];
  const depths = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep'];
  const cw = PW / depths.length, ch = PH / zones.length;
  const mx = d3.max(S, d => Math.abs(d.sensitivity_per_pct)) || 1;
  S.forEach(d => {
    const i = zones.indexOf(d.phys_zone), j = depths.indexOf(d.depth_zone);
    if (i < 0 || j < 0) return;
    const v = d.sensitivity_per_pct;
    g.append('rect').attr('x', j * cw).attr('y', i * ch).attr('width', cw - 1).attr('height', ch - 1)
      .attr('fill', v >= 0 ? warmScale(Math.abs(v) / mx) : TEAL_LIGHT)
      .attr('opacity', v >= 0 ? 1 : 0.35 + 0.65 * Math.abs(v) / mx).attr('rx', 1);
    if (d.significant) g.append('circle').attr('cx', j * cw + cw - 7).attr('cy', i * ch + 6)
      .attr('r', 1.8).attr('fill', INK);
  });
  zones.forEach((z, i) => g.append('text').attr('x', -5).attr('y', i * ch + ch / 2 + 2.5)
    .attr('text-anchor', 'end').attr('font-size', 7.2).text(short(z)));
  depths.forEach((d, j) => g.append('text').attr('x', j * cw + cw / 2).attr('y', PH + 11)
    .attr('text-anchor', 'middle').attr('font-size', 7.2).text(d.replace('Medium_Deep', 'Med-deep')));
  note(g, 'paired arsenic sensitivity per % precipitation change; dot = significant (Wilcoxon)',
       PW / 2, PH + 25, { anchor: 'middle', size: 7.2, fill: GREY });
  tag(g, 'a');
}

// (b) baseline against scenario arsenic, SSP5-8.5
{
  const g = panel(0, 1);
  const C = D.crossings.filter(d => d.base != null && d.med != null);
  const mx = d3.max(C, d => Math.max(d.base, d.hi)) * 1.08;
  const x = d3.scaleLog().domain([0.3, mx]).range([0, PW]).clamp(true);
  const y = d3.scaleLog().domain([0.3, mx]).range([PH, 0]).clamp(true);
  axisY(g, y, { values: [1, 10, 100], w: PW, fmt: d3.format('~g'), label: 'scenario 2050 As (µg/L)' });
  axisX(g, x, PH, { values: [1, 10, 100], fmt: d3.format('~g'), label: 'baseline As (µg/L)' });
  g.append('line').attr('x1', x(0.3)).attr('y1', y(0.3)).attr('x2', x(mx)).attr('y2', y(mx))
    .attr('stroke', GREY).attr('stroke-width', 0.9).attr('stroke-dasharray', '4,2');
  // WHO guideline on both axes: the upper-left quadrant is the set that crosses
  g.append('rect').attr('x', 0).attr('y', 0).attr('width', x(D.who)).attr('height', y(D.who))
    .attr('fill', WARM).attr('opacity', 0.07);
  [['x', D.who], ['y', D.who]].forEach(([ax, v]) => g.append('line')
    .attr('x1', ax === 'x' ? x(v) : 0).attr('x2', ax === 'x' ? x(v) : PW)
    .attr('y1', ax === 'y' ? y(v) : 0).attr('y2', ax === 'y' ? y(v) : PH)
    .attr('stroke', INK).attr('stroke-width', 0.8).attr('stroke-dasharray', '3,2'));
  C.forEach(d => {
    const cross = d.base <= D.who && d.med > D.who;
    g.append('line').attr('x1', x(d.base)).attr('x2', x(d.base))
      .attr('y1', y(d.lo)).attr('y2', y(d.hi))
      .attr('stroke', cross ? WARM_DEEP : MUTE).attr('stroke-width', cross ? 1.4 : 0.9);
    g.append('circle').attr('cx', x(d.base)).attr('cy', y(d.med)).attr('r', cross ? 3.6 : 2.2)
      .attr('fill', cross ? WARM_DEEP : TEAL).attr('stroke', PAPER).attr('stroke-width', cross ? 1.2 : 0.7);
    if (cross) g.append('text').attr('x', x(d.base) - 6).attr('y', y(d.med) - 6)
      .attr('text-anchor', 'end').attr('font-size', 7.2).attr('font-weight', 700).attr('fill', WARM_DEEP)
      .text(short(d.phys_zone));
  });
  note(g, 'crosses WHO guideline', 5, 10, { size: 7.2, fill: WARM_DEEP });
  note(g, 'bars span the six-model range', 5, 20, { size: 7, fill: GREY });
  tag(g, 'b');
}

// (c) national median change by contaminant
{
  const g = panel(1, 0);
  const order = ['As', 'Fe2+', 'Mn2+', 'Cr3+', 'NO3-'];
  const N = D.national.filter(d => order.includes(d.contaminant));
  const x = d3.scaleBand().domain(order).range([0, PW]).padding(0.34);
  const inner = d3.scaleBand().domain(['ssp245', 'ssp585']).range([0, x.bandwidth()]).padding(0.18);
  const ext = [d3.min(N, d => Math.min(d.pct, d.lo ?? d.pct)),
               d3.max(N, d => Math.max(d.pct, d.hi ?? d.pct))];
  const y = d3.scaleLinear().domain([Math.min(-15, ext[0] * 1.12), Math.max(15, ext[1] * 1.12)])
              .nice().range([PH, 0]);
  axisY(g, y, { ticks: 6, w: PW, fmt: v => `${v > 0 ? '+' : ''}${v}%`, label: 'national median change by 2050' });
  g.append('line').attr('x1', 0).attr('x2', PW).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', INK).attr('stroke-width', 0.9);
  N.forEach(d => {
    const xx = x(d.contaminant) + inner(d.ssp);
    g.append('rect').attr('x', xx).attr('width', inner.bandwidth())
      .attr('y', Math.min(y(0), y(d.pct))).attr('height', Math.abs(y(d.pct) - y(0)))
      .attr('fill', d.pct >= 0 ? (d.ssp === 'ssp585' ? WARM_DEEP : WARM) : (d.ssp === 'ssp585' ? TEAL_DEEP : TEAL))
      .attr('opacity', 0.92);
    if (d.lo != null && d.hi != null) {
      const cx = xx + inner.bandwidth() / 2;
      g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d.lo)).attr('y2', y(d.hi))
        .attr('stroke', INK).attr('stroke-width', 0.9).attr('opacity', 0.65);
    }
    g.append('text').attr('x', xx + inner.bandwidth() / 2).attr('y', d.pct >= 0 ? y(d.hi ?? d.pct) - 4 : y(d.lo ?? d.pct) + 9)
      .attr('text-anchor', 'middle').attr('font-size', 6.8).attr('fill', INK)
      .text(`${d.pct > 0 ? '+' : ''}${d.pct.toFixed(0)}`);
  });
  order.forEach(k => g.append('text').attr('x', x(k) + x.bandwidth() / 2).attr('y', PH + 12)
    .attr('text-anchor', 'middle').attr('font-size', 7.6).text(LAB[k]));
  const lg = g.append('g').attr('transform', `translate(${PW - 88},4)`);
  [['SSP2-4.5', WARM], ['SSP5-8.5', WARM_DEEP], ['inter-model range', INK]].forEach((d, i) => {
    if (d[0].startsWith('inter')) {
      lg.append('line').attr('x1', 5).attr('x2', 5).attr('y1', i * 10 - 6).attr('y2', i * 10 + 1)
        .attr('stroke', INK).attr('stroke-width', 0.9).attr('opacity', 0.65);
    } else {
      lg.append('rect').attr('y', i * 10 - 5).attr('width', 11).attr('height', 6).attr('fill', d[1]);
    }
    lg.append('text').attr('x', 15).attr('y', i * 10).attr('font-size', 7.2).text(d[0]);
  });
  tag(g, 'c');
}

// (d) inter-model spread of the precipitation change
{
  const g = panel(1, 1);
  const cv = D.cv;
  const x = d3.scaleBand().domain(cv.map(d => d.ssp)).range([0, PW]).padding(0.5);
  const y = d3.scaleLinear().domain([0, d3.max(cv, d => d.mean + 2.2 * d.std)]).nice().range([PH, 0]);
  axisY(g, y, { ticks: 5, w: PW, fmt: v => `+${v}%`, label: 'precipitation change ΔP by 2050' });
  cv.forEach(d => {
    const cx = x(d.ssp) + x.bandwidth() / 2;
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d.mean - d.std)).attr('y2', y(d.mean + d.std))
      .attr('stroke', TEAL_DEEP).attr('stroke-width', 2.2);
    [d.mean - d.std, d.mean + d.std].forEach(v => g.append('line')
      .attr('x1', cx - 7).attr('x2', cx + 7).attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', TEAL_DEEP).attr('stroke-width', 1.4));
    g.append('circle').attr('cx', cx).attr('cy', y(d.mean)).attr('r', 4.2)
      .attr('fill', WARM_DEEP).attr('stroke', PAPER).attr('stroke-width', 1.4);
    g.append('text').attr('x', cx).attr('y', PH + 12).attr('text-anchor', 'middle')
      .attr('font-size', 7.6).text(d.ssp.replace('ssp', 'SSP').replace(/(\d)(\d)(\d)/, '$1-$2.$3'));
    g.append('text').attr('x', cx + 12).attr('y', y(d.mean) + 3).attr('font-size', 7.2)
      .attr('fill', INK).text(`+${d.mean.toFixed(0)}%`);
    g.append('text').attr('x', cx + 12).attr('y', y(d.mean) + 12).attr('font-size', 6.8)
      .attr('fill', GREY).text(`CV ${d.cv.toFixed(2)}, n=${d.n}`);
  });
  note(g, 'central value anchored to IPCC AR6; the ensemble supplies the spread',
       PW / 2, PH + 25, { anchor: 'middle', size: 7, fill: GREY });
  tag(g, 'd');
}
console.log(save(body, 'fig4'));
