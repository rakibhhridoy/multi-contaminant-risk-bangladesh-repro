// S12 - projected mid-century change by zone and contaminant.
import { readFileSync } from 'fs';
import { canvas, save, tag, note, colorbar, d3, W2 } from './svgkit.mjs';
import { warmScale, TEAL, TEAL_DEEP, TEAL_LIGHT, INK, GREY, PAPER } from './palette.mjs';
const D = JSON.parse(readFileSync('data/s12.json'));
const LAB = { 'As': 'As', 'Mn2+': 'Mn', 'Fe2+': 'Fe', 'Cr3+': 'Cr', 'NO3-': 'NO₃' };
const CT = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-'];
const short = z => z.replace(/_/g, ' ').replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
                    .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');
const zones = [...new Set(D.rows.map(d => d.zone))];
const PW = 190, PH = 24 * zones.length, GX = 66, M = { t: 34, l: 108 };
const { body, svg } = canvas(W2, PH + M.t + 74);
const mx = d3.max(D.rows, d => Math.abs(d.pct));
['ssp245', 'ssp585'].forEach((ssp, pi) => {
  const g = svg.append('g').attr('transform', `translate(${M.l + pi * (PW + GX)},${M.t})`);
  const cw = PW / CT.length, ch = PH / zones.length;
  const M2 = new Map(D.rows.filter(d => d.ssp === ssp).map(d => [`${d.zone}|${d.contaminant}`, d.pct]));
  zones.forEach((z, i) => CT.forEach((c, j) => {
    const v = M2.get(`${z}|${c}`);
    if (v == null) return;
    g.append('rect').attr('x', j * cw).attr('y', i * ch).attr('width', cw - 1).attr('height', ch - 1)
      .attr('fill', v >= 0 ? warmScale(Math.abs(v) / mx) : TEAL)
      .attr('opacity', v >= 0 ? 1 : 0.25 + 0.7 * Math.abs(v) / mx).attr('rx', 1);
    g.append('text').attr('x', j * cw + cw / 2 - 0.5).attr('y', i * ch + ch / 2 + 2.6)
      .attr('text-anchor', 'middle').attr('font-size', 6.4)
      .attr('fill', (v >= 0 && Math.abs(v) / mx > 0.62) ? PAPER : INK)
      .text(`${v > 0 ? '+' : ''}${v.toFixed(0)}`);
  }));
  CT.forEach((c, j) => g.append('text').attr('x', j * cw + cw / 2).attr('y', -6)
    .attr('text-anchor', 'middle').attr('font-size', 7.6).text(LAB[c]));
  if (pi === 0) zones.forEach((z, i) => g.append('text').attr('x', -6).attr('y', i * ch + ch / 2 + 2.6)
    .attr('text-anchor', 'end').attr('font-size', 7.4).text(short(z)));
  g.append('text').attr('x', PW / 2).attr('y', -20).attr('text-anchor', 'middle')
    .attr('font-size', 8.5).attr('font-weight', 700)
    .text(ssp === 'ssp245' ? 'SSP2-4.5' : 'SSP5-8.5');
  tag(g, pi === 0 ? 'a' : 'b', -34, -20);
});
note(svg, 'per-zone median percentage change in concentration by 2050; warm = increase, teal = decrease',
     M.l, PH + M.t + 26, { size: 7.2, fill: GREY });
console.log(save(body, 's12'));
