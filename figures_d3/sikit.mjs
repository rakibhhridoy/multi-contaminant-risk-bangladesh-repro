// Shared 3x2 grid used by the four multi-panel supplementary figures.
import { canvas, save, tag, d3, W2 } from './svgkit.mjs';
// Panel width is DERIVED, never hard-coded: three columns, the left label
// gutter, two inter-panel gaps and a right margin must all fit inside W2.
// Hard-coding it put the third column 33 pt off the page.
export function grid6({ ph = 118, gx = 52, gy = 70, top = 26, left = 56, right = 16 } = {}) {
  const pw = Math.floor((W2 - left - 2 * gx - right) / 3);
  const H = top + ph * 2 + gy + 40;
  const { body, svg } = canvas(W2, H);
  const panel = (r, c) => svg.append('g')
    .attr('transform', `translate(${left + c * (pw + gx)},${top + r * (ph + gy)})`);
  return { body, svg, panel, PW: pw, PH: ph, H };
}
export const shortZone = z => String(z).replace(/_/g, ' ')
  .replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
  .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');
export const shortScen = s => String(s).replace(/^S(\d)_.*/, 'S$1');
