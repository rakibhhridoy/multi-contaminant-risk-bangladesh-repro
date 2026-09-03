// Headless SVG scaffolding for the Paper 2 figures.
// jsdom supplies a DOM so d3 can build the document exactly as it would in a
// browser; nothing here is interactive, and the output is a standalone SVG that
// rsvg-convert turns into the PDF LaTeX embeds.
import { JSDOM } from 'jsdom';
import * as d3 from 'd3';
import { writeFileSync } from 'fs';
import { FONT, FS, INK, RULE, GRID, GREY, PAPER } from './palette.mjs';

// Elsevier column widths, in points (1 pt = 1/72 in).
export const W1 = 255;   // single column, ~90 mm
export const W2 = 539;   // double column, ~190 mm

export function canvas(width, height) {
  const dom = new JSDOM('<!DOCTYPE html><body></body>');
  const body = d3.select(dom.window.document.body);
  const svg = body.append('svg')
    .attr('xmlns', 'http://www.w3.org/2000/svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('font-family', FONT).attr('font-size', FS.label)
    .attr('fill', INK);
  svg.append('rect').attr('width', width).attr('height', height).attr('fill', PAPER);
  return { dom, svg, body };
}

export function save(body, stem) {
  const out = `svg/${stem}.svg`;
  writeFileSync(out, body.html().replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"'));
  return out;
}

// Bold panel letter, top-left, with a white halo so it never collides with data.
export function tag(g, letter, x = -34, y = -8) {
  const t = g.append('g').attr('transform', `translate(${x},${y})`);
  t.append('text').attr('font-size', FS.tag).attr('font-weight', 700)
    .attr('fill', INK).attr('stroke', PAPER).attr('stroke-width', 3)
    .attr('paint-order', 'stroke').text(letter);
  return t;
}

export function title(g, s, x = 0, y = -12) {
  g.append('text').attr('x', x).attr('y', y)
    .attr('font-size', FS.title).attr('font-weight', 700).text(s);
}

// Axes drawn by hand rather than via d3.axisBottom, so tick density, label
// offsets and the spine treatment match Paper 1's matplotlib style exactly:
// bottom+left spines only, light grid behind the data, 3 pt ticks.
export function axisX(g, scale, h, { ticks = 6, label = '', fmt = null, values = null } = {}) {
  const t = values || scale.ticks(ticks);
  const f = fmt || scale.tickFormat(ticks);
  const ax = g.append('g').attr('transform', `translate(0,${h})`);
  ax.append('line').attr('x2', scale.range()[1]).attr('stroke', RULE).attr('stroke-width', 0.8);
  t.forEach(v => {
    const x = scale(v);
    ax.append('line').attr('x1', x).attr('x2', x).attr('y2', 3)
      .attr('stroke', RULE).attr('stroke-width', 0.8);
    ax.append('text').attr('x', x).attr('y', 12).attr('text-anchor', 'middle')
      .attr('font-size', FS.tick).text(f(v));
  });
  if (label) ax.append('text').attr('x', scale.range()[1] / 2).attr('y', 27)
    .attr('text-anchor', 'middle').attr('font-size', FS.label).text(label);
  return ax;
}

export function axisY(g, scale, { ticks = 5, label = '', fmt = null, values = null, grid = true, w = 0 } = {}) {
  const t = values || scale.ticks(ticks);
  const f = fmt || scale.tickFormat(ticks);
  const ax = g.append('g');
  ax.append('line').attr('y1', scale.range()[0]).attr('y2', scale.range()[1])
    .attr('stroke', RULE).attr('stroke-width', 0.8);
  t.forEach(v => {
    const y = scale(v);
    if (grid && w) ax.append('line').attr('x1', 0).attr('x2', w).attr('y1', y).attr('y2', y)
      .attr('stroke', GRID).attr('stroke-width', 0.6);
    ax.append('line').attr('x1', -3).attr('y1', y).attr('y2', y)
      .attr('stroke', RULE).attr('stroke-width', 0.8);
    ax.append('text').attr('x', -6).attr('y', y + 3).attr('text-anchor', 'end')
      .attr('font-size', FS.tick).text(f(v));
  });
  if (label) ax.append('text').attr('transform', `translate(-38,${(scale.range()[0]) / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').attr('font-size', FS.label).text(label);
  return ax;
}

export function note(g, s, x, y, { anchor = 'start', size = FS.note, fill = GREY, italic = false } = {}) {
  return g.append('text').attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .attr('font-size', size).attr('fill', fill)
    .attr('font-style', italic ? 'italic' : 'normal').text(s);
}

// Horizontal colour bar for the map panels.
export function colorbar(g, { x, y, w, h, vmin, vmax, ramp, label, fmt = d3.format('.2~f') }) {
  const n = 64;
  for (let i = 0; i < n; i++) {
    g.append('rect').attr('x', x + i * w / n).attr('y', y)
      .attr('width', w / n + 0.6).attr('height', h)
      .attr('fill', ramp(i / (n - 1)));
  }
  g.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', h)
    .attr('fill', 'none').attr('stroke', RULE).attr('stroke-width', 0.6);
  [[x, vmin, 'start'], [x + w, vmax, 'end']].forEach(([px, v, a]) =>
    g.append('text').attr('x', px).attr('y', y + h + 9).attr('text-anchor', a)
      .attr('font-size', FS.note).text(fmt(v)));
  if (label) g.append('text').attr('x', x + w / 2).attr('y', y - 4)
    .attr('text-anchor', 'middle').attr('font-size', FS.note).text(label);
}
// Text width estimate. jsdom has no layout engine, so a box cannot ask its text
// how wide it is; without an estimate, boxed text silently overflows its box.
// Per-character advances for the Arial/Helvetica stack, in em, good to a few
// percent, which is all a padded box needs.
const ADV = { narrow: 0.28, wide: 0.78, digit: 0.556, default: 0.53 };
export function textWidth(str, size = 8, bold = false) {
  let em = 0;
  for (const ch of String(str)) {
    if ('iljtfIr.,:;\'|!()[]'.includes(ch)) em += ADV.narrow;
    else if ('mwMW%@'.includes(ch)) em += ADV.wide;
    else if (ch >= '0' && ch <= '9') em += ADV.digit;
    else if (ch === ' ') em += 0.28;
    else em += ADV.default;
  }
  return em * size * (bold ? 1.06 : 1);
}

// A rounded pill sized to its own text, so it can never be narrower than what it
// encloses. Returns the width used, for callers that need to lay out around it.
export function pill(g, { text, x = 0, y = 0, size = 8, bold = true, padX = 12,
                          h = 17, fill, textFill, maxW = Infinity, align = 'left' }) {
  const w = Math.min(maxW, textWidth(text, size, bold) + padX * 2);
  const x0 = align === 'centre' ? x - w / 2 : x;
  g.append('rect').attr('x', x0).attr('y', y).attr('width', w).attr('height', h)
    .attr('rx', h / 2).attr('fill', fill).attr('opacity', 0.14);
  g.append('text').attr('x', x0 + w / 2).attr('y', y + h / 2 + size * 0.36)
    .attr('text-anchor', 'middle').attr('font-size', size)
    .attr('font-weight', bold ? 700 : 400).attr('fill', textFill)
    .text(text);
  return w;
}
export { d3 };
