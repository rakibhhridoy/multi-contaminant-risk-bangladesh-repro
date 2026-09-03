// Figure 6 - contaminant contributions flowing to zone-level annual burden.
import { readFileSync } from 'fs';
import { canvas, save, note, d3, W2 } from './svgkit.mjs';
import { CONTAM, WARM, WARM_DEEP, TEAL, TEAL_DEEP, INK, GREY, MUTE, PAPER, warmScale } from './palette.mjs';
import { sankey as mkSankey, sankeyLinkHorizontal, sankeyJustify } from 'd3-sankey';

const D = JSON.parse(readFileSync('data/fig6.json'));
// (A) Sankey, (B) variance decomposition. The manuscript cites Fig. 6B, so both
// panels must be present.
const H = 312, VW = 132, M = { t: 30, r: 94, b: 40, l: 62 };
const TITLE_Y = 14;   // shared baseline for the A and B panel titles
const { body, svg } = canvas(W2, H);

const LAB = { 'As': 'arsenic', 'Mn2+': 'manganese', 'Fe2+': 'iron', 'Cu2+': 'copper',
              'NO3-': 'nitrate', 'Al3+': 'aluminium', 'Cr3+': 'chromium' };
const short = z => z.replace(/_/g, ' ').replace('Floodplain', 'FP').replace('Brahmaputra', 'Brahma.')
                    .replace('Northern Terrace', 'N. Terrace').replace('Eastern Hills', 'E. Hills');

// Contaminants -> total burden -> zones. The middle node makes the figure read
// as an accounting of one quantity rather than an arbitrary bipartite mapping.
const contam = D.contributions.slice().sort((a, b) => b.pct - a.pct);
const zones = D.zones.slice().sort((a, b) => b.daly - a.daly);
const total = d3.sum(zones, d => d.daly);
const nodes = [
  ...contam.map(c => ({ name: LAB[c.contaminant] || c.contaminant, kind: 'c', key: c.contaminant })),
  { name: 'cumulative burden', kind: 'h' },
  ...zones.map(z => ({ name: short(z.zone), kind: 'z' })),
];
const hub = contam.length;
const links = [
  ...contam.map((c, i) => ({ source: i, target: hub, value: c.pct / 100 * total, key: c.contaminant })),
  ...zones.map((z, i) => ({ source: hub, target: hub + 1 + i, value: z.daly, zone: z.zone })),
];

const S = mkSankey().nodeWidth(11).nodePadding(9).nodeAlign(sankeyJustify)
  .extent([[M.l, M.t], [W2 - M.r - VW, H - M.b]]);
const graph = S({ nodes: nodes.map(d => ({ ...d })), links: links.map(d => ({ ...d })) });

// links
graph.links.forEach(l => {
  const warmSide = l.key !== undefined;
  const col = warmSide ? (CONTAM[l.key] || MUTE)
                       : warmScale(0.25 + 0.6 * (l.value / d3.max(zones, z => z.daly)));
  svg.append('path').attr('d', sankeyLinkHorizontal()(l))
    .attr('fill', 'none').attr('stroke', col)
    .attr('stroke-width', Math.max(1, l.width)).attr('stroke-opacity', 0.42);
});
// nodes
graph.nodes.forEach(n => {
  const col = n.kind === 'c' ? (CONTAM[n.key] || MUTE) : n.kind === 'h' ? INK : WARM_DEEP;
  svg.append('rect').attr('x', n.x0).attr('y', n.y0)
    .attr('width', n.x1 - n.x0).attr('height', Math.max(1, n.y1 - n.y0))
    .attr('fill', col).attr('rx', 1.5);
  // The hub sits inside the flow; anything drawn next to it lands on the ribbons.
  // Its label and total go in the bottom margin instead, with the column labels.
  if (n.kind === 'h') return;
  const left = n.kind === 'z';
  svg.append('text')
    .attr('x', left ? n.x1 + 6 : n.x0 - 6)
    .attr('y', (n.y0 + n.y1) / 2 + 2.6)
    .attr('text-anchor', left ? 'start' : 'end')
    .attr('font-size', 7.8).text(n.name);
});
// the total, on the hub
const hubNode = graph.nodes[hub];

// Bottom margin: each label centred under the column it describes, and the hub
// total centred under the hub, all on one baseline. Anything drawn beside the
// hub itself lands on the ribbons, because the hub sits inside the flow.
const BASE = H - 10;
const midOf = ns => {
  const xs = ns.flatMap(n => [n.x0, n.x1]);
  return (Math.min(...xs) + Math.max(...xs)) / 2;
};
// Two baselines. The hub total gets its own line, because centring three labels
// on one line collides when the Sankey's columns sit this close together.
svg.append('text').attr('x', (hubNode.x0 + hubNode.x1) / 2).attr('y', BASE - 12)
  .attr('text-anchor', 'middle').attr('font-size', 8).attr('font-weight', 700).attr('fill', INK)
  .text(`cumulative burden, ${d3.format(',.0f')(total)} DALYs yr⁻¹`);
svg.append('text').attr('x', midOf(graph.nodes.filter(n => n.kind === 'c'))).attr('y', BASE)
  .attr('text-anchor', 'middle').attr('font-size', 7.4).attr('fill', GREY)
  .text('contaminant contribution');
svg.append('text').attr('x', midOf(graph.nodes.filter(n => n.kind === 'z'))).attr('y', BASE)
  .attr('text-anchor', 'middle').attr('font-size', 7.4).attr('fill', GREY)
  .text('burden by zone');
// ---------------- (B) variance decomposition ----------------
{
  const bx = W2 - VW + 16, bw = VW - 42, bh = 150;
  const g = svg.append('g').attr('transform', `translate(${bx},${M.t + 34})`);
  const V = D.variance;
  const y = d3.scaleBand().domain(V.map(d => d.source)).range([0, bh]).padding(0.34);
  const x = d3.scaleLinear().domain([0, 50]).range([0, bw]);
  x.ticks(3).forEach(v => {
    g.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y2', bh)
      .attr('stroke', '#E6E6E6').attr('stroke-width', 0.6);
    g.append('text').attr('x', x(v)).attr('y', bh + 11).attr('text-anchor', 'middle')
      .attr('font-size', 7.2).text(`${v}%`);
  });
  V.forEach(d => {
    const cool = d.source.startsWith('climate') || d.source.startsWith('exposure');
    g.append('rect').attr('x', 0).attr('y', y(d.source)).attr('height', y.bandwidth())
      .attr('width', Math.max(1.5, x(d.pct))).attr('fill', cool ? TEAL : WARM).attr('opacity', 0.92);
    g.append('text').attr('x', 0).attr('y', y(d.source) - 3).attr('font-size', 7.2).text(d.source);
    g.append('text').attr('x', Math.max(1.5, x(d.pct)) + 4).attr('y', y(d.source) + y.bandwidth() / 2 + 2.6)
      .attr('font-size', 7).attr('fill', GREY).text(`${d.lt ? '<' : ''}${d.pct}%`);
  });
  g.append('text').attr('x', 0).attr('y', TITLE_Y - (M.t + 34)).attr('font-size', 8.5)
    .attr('font-weight', 700).text('B  variance in projected HI');
  note(g, 'reduction when each source is fixed', 0, bh + 24, { size: 7, fill: GREY });
}
svg.append('text').attr('x', M.l - 54).attr('y', TITLE_Y)
  .attr('font-size', 8.5).attr('font-weight', 700).text('A  contaminant to burden');
console.log(save(body, 'fig6'));
