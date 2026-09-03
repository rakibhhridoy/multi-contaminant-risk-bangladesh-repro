// Shared visual language for Paper 2 (HAZADV-D-26-01282).
//
// The warm sequential ramp and the contour blue are taken verbatim from the
// author's Paper 1 house style (Paper1/JHMRevision/figstyle.py), so the two
// manuscripts read as one body of work. Paper 1's note on the contour colour is
// worth repeating: it is "the one cool mark, so contours read over the warm ramp".
//
// Two-colour rule for this paper, set 2026-09-02:
//   WARM  -> anything carrying risk, hazard, burden or exceedance
//   TEAL  -> everything else (baselines, comparators, cool/reference series),
//            replacing the green and blue previously used
export const WARMR = ['#fdece0','#fbd5bd','#f8b894','#f2946a',
                      '#eb6834','#d9482c','#b3261e','#7d1a15'];

export const WARM_LIGHT = '#f8b894';
export const WARM_MID   = '#eb6834';
export const WARM       = '#d9482c';   // primary warm (Paper 1 WARM1)
export const WARM_DEEP  = '#7d1a15';   // deepest warm (Paper 1 WARM2)

export const TEAL       = '#2A9D8F';   // primary cool, replaces green + blue
export const TEAL_DEEP  = '#17605E';
export const TEAL_LIGHT = '#8FCFC8';

export const CONTOUR = '#104281';      // Paper 1's reserved contour blue
export const INK     = '#222222';
export const GREY    = '#9AA0A6';
export const MUTE    = '#C2C7CC';
export const RULE    = '#444444';
export const GRID    = '#E6E6E6';
export const PAPER   = '#ffffff';

// Per-contaminant assignment. Arsenic and manganese carry the paper's risk
// argument and take warm; the rest are cool or neutral so the warm marks stay
// meaningful rather than decorative.
export const CONTAM = {
  'As':   WARM_DEEP,
  'Mn2+': WARM,
  'Fe2+': TEAL_DEEP,
  'Cr3+': TEAL,
  'NO3-': TEAL_LIGHT,
  'PO43-': WARM_MID,
  'Cu2+': GREY,
  'Al3+': MUTE,
};

// Paired series (dry vs wet, arsenic-only vs multi-contaminant).
export const SERIES = { primary: WARM, secondary: TEAL, muted: MUTE };

// Typography. Matches Paper 1: sans, 9.5 pt body, bold panel tags.
export const FONT = "Arial, Helvetica, 'DejaVu Sans', sans-serif";
export const FS = { tick: 8.5, label: 9.5, title: 10.5, tag: 12.5, note: 8 };

// Continuous ramp helper: t in [0,1] -> warm colour, via piecewise interpolation
// over the eight ramp stops.
export function warmScale(t) {
  const x = Math.max(0, Math.min(1, t)) * (WARMR.length - 1);
  const i = Math.min(Math.floor(x), WARMR.length - 2);
  return mix(WARMR[i], WARMR[i + 1], x - i);
}
function mix(a, b, u) {
  const p = h => [1,3,5].map(k => parseInt(h.slice(k, k + 2), 16));
  const [r1,g1,b1] = p(a), [r2,g2,b2] = p(b);
  const c = (x,y) => Math.round(x + (y - x) * u).toString(16).padStart(2,'0');
  return `#${c(r1,r2)}${c(g1,g2)}${c(b1,b2)}`;
}
