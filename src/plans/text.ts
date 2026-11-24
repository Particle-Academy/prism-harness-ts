/**
 * The rules for text a model writes into a plan (G-360).
 *
 * Every field is model output, and the model may have been steered by a
 * document it read. So a field is REFUSED, never truncated, when it is empty,
 * over its cap, not well-formed Unicode (a lone surrogate), or contains a
 * character from FORBIDDEN: the C0 and C1 controls (line breaks included), the
 * guillemets that fence model text in the plan context, zero-width and bidi
 * controls, the line and paragraph separators, word joiners and the byte order
 * mark. A field naming one of the plan frame's own markers is refused as well,
 * so a title cannot close the frame and write text that reads as the host's.
 *
 * The same codepoint set is spelled out in all three languages. Lengths are
 * counted in Unicode code points in all three, never in UTF-16 units.
 */
export const PlanText = {
  GOAL: 200,
  TITLE: 160,
  INSTRUCTION: 600,
  QUESTION: 300,
  OPTION: 80,
  OPTIONS: 6,
  SUMMARY: 200,
  REASON: 200,
  ANSWER: 1000,
  /** Render caps inside the plan context. */
  RENDER_TITLE: 120,
  RENDER_INSTRUCTION: 400,
  RENDER_NOTE: 160,
  RENDER_ANSWER: 300,
  /** Lower-case markers of the plan frame; a field containing one is refused. */
  MARKERS: ['[plan', '[end of plan'] as readonly string[],
} as const;

const FORBIDDEN_CLASS = '\\u0000-\\u001F\\u007F-\\u009F\\u00AB\\u00BB\\u200B-\\u200F\\u2028-\\u202E\\u2060-\\u2069\\uFEFF\\uFFF9-\\uFFFB';
const FORBIDDEN = new RegExp(`[${FORBIDDEN_CLASS}]`, 'u');
const FORBIDDEN_ALL = new RegExp(`[${FORBIDDEN_CLASS}]`, 'gu');
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
const MARKER = /\[(end of plan|plan)/giu;

/** Whether a string is well-formed Unicode: no lone surrogate, the one way a JS string can be invalid text. */
export function wellFormed(value: string): boolean {
  return !LONE_SURROGATE.test(value);
}

/** Code points, the unit every cap here is counted in. */
export function planTextLength(value: string): number {
  let count = 0;
  for (const _ of value) count += 1;
  return count;
}

/** Whether a model-written field is acceptable as it stands. */
export function acceptablePlanText(value: unknown, max: number): value is string {
  if (typeof value !== 'string' || value === '' || !wellFormed(value)) return false;
  if (planTextLength(value) > max) return false;
  if (FORBIDDEN.test(value)) return false;
  const lower = value.toLowerCase();

  return !PlanText.MARKERS.some((marker) => lower.includes(marker));
}

/** Text cleaned for the plan context: forbidden characters removed, markers defused, cut to `max` code points. */
export function renderPlanText(value: string, max: number): string {
  if (!wellFormed(value)) return '';
  let clean = value.replace(FORBIDDEN_ALL, '').replace(MARKER, '($1');
  const points = [...clean];
  if (points.length > max) clean = `${points.slice(0, max - 1).join('')}…`;

  return clean;
}
