import { expect, test } from 'bun:test';
import { truncateAnnounce } from './format';

/** One user-perceived character spread over several code points. Truncating
 *  mid-cluster announces a lone ZWJ or half a flag, so each of these must come
 *  back whole or not at all. */
const ZWJ_FAMILY = '\u{1F468}‍\u{1F469}‍\u{1F467}';
const REGIONAL_FLAG = '\u{1F1FA}\u{1F1F8}';
const SKIN_TONE = '\u{1F44B}\u{1F3FD}';
/** NFD: "e" plus a combining acute. Same character as "é". */
const COMBINING_NFD = 'é';

test('a ZWJ sequence is announced whole or not at all', () => {
  // Cut after one cluster the family must survive intact, not as "👨‍".
  expect(truncateAnnounce(`${ZWJ_FAMILY} tail`, 1)).toBe(`${ZWJ_FAMILY}...`);
  expect(truncateAnnounce(`${ZWJ_FAMILY} tail`, 1)).not.toBe('\u{1F468}‍...');
});

test('a regional-indicator pair is announced whole or not at all', () => {
  expect(truncateAnnounce(`${REGIONAL_FLAG} tail`, 1)).toBe(`${REGIONAL_FLAG}...`);
});

test('a skin-tone modifier stays with its base', () => {
  expect(truncateAnnounce(`${SKIN_TONE} tail`, 1)).toBe(`${SKIN_TONE}...`);
});

test('a combining mark stays with its base, in either spelling', () => {
  expect(truncateAnnounce(`${COMBINING_NFD} tail`, 1)).toBe(`${COMBINING_NFD}...`);
  // NFC "é" is the same character and must truncate to itself, not "e".
  expect(truncateAnnounce('é tail', 1)).toBe('é...');
});

test('text at or under the limit is returned untouched', () => {
  expect(truncateAnnounce('hi', 5)).toBe('hi');
  expect(truncateAnnounce(ZWJ_FAMILY, 1)).toBe(ZWJ_FAMILY);
  expect(truncateAnnounce(REGIONAL_FLAG, 1)).toBe(REGIONAL_FLAG);
});

test('truncation cuts whole clusters across a mixed run', () => {
  const run = `${REGIONAL_FLAG}${ZWJ_FAMILY}cafe`;
  expect(truncateAnnounce(run, 3)).toBe(`${REGIONAL_FLAG}${ZWJ_FAMILY}c...`);
  expect(truncateAnnounce(run, 2)).toBe(`${REGIONAL_FLAG}${ZWJ_FAMILY}...`);
  expect(truncateAnnounce(run, 6)).toBe(`${REGIONAL_FLAG}${ZWJ_FAMILY}cafe`);
});

/** The surviving text, with the truncation marker removed, so invariants can be
 *  stated about the characters that were kept rather than about the ellipsis. */
const kept = (text: string, max: number): string => {
  const out = truncateAnnounce(text, max);
  return out.endsWith('...') ? out.slice(0, -3) : out;
};

/** Count regional indicators in `text`: an odd count means one lost its flag
 *  partner, which is what a mid-pair cut looks like. */
const indicatorsIn = (text: string): number => {
  const found = text.match(/\p{RI}/gu);
  return found === null ? 0 : found.length;
};

test('every cut point leaves whole clusters behind', () => {
  const run = `${REGIONAL_FLAG}${ZWJ_FAMILY}${SKIN_TONE}${COMBINING_NFD}ok`;
  for (let max = 1; max <= 7; max += 1) {
    const body = kept(run, max);
    // A cut never leaves a leading or trailing ZWJ, which announces as nothing.
    expect(body).not.toMatch(/^‍/u);
    expect(body).not.toMatch(/‍$/u);
    // Every regional indicator that survived kept its partner.
    expect(indicatorsIn(body) % 2).toBe(0);
  }
});