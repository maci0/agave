import { expect, test } from 'bun:test';
import { parseLocaleInt, parseLocaleNumber, truncateAnnounce, writingDirection } from './format';

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

/** A reader types the digits their own keyboard produces. `Number()` and
 *  `parseInt` read ASCII only, so every other script parsed as NaN and a field
 *  that clamps on an unparseable value fell silently to its minimum. */

/** Arabic-Indic ٠٧ as "0.7": an Arabic-Indic zero, the Arabic decimal mark and
 *  an Arabic-Indic seven. */
const ARABIC_DECIMAL = '٠٫٧';

const ARABIC_INDIC = '٥١٢';
const EXTENDED_ARABIC = '۵۱۲';
const FULLWIDTH = '５１２';
const THAI = '๕๑๒';

test('a whole number reads the same in every digit script', () => {
  for (const digits of ['512', ARABIC_INDIC, EXTENDED_ARABIC, FULLWIDTH, THAI]) {
    expect(parseLocaleInt(digits)).toBe(512);
  }
});

test('a grouped whole number is a grouping mark, not stray text', () => {
  expect(parseLocaleInt('1,024')).toBe(1024);
  // German, Swiss and Finnish group with a period.
  expect(parseLocaleInt('1.024')).toBe(1024);
  // The spaces a thousands separator is typed as, narrow no-break included.
  expect(parseLocaleInt('1 024')).toBe(1024);
  expect(parseLocaleInt('1 024')).toBe(1024);
  expect(parseLocaleInt('1 024')).toBe(1024);
  // Indian lakh and crore grouping.
  expect(parseLocaleInt('1,00,000')).toBe(100_000);
});

test('a decimal mark on a whole-number field is a rejection', () => {
  // "512.0" must not read as 5120, and "0.7" is not a whole number.
  expect(Number.isNaN(parseLocaleInt('512.0'))).toBe(true);
  expect(Number.isNaN(parseLocaleInt('0.7'))).toBe(true);
  // A group that is not a group.
  expect(Number.isNaN(parseLocaleInt('1,0000'))).toBe(true);
  expect(Number.isNaN(parseLocaleInt('1,,024'))).toBe(true);
});

test('a decimal reads the same whichever mark the locale writes', () => {
  for (const raw of ['0.7', '0,7', '٠,٧', ARABIC_DECIMAL]) {
    expect(parseLocaleNumber(raw)).toBeCloseTo(0.7, 10);
  }
  // Both spellings of a grouped four-digit decimal mean the same number.
  expect(parseLocaleNumber('1.024,5')).toBe(1024.5);
  expect(parseLocaleNumber('1,024.5')).toBe(1024.5);
});

test('a signed decimal keeps its sign away from the digits', () => {
  expect(parseLocaleNumber('-0.5')).toBe(-0.5);
  expect(parseLocaleNumber('-,5')).toBe(-0.5);
  expect(parseLocaleNumber('+0,5')).toBe(0.5);
});

test('text that is not a number stays unparseable', () => {
  for (const raw of ['', 'abc', '1e3', '512abc', '.', ',']) {
    expect(Number.isNaN(parseLocaleNumber(raw))).toBe(true);
  }
  for (const raw of ['', 'abc', '1e3', '51a']) {
    expect(Number.isNaN(parseLocaleInt(raw))).toBe(true);
  }
});

/**
 * The layout rules in the tree are all CSS logical properties, so the whole of
 * RTL support is one `dir` attribute. Without it an Arabic or Hebrew reader
 * gets an LTR page with the sidebar and the message bubbles on the wrong side.
 * These pin the direction a browser tag maps to.
 */

const RTL_TAGS = ['ar', 'ar-EG', 'he', 'he-IL', 'fa', 'fa_AF', 'ur', 'ps', 'sd', 'ug', 'ckb', 'dv', 'yi', 'arc'];

test('a right-to-left tag mirrors the page', () => {
  for (const tag of RTL_TAGS) {
    expect(writingDirection(tag)).toBe('rtl');
  }
});

test('a left-to-right tag does not', () => {
  for (const tag of ['en', 'en-US', 'de', 'fr-FR', 'ja', 'ko', 'zh-Hans', 'ru', 'hi', 'th', 'az-Latn']) {
    expect(writingDirection(tag)).toBe('ltr');
  }
});

test('an absent or blank language is left-to-right, not a crash', () => {
  expect(writingDirection(undefined)).toBe('ltr');
  expect(writingDirection(null)).toBe('ltr');
  expect(writingDirection('')).toBe('ltr');
  expect(writingDirection('  ')).toBe('ltr');
});

test('a script subtag decides over the language prefix', () => {
  /* Kurdish (Kurmanji) is written in Latin script and Azeri in Latin too, so a
     language prefix alone would mirror these pages for readers who do not read
     them right to left. The script subtag is the authoritative signal. */
  expect(writingDirection('az-Arab')).toBe('rtl');
  expect(writingDirection('az-Latn')).toBe('ltr');
  expect(writingDirection('ku-Latn')).toBe('ltr');
  expect(writingDirection('ar-Latn')).toBe('ltr');
  /* The script decides even where the language prefix says otherwise: Fulah
     is written right to left in the Adlam script, and a reader whose browser
     reports that tag must get a mirrored page. */
  expect(writingDirection('ff-Adlm')).toBe('rtl');
  expect(writingDirection('ff-Latn')).toBe('ltr');
  /* A script the list does not name is LTR, not a guess at RTL. */
  expect(writingDirection('en-Latn')).toBe('ltr');
});
