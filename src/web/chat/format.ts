/** Display formatting. Locale-aware throughout, so a grouped thousands
 *  separator and a decimal comma follow the reader's browser. */

/** Locale list for the formatters below. The UI passes nothing and reads the
 *  browser's; a test names one, so a locale that is not the build default is
 *  still checkable. */
export type Locales = string | Array<string> | undefined;

/** Fixed-fraction number for tok/s and similar UI values. */
export const fmtNum = (amount: number, digits: number, locales?: Locales): string =>
  amount.toLocaleString(locales, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** Cluster rule set shared by every truncation here. Built once: the rules
 *  depend only on the resolved locale, not on the text being cut. */
const graphemeSegmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

/** Locale-aware integer for token counts and millisecond totals. */
export const fmtInt = (amount: number | string, locales?: Locales): string =>
  Number(amount).toLocaleString(locales, { maximumFractionDigits: 0 });

/** A ratio, 0..1, as the locale writes a percentage: the mark sits where that
 *  locale's digits put it, and a bidi locale gets the isolation mark that keeps
 *  "12,3 %" from being reordered around it.
 *
 *  Appending "%" to an already-scaled number puts the mark on the wrong side of
 *  the digits in Arabic and Hebrew, and drops the mark where the locale wants
 *  one (`12,3 %` in de-DE and fr-FR, against `12,3%`). */
export const fmtPercent = (ratio: number, digits: number, locales?: Locales): string =>
  ratio.toLocaleString(locales, {
    style: 'percent',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

/** A byte count as whole megabytes. Intl supplies the unit label and the
 *  spacing that locale puts around it, so a caller appends nothing: a hardcoded
 *  " MB" glued onto a grouped number reads wrong wherever the two conventions
 *  disagree. */
export const fmtMegabytes = (bytes: number, locales?: Locales): string =>
  (bytes / 1024 / 1024).toLocaleString(locales, {
    style: 'unit',
    unit: 'megabyte',
    unitDisplay: 'short',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });

/** Truncate on grapheme cluster boundaries, so what is announced is whole
 *  characters rather than a code-point prefix that can end mid-cluster.
 *
 *  Code points are not enough: a ZWJ sequence (👨‍👩‍👧), a regional-indicator
 *  pair (🇺🇸), a skin-tone modifier (👋🏽) or a base plus combining marks is one
 *  user-perceived character spread over several code points. Cutting between
 *  them leaves a lone ZWJ or half a flag, which a screen reader announces as
 *  noise. `Intl.Segmenter` is the platform's own cluster rule set, so this
 *  agrees with how the browser lays the same text out. */
export const truncateAnnounce = (text: string, maxChars: number): string => {
  const clusters = Array.from(graphemeSegmenter.segment(text), (part) => part.segment);
  if (clusters.length <= maxChars) {return text;}
  return `${clusters.slice(0, maxChars).join('')}...`;
};

/** Digit zero of a script, for each decimal digit set a locale may write a
 *  number in. The zero runs contiguously to nine in every one of them (ASCII,
 *  Arabic-Indic U+0660, Extended Arabic-Indic U+06F0, fullwidth U+FF10, Thai
 *  U+0E50, small Arabic forms U+FE50, Myanmar U+104A0), so one start plus the
 *  offset covers a whole set. `Number()` and `parseInt` read ASCII digits only,
 *  so `Number('٥١٢')` is NaN: a field typed on a keyboard in another script
 *  fails to parse, and a clamped field silently becomes its minimum. */
const DIGIT_ZERO_BY_SCRIPT: ReadonlyArray<number> = [
  0x30, 0x660, 0x6F0, 0xFF10, 0xE50, 0xFE50, 0x1_04A0,
];

/** Translate every digit in `text` to its ASCII counterpart, leaving all other
 *  characters alone: grouping marks, sign and stray text survive, so a caller
 *  strips or rejects them itself. Only the sets above are mapped; a digit from
 *  another script passes through and the caller's own parse rejects it. */
export const asciiDigits = (text: string): string => {
  let out = '';
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    const zero = DIGIT_ZERO_BY_SCRIPT.find((start) => cp >= start && cp <= start + 9);
    out += zero === undefined ? ch : String.fromCodePoint(0x30 + (cp - zero));
  }
  return out;
};

/** Spaces a reader types as a thousands separator, U+00A0 and U+202F
 *  included. Stripped before the decimal mark is located, so a space cannot be
 *  mistaken for one. */
const stripSpaces = (text: string): string => text.replaceAll(/[\s  ]/gu, '');

/** A whole-number side carries only grouping marks, and each sits in front of a
 *  whole group: three digits under the western rule, two where another group
 *  follows (the Indian lakh and crore pattern, so "1,00,000" is a hundred
 *  thousand). That is what tells a reader's grouping mark from a second decimal
 *  mark, so "1,0000" is rejected instead of read as ten thousand. A sign and a
 *  bare digit run pass through untouched. */
const isGroupedWhole = (text: string): boolean => {
  if (!text.includes('.') && !text.includes(',')) {return /^[+-]?\d*$/u.test(text);}
  if (!/^[+-]?\d{1,3}(?:[.,]\d+)+$/u.test(text)) {return false;}
  const runs = text.slice(text.search(/[.,]/u) + 1).split(/[.,]/u);
  return runs.every((run, index) => run.length === 3 || (run.length === 2 && index < runs.length - 1));
};

/** A number split at its decimal mark, with every mark located rather than
 *  assumed: `whole` and `fraction` are the ASCII digit runs either side of it,
 *  and `ok` is false when the text carries a mark the reader could not have
 *  meant (a stray letter fails the same way). */
type Split = { whole: string; fraction: string; ok: boolean };

/** Classify the marks in an ASCII-digit text by the rule a locale keyboard
 *  follows: the last `.` or `,` is the decimal mark unless exactly three
 *  digits follow it to the end, in which case every mark in the text is a
 *  grouping mark. So "1,024" and "1.024" group, "512.0" and "0,7" carry a
 *  decimal, and "1.024,5" (de-DE) and "1,024.5" (en-US) both split at the last
 *  mark with the other dropped as grouping. One rule, so the temperature field
 *  and the whole-number token budget cannot read each other's input as a
 *  different value. */
const splitAtDecimal = (text: string): Split => {
  const bad: Split = { whole: '', fraction: '', ok: false };
  const last = Math.max(text.lastIndexOf('.'), text.lastIndexOf(','));
  /* Three digits to the end is a final group, so that mark groups too, and
     every mark before it is checked against the same grouping rule. */
  if (last !== -1 && /^\d{3}$/u.test(text.slice(last + 1))) {
    if (!isGroupedWhole(text)) {return bad;}
    return { whole: text.replaceAll(/[.,]/gu, ''), fraction: '', ok: true };
  }
  if (last === -1) {
    return /^[+-]?\d*$/u.test(text) ? { whole: text, fraction: '', ok: true } : bad;
  }
  const head = text.slice(0, last);
  const fraction = text.slice(last + 1);
  if (!isGroupedWhole(head) || !/^\d*$/u.test(fraction)) {return bad;}
  return { whole: head.replaceAll(/[.,]/gu, ''), fraction, ok: true };
};

/** Parse a whole number the reader typed, in any digit script, tolerating the
 *  grouping marks their locale writes. Text that is anything but digits, an
 *  optional sign and a whole number is NaN, which the range check and the
 *  inline field error already treat as invalid. A decimal mark is a rejection
 *  here rather than a rounding step: "512.0" must not read as 5120 on a field
 *  that asks for a whole number. */
export const parseLocaleInt = (raw: string): number => {
  const split = splitAtDecimal(stripSpaces(asciiDigits(raw)));
  if (!split.ok || split.fraction !== '') {return Number.NaN;}
  return /^[+-]?\d+$/u.test(split.whole) ? Math.trunc(Number(split.whole)) : Number.NaN;
};

/** Parse a decimal the reader typed, in any digit script, with its grouping
 *  and decimal marks as its locale writes them: "1.024,5" (de-DE) and
 *  "1,024.5" (en-US) both read as 1024.5, and "1.024" (fi groups with a
 *  period) reads as 1024. Arabic decimal U+066B becomes ASCII first. The sign
 *  is kept apart from the digits, so a leading mark ("-,5") still parses while
 *  stray text ("1a024") does not. */
export const parseLocaleNumber = (raw: string): number => {
  const spaced = stripSpaces(asciiDigits(raw).replaceAll('٫', '.'));
  const split = splitAtDecimal(spaced);
  if (!split.ok) {return Number.NaN;}
  let sign = '';
  if (split.whole.startsWith('-') || split.whole.startsWith('+')) {
    sign = split.whole.slice(0, 1);
  }
  const whole = sign === '' ? split.whole : split.whole.slice(1);
  if (!/^\d*$/u.test(whole)) {return Number.NaN;}
  if (whole === '' && split.fraction === '') {return Number.NaN;}
  if (split.fraction === '') {return Number(`${sign}${whole}`);}
  return Number(`${sign}${whole === '' ? '0' : whole}.${split.fraction}`);
};

/** Local calendar date as YYYY-MM-DD. `toISOString().slice(0, 10)` is UTC, so a
 *  US-evening export would be stamped with tomorrow's date. */
export const localDateYmd = (now: Date = new Date()): string => {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

/** Context-window counter: 4 reads as 4, 4096 as 4K. */
export const fmtCtx = (tokens: number): string =>
  tokens >= 1024 ? `${fmtInt(Math.round(tokens / 1024))}K` : fmtInt(tokens);

/** Language roots whose layout has to be mirrored: Arabic, Aramaic, Central
 *  Kurdish (Sorani), Divehi, Persian, Hebrew, Kashmiri, Kurdish, Pashto,
 *  Sindhi, Uyghur, Urdu and Yiddish. A language root list rather than a locale
 *  list, because direction is a property of the writing system: a language not
 *  named here still gets `rtl` when its tag starts with one of these roots, so
 *  a script added to a language later, or a regional tag this list has never
 *  heard of, is covered without editing it. */
const RTL_LANGUAGE_ROOTS = new Set(['ar', 'arc', 'ckb', 'dv', 'fa', 'he', 'ks', 'ku', 'ps', 'sd', 'ug', 'ur', 'yi']);

/** Script subtags whose writing direction is right to left: Adlam, Arabic,
 *  Hebrew, Mandaic, Mende Kikakui, N'Ko, Hanifi Rohingya, Sogdian, Syriac,
 *  Thaana and Yezidi. This is the authoritative list when a tag carries one:
 *  the script decides, not the language, so `az-Arab` and `az-Latn` disagree on
 *  purpose and `ff-Adlm` (Fulah in the Arabic-derived Adlam script) is right to
 *  left although its language prefix is not Arabic. */
const RTL_SCRIPTS = new Set(['adlm', 'arab', 'hebr', 'mand', 'mend', 'nkoo', 'rohg', 'sogo', 'syrc', 'thaa', 'yezi']);

/** A four-letter script subtag out of a BCP 47 tag: the one no region code
 *  collides with. It may come last or carry a region and a variant after it,
 *  as in `az-Arab`, `az-Arab-IR` and `ar-Arab-IR-EG`. */
const SCRIPT_SUBTAG = /-(?<script>[a-z]{4})(?:-[a-z0-9]{2,3})*(?:-[a-z0-9]{4,8})*(?:-[a-z0-9]{4,8})*\b/u;

/** Writing direction for a BCP 47 language tag: `rtl` or `ltr`.
 *
 *  The UI is laid out with CSS logical properties throughout, so mirroring is
 *  a `dir` attribute and nothing else: without it an Arabic or Hebrew reader
 *  gets an LTR page with the sidebar and the message bubbles on the wrong side
 *  and the composer pinned to the wrong edge. `navigator.language` is the only
 *  signal the browser offers, so it decides — no preference to store, no
 *  reload, and it follows the reader if they change it in the browser. */
export const writingDirection = (language: string | undefined | null): 'ltr' | 'rtl' => {
  const tag = (language ?? '').trim().toLowerCase();
  if (tag === '') {return 'ltr';}
  const script = SCRIPT_SUBTAG.exec(tag)?.groups?.script;
  if (script !== undefined) {return RTL_SCRIPTS.has(script) ? 'rtl' : 'ltr';}
  /* No script subtag, so the language decides. `ar-EG` and `fa_AF` alike: the
     part before the first separator is the language. */
  const root = tag.split(/[-_]/u)[0] ?? '';
  return RTL_LANGUAGE_ROOTS.has(root) ? 'rtl' : 'ltr';
};
