/** Display formatting. Locale-aware throughout, so a grouped thousands
 *  separator and a decimal comma follow the reader's browser. */

/** Locale list for the formatters below. The UI passes nothing and reads the
 *  browser's; a test names one, so a locale that is not the build default is
 *  still checkable. */
export type Locales = string | Array<string> | undefined;

/** Fixed-fraction number for tok/s and similar UI values. */
export const fmtNum = (amount: number, digits: number, locales?: Locales): string =>
  amount.toLocaleString(locales, { minimumFractionDigits: digits, maximumFractionDigits: digits });

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

/** Truncate by Unicode code points so surrogate pairs (emoji, some CJK) are not split. */
export const truncateAnnounce = (text: string, maxChars: number): string => {
  // oxlint-disable-next-line typescript-eslint/no-misused-spread -- code-point iteration, not character spread
  const chars = [...text];
  if (chars.length <= maxChars) {return text;}
  return `${chars.slice(0, maxChars).join('')}...`;
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
