/** Display formatting. Locale-aware throughout, so a grouped thousands
 *  separator and a decimal comma follow the reader's browser. */

/** Fixed-fraction number for tok/s, percentages and similar UI values. */
export const fmtNum = (value: number, digits: number): string =>
  Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });;

/** Locale-aware integer for token counts and millisecond totals. */
export const fmtInt = (value: number | string): string =>
  Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });;

/** Truncate by Unicode code points so surrogate pairs (emoji, some CJK) are not split. */
export const truncateAnnounce = (text: string, maxChars: number): string => {
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
export const fmtCtx = (value: number): string =>
  value >= 1024 ? `${fmtInt(Math.round(value / 1024))}K` : fmtInt(value);;
