/** Display formatting. Locale-aware throughout, so a grouped thousands
 *  separator and a decimal comma follow the reader's browser. */

/** Fixed-fraction number for tok/s, percentages and similar UI values. */
export const fmtNum = (amount: number, digits: number): string =>
  amount.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** Locale-aware integer for token counts and millisecond totals. */
export const fmtInt = (amount: number | string): string =>
  Number(amount).toLocaleString(undefined, { maximumFractionDigits: 0 });

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
