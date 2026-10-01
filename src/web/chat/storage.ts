/** Browser-persisted UI state.
 *
 *  The system prompt is tab-scoped (sessionStorage) so prompt text does not
 *  survive the tab; the sampling settings and the stats toggle are
 *  per-browser (localStorage). One read and one write cover the whole
 *  sampling record, so a new setting cannot be persisted on one path and
 *  forgotten on another.
 */

import { asciiDigits, parseLocaleInt, parseLocaleNumber } from './format';
import type { Sampling } from './types';

/** Key the system prompt used before it moved to sessionStorage. */
const SYSTEM_PROMPT_KEY = 'agave_system_prompt';

const TEMPERATURE_KEY = 'agave_temperature';
const TOP_P_KEY = 'agave_top_p';
// oxlint-disable-next-line @rikalabs/no-hardcoded-secrets -- a localStorage key, not a credential
const MAX_TOKENS_KEY = 'agave_max_tokens';
const SHOW_STATS_KEY = 'agave_show_stats';

export const MAX_TOKENS_MIN = 1;
export const MAX_TOKENS_MAX = 4096;
// oxlint-disable-next-line @rikalabs/no-hardcoded-secrets -- a token budget, not a credential
const MAX_TOKENS_DEFAULT = '512';

/** Read the system prompt, migrating the legacy localStorage key once.
 *  The old key is deleted in the same pass, so the prompt cannot linger in a
 *  store the user was told was not used. */
const readSystemPrompt = (): string => {
  const current = sessionStorage.getItem(SYSTEM_PROMPT_KEY);
  if (current !== null) {return current;}
  // One-time migration of a legacy storage key; not a runtime environment fallback.
  const previous = localStorage.getItem(SYSTEM_PROMPT_KEY);
  if (previous === null) {return '';}
  sessionStorage.setItem(SYSTEM_PROMPT_KEY, previous);
  localStorage.removeItem(SYSTEM_PROMPT_KEY);
  return previous;
};

/** Clamp a raw max-tokens field to the allowed range; unparseable text becomes
 *  the minimum. The field is read with `parseLocaleInt`, so digits in any
 *  script and the grouping marks the reader's locale writes are accepted
 *  instead of falling through to the minimum as a bare `Number()` would. */
export const clampMaxTokens = (raw: string): number => {
  const parsed = parseLocaleInt(raw);
  if (Number.isNaN(parsed) || parsed < MAX_TOKENS_MIN) {return MAX_TOKENS_MIN;}
  if (parsed > MAX_TOKENS_MAX) {return MAX_TOKENS_MAX;}
  return parsed;
};

/** A field is valid only when it holds a whole number inside the range, with no
 *  stray text. */
export const isMaxTokensValid = (raw: string): boolean => {
  const parsed = parseLocaleInt(raw);
  if (Number.isNaN(parsed) || parsed < MAX_TOKENS_MIN || parsed > MAX_TOKENS_MAX) {return false;}
  /* What the reader typed must be the same number and nothing else, so "512"
     passes while "512.0", "1e3" and "٥١٢abc" do not. The comparison is against
     the canonical digits the parser read, which is what lets the same value
     written in another digit script, or with the locale's grouping marks, stay
     valid instead of reading as stray text. */
  return String(parsed) === asciiDigits(raw).replaceAll(/[\s ,٬،]/gu, '');
};

/** The stored max-tokens field, normalized on the way out. A stored value can
 *  be empty or out of range (an older build, or a field the user left
 *  mid-edit), so the reader clamps instead of trusting it. */
const readMaxTokens = (): string => {
  const stored = localStorage.getItem(MAX_TOKENS_KEY);
  if (stored === null) { return MAX_TOKENS_DEFAULT; }
  if (isMaxTokensValid(stored)) { return stored; }
  return String(clampMaxTokens(stored));
};

/** A stored number, or `fallback` when the key is missing, blank or not a
 *  finite number. `parseLocaleNumber('')` is NaN, so blank is checked before
 *  coercing. Read with the locale parser for the same reason as the token
 *  budget: a value written by an earlier build in another digit script, or with
 *  the reader's own grouping marks, still loads. */
const readFinite = (key: string, fallback: number): number => {
  const raw = localStorage.getItem(key)?.trim() ?? '';
  const value = raw === '' ? Number.NaN : parseLocaleNumber(raw);
  return Number.isFinite(value) ? value : fallback;
};

/** The stored settings, normalized on the way out. A missing or unparseable
 *  key falls back to the engine default rather than poisoning the turn. */
export const readSampling = (): Sampling => ({
  temperature: readFinite(TEMPERATURE_KEY, 0),
  topP: readFinite(TOP_P_KEY, 1),
  maxTokens: readMaxTokens(),
  system: readSystemPrompt(),
});

/** Persist the sampling record. A max-tokens field the user is still typing
 *  into is not stored, so a half-typed value cannot survive a reload; the
 *  system prompt is tab-scoped and the rest is per-browser. */
export const writeSampling = (next: Sampling): void => {
  localStorage.setItem(TEMPERATURE_KEY, String(next.temperature));
  localStorage.setItem(TOP_P_KEY, String(next.topP));
  if (isMaxTokensValid(next.maxTokens)) { localStorage.setItem(MAX_TOKENS_KEY, next.maxTokens); }
  sessionStorage.setItem(SYSTEM_PROMPT_KEY, next.system);
};

/** Drop the system prompt from both stores, so a cleared prompt cannot be
 *  resurrected by a later migration. */
export const clearStoredSystemPrompt = (): void => {
  sessionStorage.removeItem(SYSTEM_PROMPT_KEY);
  localStorage.removeItem(SYSTEM_PROMPT_KEY);
};

export const readShowStats = (): boolean => localStorage.getItem(SHOW_STATS_KEY) === '1';

export const writeShowStats = (on: boolean): void => {
  localStorage.setItem(SHOW_STATS_KEY, on ? '1' : '0');
};
