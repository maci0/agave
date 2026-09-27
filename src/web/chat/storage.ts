/** Browser-persisted UI state.
 *
 *  The system prompt is tab-scoped (sessionStorage) so prompt text does not
 *  survive the tab; the sampling settings and the stats toggle are
 *  per-browser (localStorage). One read and one write cover the whole
 *  sampling record, so a new setting cannot be persisted on one path and
 *  forgotten on another.
 */

import type { Sampling } from './types';

/** Key the system prompt used before it moved to sessionStorage. */
const SYSTEM_PROMPT_KEY = 'agave_system_prompt';

const TEMPERATURE_KEY = 'agave_temperature';
const TOP_P_KEY = 'agave_top_p';
const MAX_TOKENS_KEY = 'agave_max_tokens';
const SHOW_STATS_KEY = 'agave_show_stats';

export const MAX_TOKENS_MIN = 1;
export const MAX_TOKENS_MAX = 4096;
const MAX_TOKENS_DEFAULT = '512';

const readNumber = (key: string, fallback: number): number => {
  const raw = localStorage.getItem(key);
  if (raw === null) {return fallback;}
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
};

/** Read the system prompt, migrating the legacy localStorage key once.
 *  The old key is deleted in the same pass, so the prompt cannot linger in a
 *  store the user was told was not used. */
const readSystemPrompt = (): string => {
  const current = sessionStorage.getItem(SYSTEM_PROMPT_KEY);
  if (current !== null) {return current;}
  // One-time migration of a legacy storage key; not a runtime environment fallback.
  const legacy = localStorage.getItem(SYSTEM_PROMPT_KEY);
  if (legacy === null) {return '';}
  sessionStorage.setItem(SYSTEM_PROMPT_KEY, legacy);
  localStorage.removeItem(SYSTEM_PROMPT_KEY);
  return legacy;
};

/** Clamp a raw max-tokens field to the allowed range; unparseable text becomes
 *  the minimum. */
export const clampMaxTokens = (raw: string): number => {
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < MAX_TOKENS_MIN) {return MAX_TOKENS_MIN;}
  if (parsed > MAX_TOKENS_MAX) {return MAX_TOKENS_MAX;}
  return parsed;
};

/** A field is valid only when it holds a whole number inside the range, with no
 *  stray text. */
export const isMaxTokensValid = (raw: string): boolean => {
  const parsed = Number.parseInt(raw, 10);
  return !Number.isNaN(parsed) && String(parsed) === raw.trim() && parsed >= MAX_TOKENS_MIN && parsed <= MAX_TOKENS_MAX;
};

/** The stored settings, normalized on the way out. A stored value can be empty
 *  or out of range (an older build, or a field the user left mid-edit), so the
 *  reader clamps instead of trusting it. */
export const readSampling = (): Sampling => {
  const stored = localStorage.getItem(MAX_TOKENS_KEY);
  return {
    temperature: readNumber(TEMPERATURE_KEY, 0),
    topP: readNumber(TOP_P_KEY, 1),
    maxTokens: stored === null ? MAX_TOKENS_DEFAULT : (isMaxTokensValid(stored) ? stored : String(clampMaxTokens(stored))),
    system: readSystemPrompt(),
  };
};

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
