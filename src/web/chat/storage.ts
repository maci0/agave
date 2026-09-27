/** Browser-persisted UI state. The system prompt is tab-scoped (sessionStorage)
 *  so prompt text does not survive the tab; the sampling settings and the stats
 *  toggle are per-browser (localStorage). */

/** Key the system prompt used before it moved to sessionStorage. */
const LEGACY_SYSTEM_PROMPT_KEY = 'agave_system_prompt';

const TEMPERATURE_KEY = 'agave_temperature';
const TOP_P_KEY = 'agave_top_p';
const MAX_TOKENS_KEY = 'agave_max_tokens';
const SHOW_STATS_KEY = 'agave_show_stats';

export const MAX_TOKENS_MIN = 1;
export const MAX_TOKENS_MAX = 4096;

/** Read the system prompt, migrating the legacy localStorage key once.
 *  The old key is deleted in the same pass, so the prompt cannot linger in a
 *  store the user was told was not used. */
export function readSystemPrompt(): string {
  const current = sessionStorage.getItem(LEGACY_SYSTEM_PROMPT_KEY);
  if (current !== null) {return current;}
  // One-time migration of a legacy storage key; not a runtime environment fallback.
  const legacy = localStorage.getItem(LEGACY_SYSTEM_PROMPT_KEY);
  if (legacy === null) {return '';}
  sessionStorage.setItem(LEGACY_SYSTEM_PROMPT_KEY, legacy);
  localStorage.removeItem(LEGACY_SYSTEM_PROMPT_KEY);
  return legacy;
}

export function writeSystemPrompt(value: string): void {
  sessionStorage.setItem(LEGACY_SYSTEM_PROMPT_KEY, value);
}

export function clearSystemPrompt(): void {
  sessionStorage.removeItem(LEGACY_SYSTEM_PROMPT_KEY);
  localStorage.removeItem(LEGACY_SYSTEM_PROMPT_KEY);
}

function readNumber(key: string, fallback: number): number {
  const raw = localStorage.getItem(key);
  if (raw === null) {return fallback;}
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function readTemperature(): number {
  return readNumber(TEMPERATURE_KEY, 0);
}

export function writeTemperature(value: number): void {
  localStorage.setItem(TEMPERATURE_KEY, String(value));
}

export function readTopP(): number {
  return readNumber(TOP_P_KEY, 1);
}

export function writeTopP(value: number): void {
  localStorage.setItem(TOP_P_KEY, String(value));
}

/** A stored value can be empty or out of range (an older build, or a field the
 *  user left mid-edit), so the reader normalizes instead of trusting it. */
export function readMaxTokens(): string {
  const raw = localStorage.getItem(MAX_TOKENS_KEY);
  if (raw === null) {return '512';}
  return isMaxTokensValid(raw) ? raw : String(clampMaxTokens(raw));
}

export function writeMaxTokens(value: string): void {
  localStorage.setItem(MAX_TOKENS_KEY, value);
}

/** Clamp a raw max-tokens field to the allowed range; unparseable text becomes
 *  the minimum. */
export function clampMaxTokens(raw: string): number {
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < MAX_TOKENS_MIN) {return MAX_TOKENS_MIN;}
  if (parsed > MAX_TOKENS_MAX) {return MAX_TOKENS_MAX;}
  return parsed;
}

/** A field is valid only when it holds a whole number inside the range, with no
 *  stray text. */
export function isMaxTokensValid(raw: string): boolean {
  const parsed = Number.parseInt(raw, 10);
  return !Number.isNaN(parsed) && String(parsed) === raw.trim() && parsed >= MAX_TOKENS_MIN && parsed <= MAX_TOKENS_MAX;
}

export function readShowStats(): boolean {
  return localStorage.getItem(SHOW_STATS_KEY) === '1';
}

export function writeShowStats(on: boolean): void {
  localStorage.setItem(SHOW_STATS_KEY, on ? '1' : '0');
}
