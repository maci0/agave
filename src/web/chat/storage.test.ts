/**
 * Tests for the persisted sampling settings.
 *
 * The property these pin is that a value the reader typed in their own locale
 * reaches the engine. `Number()` and `parseInt` read ASCII digits only, so a
 * token budget typed in Arabic-Indic or fullwidth digits parsed as NaN and the
 * field clamped to its minimum: a reader asking for 512 got a budget of one
 * token. The same held for a stored temperature written with a decimal comma.
 *
 * Run: bun test src/web (also wired into scripts/lint-web.sh).
 */

import { afterAll, beforeEach, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';

import {
  MAX_TOKENS_MAX,
  MAX_TOKENS_MIN,
  clampMaxTokens,
  isMaxTokensValid,
  readSampling,
} from './storage';

GlobalRegistrator.register({ url: 'http://127.0.0.1:49453' });

/** Arabic-Indic, Extended Arabic (Persian), fullwidth and Thai digits, all
 *  reading "512". A keyboard in any of those scripts produced text the
 *  parser rejected. */
const SAME_NUMBER = ['512', '٥١٢', '۵۱۲', '５１２', '๕๑๒'];

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

/* `bun test src/web` loads every file into one process, so a registration that
   is never released makes the next file's own `register()` throw. The other
   three DOM suites unregister in `afterAll`; without this one, storage.test.ts
   loaded before turn.test.tsx (alphabetical) and the whole gate failed on
   "Happy DOM has already been globally registered" before running a single
   turn test. */
afterAll(async () => { await GlobalRegistrator.unregister(); });

test('a token budget typed in any digit script is accepted, not clamped to the minimum', () => {
  for (const raw of SAME_NUMBER) {
    expect(isMaxTokensValid(raw)).toBe(true);
    expect(clampMaxTokens(raw)).toBe(512);
  }
});

test('a grouped token budget is the number the reader typed', () => {
  /* German, Swiss and Finnish group with a period, en-US with a comma, and
     every locale accepts a space; the narrow no-break space is what a locale
     keyboard emits. */
  expect(clampMaxTokens('1,024')).toBe(1024);
  expect(clampMaxTokens('1.024')).toBe(1024);
  expect(clampMaxTokens('1\u202F024')).toBe(1024);
  expect(isMaxTokensValid('1,024')).toBe(true);
});

test('stray text is still rejected on a token budget', () => {
  for (const raw of ['512abc', '1e3', '0.7', '', String(MAX_TOKENS_MAX + 1)]) {
    expect(isMaxTokensValid(raw)).toBe(false);
  }
});

test('an out-of-range budget clamps to the bound it passed, not to the minimum', () => {
  expect(clampMaxTokens('0')).toBe(MAX_TOKENS_MIN);
  expect(clampMaxTokens(String(MAX_TOKENS_MAX + 1))).toBe(MAX_TOKENS_MAX);
});

test('a stored sampling value written in another digit script still loads', () => {
  localStorage.setItem('agave_temperature', '٠٫٧');
  localStorage.setItem('agave_top_p', '0,9');
  localStorage.setItem('agave_max_tokens', '٥١٢');
  const sampling = readSampling();
  expect(sampling.temperature).toBeCloseTo(0.7, 10);
  expect(sampling.topP).toBeCloseTo(0.9, 10);
  /* Kept as the reader wrote it, so the field still shows what they typed. */
  expect(sampling.maxTokens).toBe('٥١٢');
});

test('an unparseable stored number still falls back to the engine default', () => {
  localStorage.setItem('agave_temperature', 'warm');
  localStorage.setItem('agave_top_p', '');
  expect(readSampling().temperature).toBe(0);
  expect(readSampling().topP).toBe(1);
});
