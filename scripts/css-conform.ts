#!/usr/bin/env bun
/**
 * Rewrite a Tailwind 4 build so it passes W3C validation (vnu --css), with the
 * same behavior in every browser. Called by scripts/build-web.sh on each
 * stylesheet it emits; the argument is rewritten in place.
 *
 * Tailwind emits three constructs the W3C CSS checker rejects:
 *   - `@property --tw-*` registrations (inherits: false plus an initial value);
 *   - an `@layer properties { @supports (…margin-trim…) { *, … { --tw-*: … } } }`
 *     fallback that applies those same initial values only where @property is
 *     missing;
 *   - `--tw-gradient-*` custom properties inside `transition-property` lists.
 *
 * The rewrite keeps the fallback and drops its @supports guard, so every browser
 * gets the universal reset (which is what makes the variables behave as
 * non-inheriting, the way @property would), and removes the registrations. The
 * gradient entries go because no utility here sets a gradient; the script fails
 * if one ever does, rather than silently breaking its transition.
 *
 * Usage: bun scripts/css-conform.ts path/to/style.css
 */
import { argv } from 'node:process';

const LAYER_OPEN = '@layer properties{@supports ';
const GRADIENT_TRANSITION = ',--tw-gradient-from,--tw-gradient-via,--tw-gradient-to';
const PROPERTY_RULE = /@property --[\w-]+\{[^{}]*\}/gu;

const fail = (message: string): never => {
  throw new Error(`css-conform: ${message}`);
};

/** Index just past the brace that closes the block opened at `open`. */
const blockEnd = (css: string, open: number): number => {
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === '{') { depth += 1; }
    if (css[index] === '}') {
      depth -= 1;
      if (depth === 0) { return index + 1; }
    }
  }
  return fail('unbalanced braces in the properties layer');
};

/** Drop the @supports guard around the properties-layer reset. */
const unwrapPropertiesLayer = (css: string): string => {
  const start = css.indexOf(LAYER_OPEN);
  if (start === -1) { return fail('no @layer properties fallback; Tailwind output changed, revisit this script'); }
  const supportsAt = start + '@layer properties{'.length;
  const bodyOpen = css.indexOf('{', supportsAt + '@supports '.length);
  const supportsEnd = blockEnd(css, bodyOpen);
  const body = css.slice(bodyOpen + 1, supportsEnd - 1);
  return css.slice(0, supportsAt) + body + css.slice(supportsEnd);
};

const target = argv[2] ?? fail('usage: bun scripts/css-conform.ts <style.css>');
const source = await Bun.file(target).text();
if (source.includes('--tw-gradient-from:')) {
  fail('a gradient utility is in use; its transition needs --tw-gradient-*, so this rewrite no longer applies');
}
const conformed = unwrapPropertiesLayer(source)
  .replaceAll(PROPERTY_RULE, '')
  .replaceAll(GRADIENT_TRANSITION, '');
await Bun.write(target, conformed);
