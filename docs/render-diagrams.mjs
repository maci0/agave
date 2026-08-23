#!/usr/bin/env bun
/**
 * Render all Mermaid diagrams in docs/tutorial/*.md to SVG/PNG.
 * Resolves CSS custom properties before rasterizing (resvg doesn't support var() or color-mix()).
 *
 * Usage:
 *   bun run docs/render-diagrams.mjs [--out-dir docs/diagrams] [--png] [--svg]
 *
 * Requires (installed globally via bun):
 *   bun add -g beautiful-mermaid @resvg/resvg-js
 */

import { renderMermaidSVG } from 'beautiful-mermaid';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join, basename, extname } from 'node:path';

// All 7 colors required — resvg fails silently if any is missing.
const THEME = {
  bg:      '#ffffff',
  fg:      '#1a1a2e',
  accent:  '#4a6cf7',
  line:    '#4a6cf7',
  muted:   '#6b7280',
  surface: '#e8f0fe',
  border:  '#4a6cf7',
};

// Resolve CSS custom properties (var(--xxx)) to hex values before rasterizing.
const resolveVars = (svg, theme) =>
  svg
    .replaceAll('var(--bg)', theme.bg)
    .replaceAll('var(--fg)', theme.fg)
    .replaceAll('var(--accent)', theme.accent)
    .replaceAll('var(--line)', theme.line)
    .replaceAll('var(--muted)', theme.muted)
    .replaceAll('var(--surface)', theme.surface)
    .replaceAll('var(--border)', theme.border);

// Extract mermaid blocks from a Markdown file.
const extractDiagrams = (md) => {
  const blocks = [];
  for (const m of md.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
    blocks.push({ source: m[1].trim(), index: blocks.length });
  }
  return blocks;
};

// Parse CLI args
const args = process.argv.slice(2);
const outDir = args.includes('--out-dir') ? args[args.indexOf('--out-dir') + 1] : 'docs/diagrams';
// oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- boolean operands: default-PNG requires falsy-or semantics
const emitPng = args.includes('--png') || !args.includes('--svg');
const emitSvg = args.includes('--svg');

mkdirSync(outDir, { recursive: true });

const tutorialDir = 'docs/tutorial';
const files = readdirSync(tutorialDir).filter(f => f.endsWith('.md'));

let totalDiagrams = 0;
let errors = 0;

for (const file of files) {
  const md = readFileSync(join(tutorialDir, file), 'utf8');
  const diagrams = extractDiagrams(md);
  if (diagrams.length === 0) { continue; }

  // oxlint-disable-next-line @rikalabs/no-pass-through-intermediate-vars -- reused by mkdir and every write path below
  const fileDir = join(outDir, basename(file, extname(file)));
  mkdirSync(fileDir, { recursive: true });

  for (const { source, index } of diagrams) {
    const name = `diagram-${String(index + 1).padStart(2, '0')}`;
    try {
      const resolvedSvg = resolveVars(renderMermaidSVG(source, { theme: THEME }), THEME);

      if (emitSvg) {
        writeFileSync(join(fileDir, `${name}.svg`), resolvedSvg);
      }
      if (emitPng) {
        writeFileSync(
          join(fileDir, `${name}.png`),
          new Resvg(resolvedSvg, { fitTo: { mode: 'zoom', value: 2 } }).render().asPng(),
        );
      }
      totalDiagrams += 1;
    } catch (error) {
      // Batch renderer: account for the failure, keep rendering siblings, report count at exit.
      // oxlint-disable-next-line @rikalabs/no-silent-catch-fallback -- failures are counted and reported, not swallowed
      console.error(`ERROR: ${file} diagram ${index + 1}: ${error.message}`);
      errors += 1;
    }
  }
}

console.log(`Rendered ${totalDiagrams} diagrams to ${outDir}/ (${errors} errors)`);
