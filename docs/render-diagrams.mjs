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
function resolveVars(svg, theme) {
  return svg
    .replaceAll(/var\(--bg\)/g, theme.bg)
    .replaceAll(/var\(--fg\)/g, theme.fg)
    .replaceAll(/var\(--accent\)/g, theme.accent)
    .replaceAll(/var\(--line\)/g, theme.line)
    .replaceAll(/var\(--muted\)/g, theme.muted)
    .replaceAll(/var\(--surface\)/g, theme.surface)
    .replaceAll(/var\(--border\)/g, theme.border);
}

// Extract mermaid blocks from a Markdown file.
function extractDiagrams(md) {
  const blocks = [];
  const re = /```mermaid\n([\s\S]*?)```/g;
  let m;
  let idx = 0;
  while ((m = re.exec(md)) !== null) {
    blocks.push({ source: m[1].trim(), index: idx++ });
  }
  return blocks;
}

// Parse CLI args
const args = process.argv.slice(2);
const outDir = args.includes('--out-dir') ? args[args.indexOf('--out-dir') + 1] : 'docs/diagrams';
const emitPng = args.includes('--png') ?? !args.includes('--svg');
const emitSvg = args.includes('--svg');

mkdirSync(outDir, { recursive: true });

const tutorialDir = 'docs/tutorial';
const files = readdirSync(tutorialDir).filter(f => f.endsWith('.md'));

let totalDiagrams = 0;
let errors = 0;

for (const file of files) {
  const md = readFileSync(join(tutorialDir, file), 'utf8');
  const diagrams = extractDiagrams(md);
  if (diagrams.length === 0) {continue;}

  const stem = basename(file, extname(file));
  const fileDir = join(outDir, stem);
  mkdirSync(fileDir, { recursive: true });

  for (const { source, index } of diagrams) {
    const name = `diagram-${String(index + 1).padStart(2, '0')}`;
    try {
      const rawSvg = renderMermaidSVG(source, { theme: THEME });
      const resolvedSvg = resolveVars(rawSvg, THEME);

      if (emitSvg) {
        writeFileSync(join(fileDir, `${name}.svg`), resolvedSvg);
      }
      if (emitPng) {
        const png = new Resvg(resolvedSvg, {
          fitTo: { mode: 'zoom', value: 2 },
        }).render().asPng();
        writeFileSync(join(fileDir, `${name}.png`), png);
      }
      totalDiagrams++;
    } catch (error) {
      console.error(`ERROR: ${file} diagram ${index + 1}: ${error.message}`);
      errors++;
    }
  }
}

console.log(`Rendered ${totalDiagrams} diagrams to ${outDir}/ (${errors} errors)`);
