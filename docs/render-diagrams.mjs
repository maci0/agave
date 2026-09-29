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
import path from 'node:path';

/**
 * All 7 colors required, resvg fails silently if any is missing.
 * The paper set from docs/brand/README.md, stepped for a white canvas. Per-role
 * node colors live in the classDef lines of the mermaid blocks; see
 * docs/CONTRIBUTING.md for that palette.
 */
const THEME = {
  bg:      '#ffffff',
  fg:      '#2a2522',
  accent:  '#c4823a',
  line:    '#94908c',
  muted:   '#75716f',
  surface: '#f1efec',
  border:  '#9c938a',
};

/**
 * Layout measures text with Inter metrics; Adwaita Sans is Inter's GNOME cut
 * with the same metrics, so it is the rasterizer's face and the SVG's first
 * local fallback. No webfont is fetched.
 */
const LAYOUT_FONT = 'Inter';
const RASTER_FONT = 'Adwaita Sans';

/**
 * The renderer derives its secondary inks with color-mix(), which resvg cannot
 * draw and which washes the warm ink toward grey. These are the hand-picked
 * steps of the ink ramp: --_text-sec and --_line clear 4.5:1 and 3:1 on white,
 * the pale steps carry no text.
 */
const INK_STEPS = {
  '--_text-sec': THEME.muted,
  '--_text-muted': '#a8a4a0',
  '--_text-faint': '#cbc7c3',
  '--_line': THEME.line,
  '--_arrow': '#504b47',
  '--_node-fill': THEME.surface,
  '--_node-stroke': THEME.border,
  '--_group-hdr': '#f7f5f3',
  '--_inner-stroke': '#e2ded9',
  '--_key-badge': '#eceae6',
};

/** Pin the brand faces and ink steps into the renderer's style block. */
const brandStyle = (svg) => {
  let styled = svg
    .replace(/^\s*@import url\('https:\/\/fonts\.googleapis\.com[^\n]*\n/mu, '')
    .replace(
      /text \{ font-family: 'Inter', system-ui, sans-serif; \}/u,
      `text { font-family: ${LAYOUT_FONT}, '${RASTER_FONT}', system-ui, sans-serif; }`,
    );
  for (const [name, value] of Object.entries(INK_STEPS)) {
    styled = styled.replace(new RegExp(`(?<decl>${name}:\\s*)[^;]+;`, 'u'), `$<decl>${value};`);
  }
  if (styled.includes('color-mix(') || styled.includes('fonts.googleapis')) {
    throw new Error('render-diagrams: renderer style block changed; update brandStyle');
  }
  return styled;
};

/**
 * Replace every var(--name) and var(--name, fallback) with its value. resvg
 * draws no CSS custom property, so an unresolved stroke (every edge uses
 * var(--_line)) would vanish from the PNG. Values come from the theme and the
 * declarations in the renderer's own style block, resolved until none remain.
 */
const VAR_USE = /var\((?<name>--[\w-]+)(?:,\s*[^()]*(?:\([^()]*\))?[^()]*)?\)/gu;
const VAR_DECL = /(?<name>--[\w-]+):\s*(?<value>[^;]+);/gu;
const MAX_VAR_DEPTH = 8;

const resolveVars = (svg, theme) => {
  const values = new Map(Object.entries(theme).map(([name, value]) => [`--${name}`, value]));
  for (const match of svg.matchAll(VAR_DECL)) {
    if (!values.has(match.groups.name)) { values.set(match.groups.name, match.groups.value.trim()); }
  }
  let resolved = svg;
  for (let depth = 0; depth < MAX_VAR_DEPTH && resolved.includes('var(--'); depth += 1) {
    resolved = resolved.replaceAll(VAR_USE, (use, name) => values.get(name) ?? use);
  }
  if (resolved.includes('var(--')) { throw new Error('render-diagrams: unresolved CSS variable'); }
  return resolved;
};

/**
 * The layout engine gives some nested subgraphs a zero-size frame, and their
 * titles then print over each other in the corner. Drop such a frame, its
 * header bar and its title; the nodes inside are laid out and drawn anyway.
 */
const COLLAPSED_GROUP =
  /\s*<rect [^>]*width="0" height="0"[^>]*\/>\s*<rect [^>]*width="0" height="\d+"[^>]*\/>\s*<text [^>]*>.*?<\/text>/gu;
const dropCollapsedGroups = (svg) => svg.replaceAll(COLLAPSED_GROUP, '');

// Extract mermaid blocks from a Markdown file.
const extractDiagrams = (md) => {
  const blocks = [];
  for (const m of md.matchAll(/```mermaid\n(?<source>[\s\S]*?)```/gu)) {
    blocks.push({ source: m.groups.source.trim(), index: blocks.length });
  }
  return blocks;
};

// Parse CLI args
const args = process.argv.slice(2);
const outDir = args.includes('--out-dir') ? args[args.indexOf('--out-dir') + 1] : 'docs/diagrams';
// oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- boolean operands: default-PNG requires falsy-or semantics
const emitPng = args.includes('--png') || !args.includes('--svg');
const emitSvg = args.includes('--svg');

const tutorialDir = 'docs/tutorial';
const found = await Array.fromAsync(new Bun.Glob('*.md').scan(tutorialDir));
const files = found.toSorted((a, b) => a.localeCompare(b));

let totalDiagrams = 0;
let errors = 0;

for (const file of files) {
  const diagrams = extractDiagrams(await Bun.file(path.join(tutorialDir, file)).text());
  if (diagrams.length === 0) { continue; }

  // Bun.write creates missing parent directories, so no mkdir step.
  const fileDir = path.join(outDir, path.basename(file, path.extname(file)));

  for (const { source, index } of diagrams) {
    const name = `diagram-${String(index + 1).padStart(2, '0')}`;
    try {
      const rendered = renderMermaidSVG(source, { ...THEME, font: LAYOUT_FONT });
      const resolvedSvg = resolveVars(dropCollapsedGroups(brandStyle(rendered)), THEME);

      if (emitSvg) {
        await Bun.write(path.join(fileDir, `${name}.svg`), resolvedSvg);
      }
      if (emitPng) {
        await Bun.write(
          path.join(fileDir, `${name}.png`),
          new Resvg(resolvedSvg, { fitTo: { mode: 'zoom', value: 2 }, font: { loadSystemFonts: true, defaultFontFamily: RASTER_FONT, sansSerifFamily: RASTER_FONT } }).render().asPng(),
        );
      }
      totalDiagrams += 1;
    } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- batch renderer: failures counted and reported, never swallowed
      console.error(`ERROR: ${file} diagram ${index + 1}: ${error.message}`);
      errors += 1;
    }
  }
}

console.log(`Rendered ${totalDiagrams} diagrams to ${outDir}/ (${errors} errors)`);
