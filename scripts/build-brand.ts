#!/usr/bin/env bun
/**
 * Build the brand assets in docs/brand/ (see docs/brand/README.md).
 *
 * Usage: BRAND_FONT_BOLD=… BRAND_FONT_REGULAR=… bun scripts/build-brand.ts
 *   BRAND_FONT_BOLD     Adwaita Mono Bold .ttf (fc-match -f '%{file}' 'Adwaita Mono:bold')
 *   BRAND_FONT_REGULAR  Adwaita Mono Regular .ttf
 *
 * Needs uv (text is outlined by scripts/brand-glyphs.py) and rsvg-convert (the
 * social card PNG). The mark is geometry below; every file it appears in, and
 * the data URIs in src/web/ui/theme.css and the two favicons, are derived from
 * it, so a mark change means rerunning this and pasting the printed URIs.
 */
import path from 'node:path';

const findRoot = async (start: string): Promise<string> => {
  let dir = start;
  while (!(await Bun.file(path.join(dir, 'build.zig')).exists())) {
    const parent = path.dirname(dir);
    if (parent === dir) { throw new Error('build-brand: no build.zig above this script'); }
    dir = parent;
  }
  return dir;
};

const ROOT = await findRoot(import.meta.dir);
const OUT = path.join(ROOT, 'docs/brand');

const requireEnv = (name: string): string => {
  const value = Bun.env[name];
  if (value === undefined || value === '') {
    throw new Error(`build-brand: set ${name} to the font file (fc-match -f '%{file}' 'Adwaita Mono')`);
  }
  return value;
};
const FONT_BOLD = requireEnv('BRAND_FONT_BOLD');
const FONT_REGULAR = requireEnv('BRAND_FONT_REGULAR');

/** Palette, mirrored from theme.css and the paper set in docs/brand/README.md. */
const C = {
  night: '#1a1714',
  char: '#231f1c',
  ash: '#2c2825',
  sand: '#ebe3db',
  dust: '#aba39b',
  amber: '#d4a574',
  sage: '#8faa7b',
  signal: '#e57373',
  paper: '#ffffff',
  ink: '#2a2522',
  stone: '#6b625b',
  sageDeep: '#5a7a48',
  amberDeep: '#9a5f22',
  divider: '#3a3430',
} as const;

// ---- The mark ----------------------------------------------------------------
/* Five broad blades radiate from a base point on a 64-unit square. A blade is
   widest at the base and ends in a spine tip; side blades arch outward by an
   amount that grows with their angle. Blades are layered back to front, and
   every front blade cuts a GAP-wide channel into the ones behind it (an SVG
   mask), so the mark is one flat color that still reads as separate leaves. */
const BASE_X = 32;
const BASE_Y = 57;
const GAP = 3;
const HEART_RADIUS = 8;
const ARCH = 0.28;
const DEG = Math.PI / 180;

type Blade = { angle: number; len: number; width: number };

/** Back to front: outer pair, inner pair, center. */
const LAYERS: ReadonlyArray<ReadonlyArray<Blade>> = [
  [
    { angle: -60, len: 34, width: 16 },
    { angle: 60, len: 34, width: 16 },
  ],
  [
    { angle: -27, len: 45, width: 17 },
    { angle: 27, len: 45, width: 17 },
  ],
  [{ angle: 0, len: 52, width: 17 }],
];

const fmt = (n: number): string => String(Math.round(n * 10) / 10);

const bladePath = ({ angle, len, width }: Blade): string => {
  const rad = angle * DEG;
  const at = (along: number, across: number): string => {
    const arch = Math.sign(angle) * (Math.abs(angle) / 90) * ARCH * len * (along / len) ** 2;
    const lx = across + arch;
    const x = BASE_X + lx * Math.cos(rad) + along * Math.sin(rad);
    const y = BASE_Y + lx * Math.sin(rad) - along * Math.cos(rad);
    return `${fmt(x)} ${fmt(y)}`;
  };
  const half = width / 2;
  return (
    `M${at(0, -half)}` +
    `C${at(len * 0.35, -half * 1.02)} ${at(len * 0.75, -half * 0.55)} ${at(len, 0)}` +
    `C${at(len * 0.75, half * 0.55)} ${at(len * 0.35, half * 1.02)} ${at(0, half)}Z`
  );
};

/** A blade outline as a mask cut: filled and stroked GAP wide in black. */
const gapCut = (d: string): string =>
  `<path d="${d}" fill="#000" stroke="#000" stroke-width="${GAP}" stroke-linejoin="round"/>`;

/** The mark's defs and blades in `fill`, ids prefixed with `id`. */
const markBody = (fill: string, id: string): string => {
  const paths = LAYERS.map((layer) => layer.map((blade) => bladePath(blade)));
  const masks = paths
    .slice(0, -1)
    .map((_layer, index) => {
      const front = paths.slice(index + 1).flat().map((d) => gapCut(d)).join('');
      return `<mask id="${id}${index}" maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64"><rect width="64" height="64" fill="#fff"/>${front}</mask>`;
    })
    .join('');
  const groups = paths
    .map((layer, index) => {
      const mask = index < paths.length - 1 ? ` mask="url(#${id}${index})"` : '';
      return `<g${mask}>${layer.map((d) => `<path d="${d}"/>`).join('')}</g>`;
    })
    .join('');
  /* The heart covers the point where every gap converges; the clip sets the
     rosette on its ground line. */
  const heart = `<path d="M${BASE_X - HEART_RADIUS} ${BASE_Y}A${HEART_RADIUS} ${HEART_RADIUS} 0 0 1 ${BASE_X + HEART_RADIUS} ${BASE_Y}Z"/>`;
  const clip = `<clipPath id="${id}c"><rect width="64" height="${BASE_Y}"/></clipPath>`;
  return `<defs>${masks}${clip}</defs><g fill="${fill}" clip-path="url(#${id}c)">${groups}${heart}</g>`;
};

/** The mark scaled so its 64-unit square is `size` px, top-left at x,y. */
const placeMark = (fill: string, x: number, y: number, size: number, id: string): string =>
  `<g transform="translate(${fmt(x)} ${fmt(y)}) scale(${fmt(size / 64)})">${markBody(fill, id)}</g>`;

/** Top of the mark's box so its ground line lands on `baseline`. */
const markTop = (baseline: number, size: number): number => baseline - (BASE_Y * size) / 64;

// ---- Type ----------------------------------------------------------------------
type Outline = { d: string; width: number };

const outline = async (font: string, label: string, size: number, tracking = 0): Promise<Outline> => {
  const script = path.join(ROOT, 'scripts/brand-glyphs.py');
  // Font outlining needs fontTools; there is no in-process TTF reader here.
  const printed = await Bun.$`uv run -q ${script} ${font} ${label} ${size} ${tracking}`.text();
  const [width = '', d = ''] = printed.trim().split('\n');
  const advance = Number(width);
  if (!Number.isFinite(advance) || d === '') {
    throw new Error(`build-brand: brand-glyphs.py printed no outline for "${label}"`);
  }
  return { d: d.replaceAll(/-?\d+\.\d+/gu, (n) => fmt(Number(n))), width: advance };
};

const text = async (font: string, label: string, size: number, fill: string, x: number, y: number): Promise<string> => {
  const outlined = await outline(font, label, size);
  return `<path fill="${fill}" transform="translate(${fmt(x)} ${fmt(y)})" d="${outlined.d}"/>`;
};

const svg = (width: number, height: number, parts: ReadonlyArray<string>, title: string): string =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${title}"><title>${title}</title>${parts.join('')}</svg>\n`;

const write = async (name: string, content: string): Promise<void> => {
  await Bun.write(path.join(OUT, name), content);
};

// ---- Marks ---------------------------------------------------------------------
const markFile = (fill: string): string =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 4 64 54" width="64" height="54" role="img" aria-label="Agave"><title>Agave</title>${markBody(fill, 'm')}</svg>\n`;
await write('mark.svg', markFile(C.sage));
await write('mark-deep.svg', markFile(C.sageDeep));
await write('mark-sand.svg', markFile(C.sand));
await write('mark-ink.svg', markFile(C.ink));

// ---- Lockups -------------------------------------------------------------------
const word = await outline(FONT_BOLD, 'agave', 100, -2);
const tagline = await outline(FONT_REGULAR, 'LLM INFERENCE ENGINE', 22, 3.2);
const LOCKUP = { mark: 122, gap: 26, wordBaseline: 62, tagBaseline: 108, height: 116 };

const lockup = (markFill: string, wordFill: string, tagFill: string): string => {
  const wordX = LOCKUP.mark + LOCKUP.gap;
  const width = Math.ceil(wordX + Math.max(word.width, tagline.width) + 8);
  return svg(
    width,
    LOCKUP.height,
    [
      placeMark(markFill, 0, markTop(LOCKUP.tagBaseline, LOCKUP.mark), LOCKUP.mark, 'l'),
      `<path fill="${wordFill}" transform="translate(${wordX} ${LOCKUP.wordBaseline})" d="${word.d}"/>`,
      `<path fill="${tagFill}" transform="translate(${wordX + 3} ${LOCKUP.tagBaseline})" d="${tagline.d}"/>`,
    ],
    'Agave, LLM inference engine',
  );
};
await write('lockup-dark.svg', lockup(C.sage, C.sand, C.dust));
await write('lockup-light.svg', lockup(C.sageDeep, C.ink, C.stone));

// ---- App icon ------------------------------------------------------------------
await write(
  'app-icon.svg',
  svg(
    512,
    512,
    [
      `<rect width="512" height="512" rx="112" fill="${C.night}"/>`,
      `<rect x="8" y="8" width="496" height="496" rx="104" fill="none" stroke="${C.divider}" stroke-width="4"/>`,
      placeMark(C.sage, 88, 94, 336, 'i'),
    ],
    'Agave',
  ),
);

// ---- Social card (GitHub's 1280x640 preview) -------------------------------------
const CARD = { width: 1280, height: 640, mark: 172, baseline: 292, left: 96 };
const cardWord = await outline(FONT_BOLD, 'agave', 150, -3);
const cardLine = await text(FONT_REGULAR, 'Zig LLM inference. Every kernel, quant and model written here.', 30, C.dust, 98, 430);
const cardBackends = await outline(FONT_REGULAR, 'CPU  METAL  VULKAN  CUDA  ROCM  WEBGPU', 20, 4);
const cardMarkTop = markTop(CARD.baseline, CARD.mark);
await write(
  'social-card.svg',
  svg(
    CARD.width,
    CARD.height,
    [
      `<rect width="${CARD.width}" height="${CARD.height}" fill="${C.night}"/>`,
      `<g opacity="0.07">${placeMark(C.sage, 760, 150, 620, 'w')}</g>`,
      `<rect x="0" y="${CARD.height - 8}" width="${CARD.width}" height="8" fill="${C.amber}"/>`,
      placeMark(C.sage, CARD.left, cardMarkTop, CARD.mark, 'c'),
      `<path fill="${C.sand}" transform="translate(${CARD.left + CARD.mark + 36} ${CARD.baseline})" d="${cardWord.d}"/>`,
      cardLine,
      `<path fill="${C.amber}" transform="translate(98 500)" d="${cardBackends.d}"/>`,
    ],
    'Agave: Zig LLM inference engine',
  ),
);
/* GitHub's social preview takes a PNG. resvg-js is not a repo dependency, so
   the system rasterizer draws it. */
await Bun.$`rsvg-convert -w ${CARD.width} ${path.join(OUT, 'social-card.svg')} -o ${path.join(OUT, 'social-card.png')}`;

// ---- Palette sheet ---------------------------------------------------------------
type Swatch = { name: string; hex: string; role: string; ink: string };
const PRODUCT: ReadonlyArray<Swatch> = [
  { name: 'Night', hex: C.night, role: 'page background', ink: C.sand },
  { name: 'Char', hex: C.char, role: 'cards, bars', ink: C.sand },
  { name: 'Ash', hex: C.ash, role: 'popovers, muted', ink: C.sand },
  { name: 'Sand', hex: C.sand, role: 'text', ink: C.night },
  { name: 'Dust', hex: C.dust, role: 'secondary text', ink: C.night },
  { name: 'Amber', hex: C.amber, role: 'actions, focus', ink: C.night },
  { name: 'Sage', hex: C.sage, role: 'the mark, success', ink: C.night },
  { name: 'Signal', hex: C.signal, role: 'errors', ink: C.night },
];
const PAPER: ReadonlyArray<Swatch> = [
  { name: 'Paper', hex: C.paper, role: 'print, docs', ink: C.ink },
  { name: 'Ink', hex: C.ink, role: 'text on paper', ink: C.sand },
  { name: 'Stone', hex: C.stone, role: 'secondary text', ink: C.sand },
  { name: 'Sage Deep', hex: C.sageDeep, role: 'the mark on paper', ink: C.paper },
  { name: 'Amber Deep', hex: C.amberDeep, role: 'accents on paper', ink: C.paper },
];
const TILE = 136;
const PITCH = 148;
const MARGIN = 32;

const swatchTile = async (swatch: Swatch, x: number, top: number): Promise<string> => {
  const labels = await Promise.all([
    text(FONT_BOLD, swatch.name, 15, swatch.ink, x + 12, top + 28),
    text(FONT_REGULAR, swatch.hex.toUpperCase(), 12, swatch.ink, x + 12, top + 50),
    text(FONT_REGULAR, swatch.role, 11, swatch.ink, x + 12, top + 122),
  ]);
  const frame = `<rect x="${x}" y="${top}" width="${TILE}" height="${TILE}" rx="10" fill="${swatch.hex}" stroke="${C.divider}" stroke-width="1.5"/>`;
  return [frame, ...labels].join('');
};

const swatchRow = async (row: ReadonlyArray<Swatch>, top: number): Promise<string> => {
  const tiles = await Promise.all(row.map((swatch, index) => swatchTile(swatch, MARGIN + index * PITCH, top)));
  return tiles.join('');
};
const paletteParts = await Promise.all([
  text(FONT_REGULAR, 'PRODUCT, DARK', 12, C.dust, MARGIN, 40),
  swatchRow(PRODUCT, 56),
  text(FONT_REGULAR, 'PAPER, LIGHT', 12, C.dust, MARGIN, 240),
  swatchRow(PAPER, 256),
]);
await write('palette.svg', svg(1240, 436, [`<rect width="1240" height="436" rx="16" fill="${C.char}"/>`, ...paletteParts], 'Agave palette'));

// ---- Icon sheet ----------------------------------------------------------------
// Semantic name and Lucide file, in the order src/web/ui/icons.tsx reads.
const ICONS: ReadonlyArray<readonly [string, string]> = [
  ['NewIcon', 'plus'],
  ['ExportIcon', 'download'],
  ['ClearIcon', 'eraser'],
  ['AboutIcon', 'info'],
  ['MenuIcon', 'menu'],
  ['CloseIcon', 'x'],
  ['DeleteIcon', 'trash'],
  ['CopyIcon', 'copy'],
  ['RegenerateIcon', 'refresh-cw'],
  ['AttachImageIcon', 'image'],
  ['SettingsIcon', 'sliders-horizontal'],
  ['JumpToLatestIcon', 'arrow-down'],
];
const COLUMNS = 6;
type IconModule = { __iconData: { node: ReadonlyArray<readonly [string, Readonly<Record<string, string>>]> } };

/** The SVG elements of one Lucide glyph, from the icon module lucide-react ships. */
const glyphElements = async (file: string): Promise<string> => {
  const specifier = path.join(ROOT, `node_modules/lucide-react/dist/esm/icons/${file}.mjs`);
  /* SAFETY: every lucide-react icon module exports __iconData {name, size, node};
     the package's .d.ts covers only the components, so the import is untyped. */
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- untyped dynamic import, shape stated above
  const iconModule = (await import(specifier)) as IconModule;
  // oxlint-disable-next-line no-underscore-dangle -- lucide's own export name
  const elements = iconModule.__iconData.node.map(([tag, attributes]) => {
    const list = Object.entries(attributes)
      .filter(([key]) => key !== 'key')
      .map(([key, value]) => `${key}="${value}"`)
      .join(' ');
    return `<${tag} ${list}/>`;
  });
  return elements.join('');
};

const iconTile = async (name: string, file: string, index: number): Promise<string> => {
  const x = MARGIN + (index % COLUMNS) * 200;
  const y = MARGIN + Math.floor(index / COLUMNS) * 190;
  const [glyph, title, source] = await Promise.all([
    glyphElements(file),
    text(FONT_BOLD, name, 13, C.amber, x + 14, y + 132),
    text(FONT_REGULAR, file, 11, C.dust, x + 14, y + 152),
  ]);
  return [
    `<rect x="${x}" y="${y}" width="184" height="170" rx="10" fill="${C.night}" stroke="${C.divider}" stroke-width="1.5"/>`,
    `<g transform="translate(${x + 68} ${y + 34}) scale(2)" fill="none" stroke="${C.sand}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${glyph}</g>`,
    title,
    source,
  ].join('');
};
const iconTiles = await Promise.all(ICONS.map(([name, file], index) => iconTile(name, file, index)));
await write('icons.svg', svg(1240, 420, [`<rect width="1240" height="420" rx="16" fill="${C.char}"/>`, ...iconTiles], 'Agave icon set'));

// ---- Data URIs for theme.css and the favicons --------------------------------------
const encode = (markup: string, spaces: boolean): string => {
  const escaped = markup
    .replaceAll('"', "'")
    .replaceAll('%', '%25')
    .replaceAll('#', '%23')
    .replaceAll('<', '%3C')
    .replaceAll('>', '%3E');
  return spaces ? escaped.replaceAll(' ', '%20') : escaped;
};
const maskUri = encode(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 4 64 54">${markBody('#000', 'a')}</svg>`, false);
const faviconUri = encode(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="${C.night}"/><g transform="translate(11 11.5) scale(.66)">${markBody(C.sage, 'f')}</g></svg>`,
  true,
);
console.log(`--mark (theme.css):\n${maskUri}\n\nfavicon href (head.html, web/index.html):\n${faviconUri}`);
