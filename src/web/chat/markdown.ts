/* Markdown rendering for model responses.
 *
 *  Security posture, unchanged from the pre-React implementation: marked and
 *  DOMPurify load from a pinned CDN URL with an SRI hash, on the first response
 *  rather than on page load (nothing on the first paint needs them, and the CSP
 *  in src/server/server.zig allowlists exactly that origin). Every HTML string
 *  below is sanitized through DOMPurify before it reaches the DOM, and link
 *  schemes are re-checked afterwards in case a sanitizer gap lets one through
 *  (CWE-79). When the CDN is blocked the renderer falls back to escaped plain
 *  text, and a rebuild is queued for when the libraries land.
 */

const MARKED_URL = 'https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js';
const MARKED_INTEGRITY = 'sha384-zbcZAIxlvJtNE3Dp5nxLXdXtXyxwOdnILY1TDPVmKFhl4r4nSUG1r8bcFXGVa4Te';
const PURIFY_URL = 'https://cdn.jsdelivr.net/npm/dompurify@3.4.14/dist/purify.min.js';
const PURIFY_INTEGRITY = 'sha384-46dPGH1XlTmj7bc50bqLjTdORXs/3EP2QpA/6EWbelYWOY9VGp+87RT61S3Mcslb';
const HLJS_URL = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js';
const HLJS_INTEGRITY = 'sha384-F/bZzf7p3Joyp5psL90p/p89AZJsndkSoGwRpXcZhleCWhd8SnRuoYo4d0yirjJp';
const HLJS_STYLE_URL = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/kimbie-dark.min.css';
const HLJS_STYLE_INTEGRITY = 'sha384-o5F1vUaMNOmou1sQrsWiFo4/QUGSV0svqNZW+EesmKxWC8MpFJcveBhAyfvTHbGb';

/** Give up on a CDN script that neither loads nor errors. A proxy that accepts
 *  the connection and then stalls fires no event, which would leave the load
 *  promise pending forever: the plain-text fallback rendered and the upgrade
 *  never ran. */
const CDN_SCRIPT_TIMEOUT_MS = 10_000;

/**
 * The deferred CDN libraries, read off `globalThis` rather than as bare
 * identifiers. A free `marked` reference throws a ReferenceError until the
 * script that declares it has run, and the first response arrives before that
 * on a cold page; a missing property on globalThis is simply undefined.
 */
const cdn = globalThis as {
  marked?: MarkedStatic;
  DOMPurify?: DOMPurifyStatic;
  hljs?: HljsStatic;
};

let markdownLoad: Promise<boolean> | null = null;
let highlightLoad: Promise<boolean> | null = null;
let markedConfigured = false;

/** Append a pinned CDN script. Resolves false on load failure or timeout; every
 *  caller has a working fallback, so a blocked CDN degrades the page instead of
 *  breaking it. */
function loadCdnScript(url: string, integrity: string): Promise<boolean> {
  return new Promise(function (resolve) {
    const script = document.createElement('script');
    script.src = url;
    script.integrity = integrity;
    script.crossOrigin = 'anonymous';
    script.referrerPolicy = 'no-referrer';
    const timer = setTimeout(function () { settle(false); }, CDN_SCRIPT_TIMEOUT_MS);
    function settle(ok: boolean) {
      clearTimeout(timer);
      resolve(ok);
    }
    script.addEventListener('load', function () { settle(true); });
    script.addEventListener('error', function () { settle(false); });
    document.head.append(script);
  });
}

/** Fetch marked and DOMPurify. Resolves false when either fails; the renderers
 *  then fall back to escaped plain text. */
export function loadMarkdown(): Promise<boolean> {
  if (markdownLoad) {return markdownLoad;}
  if (cdn.marked && cdn.DOMPurify) {return Promise.resolve(true);}
  markdownLoad = Promise.all([loadCdnScript(MARKED_URL, MARKED_INTEGRITY), loadCdnScript(PURIFY_URL, PURIFY_INTEGRITY)]).then(
    function () {return Boolean(cdn.marked && cdn.DOMPurify);},
  );
  return markdownLoad;
}

/** Fetch highlight.js and its theme on the first code block. Copy and language
 *  chrome do not wait for it. */
export function loadHighlightJs(): Promise<boolean> {
  if (highlightLoad) {return highlightLoad;}
  if (cdn.hljs) {return Promise.resolve(true);}
  highlightLoad = new Promise(function (resolve) {
    if (!document.querySelector('link[data-agave-hljs]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = HLJS_STYLE_URL;
      link.integrity = HLJS_STYLE_INTEGRITY;
      link.crossOrigin = 'anonymous';
      link.referrerPolicy = 'no-referrer';
      link.dataset.agaveHljs = '1';
      document.head.append(link);
    }
    loadCdnScript(HLJS_URL, HLJS_INTEGRITY).then(function (ok) { resolve(ok && Boolean(cdn.hljs)); });
  });
  return highlightLoad;
}

function escapeHtmlEntities(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function escapeHtmlText(text: string): string {
  return escapeHtmlEntities(text).replaceAll('\n', '<br>');
}

function markdownToHtml(source: string): string {
  if (!cdn.marked) {return escapeHtmlText(source);}
  if (!markedConfigured) {
    cdn.marked.setOptions({ breaks: true, gfm: true });
    markedConfigured = true;
  }
  try {
    return cdn.marked.parse(source);
  } catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- fall back to escaped plain text instead of killing the response render
    return escapeHtmlText(source);
  }
}

/** Rewrite <think> blocks into a collapsed chain-of-thought disclosure. An
 *  unclosed tag is stripped and escaped so marked cannot treat the remainder as
 *  raw HTML (marked 11 passes HTML through; CWE-79). */
function expandThinkBlocks(content: string): string {
  if (!content.includes('<think>')) {return content;}
  let thinkIndex = 0;
  const expanded = content.replaceAll(/<think>([\s\S]*?)<\/think>\s*/g, function (_match, part: string) {
    const thought = part.trim();
    if (!thought) {return '';}
    thinkIndex += 1;
    return `<details class="think-block"><summary>Chain of thought ${thinkIndex}</summary><div class="think-content">${escapeHtmlText(thought)}</div></details>`;
  });
  if (expanded.startsWith('<think>')) {return escapeHtmlEntities(expanded.slice(7));}
  return expanded;
}

/** Demote headings two levels so a model response cannot outrank the page
 *  structure. Attributes are not copied across (CWE-79): re-applying them
 *  could reintroduce handlers if a sanitizer gap exists. */
function demoteHeadings(root: HTMLElement): void {
  for (const heading of root.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
    const level = Number.parseInt(heading.tagName.charAt(1), 10);
    const next = Math.min(level + 2, 6);
    if (next === level) {continue;}
    const replacement = document.createElement(`h${next}`);
    while (heading.firstChild) {replacement.append(heading.firstChild);}
    heading.parentNode?.replaceChild(replacement, heading);
  }
}

function wrapTables(root: HTMLElement): void {
  for (const table of root.querySelectorAll('table')) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    wrapper.setAttribute('tabindex', '0');
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Data table');
    table.parentNode?.insertBefore(wrapper, table);
    wrapper.append(table);
    for (const header of table.querySelectorAll('th')) {
      if (!header.getAttribute('scope')) {header.setAttribute('scope', 'col');}
    }
  }
}

/** Neutralize active-content URL schemes that may survive sanitizer gaps (CWE-79). */
function hardenLinks(root: HTMLElement): void {
  for (const anchor of root.querySelectorAll('a[href]')) {
    const href = anchor.getAttribute('href') ?? '';
    const lower = href.trim().toLowerCase();
    const isData = lower.startsWith('data:');
    const isSafeDataImage = lower.startsWith('data:image/') && !lower.startsWith('data:image/svg');
    if (lower.startsWith('javascript:') || lower.startsWith('vbscript:') || // oxlint-disable-line no-script-url -- scheme blocklist, not a script URL assignment
        (isData && !isSafeDataImage)) {
      anchor.removeAttribute('href');
      continue;
    }
    if (href && !href.startsWith('#')) {
      anchor.setAttribute('target', '_blank');
      anchor.setAttribute('rel', 'noopener noreferrer');
      if (!anchor.querySelector('.sr-only-newtab')) {
        const tip = document.createElement('span');
        tip.className = 'sr-only sr-only-newtab';
        tip.textContent = ' (opens in new tab)';
        anchor.append(tip);
      }
    }
  }
}

function decorateCodeBlock(block: Element): void {
  const pre = block.parentElement;
  if (!pre || pre.querySelector('.copy-btn')) {return;}
  const lang = block.className.match(/language-(\w+)/)?.[1] ?? '';
  if (lang) {
    const label = document.createElement('span');
    label.className = 'code-lang';
    label.textContent = lang;
    pre.append(label);
  }
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copy-btn';
  copy.textContent = 'Copy';
  copy.setAttribute('aria-label', lang ? `Copy ${lang} code` : 'Copy code');
  copy.addEventListener('click', function () {
    void copyText(block.textContent ?? '').then(function (result) {
      copy.textContent = result === 'copied' ? 'Copied' : 'Failed';
      setTimeout(function () { copy.textContent = 'Copy'; }, 2000);
    });
  });
  pre.append(copy);
}

function highlightCodeBlocks(root: HTMLElement): void {
  const blocks = root.querySelectorAll('pre code');
  if (blocks.length === 0) {return;}
  for (const block of blocks) {decorateCodeBlock(block);}
  const apply = function () {
    if (!cdn.hljs || !root.isConnected) {return;}
    for (const block of root.querySelectorAll('pre code')) {
      if (block.classList.contains('hljs')) {continue;}
      cdn.hljs.highlightElement(block as HTMLElement);
    }
  };
  if (cdn.hljs) {apply(); return;}
  void loadHighlightJs().then(function (ok) { if (ok) {apply();} });
}

/** Copy to the clipboard. The caller owns the label and its revert timer, so
 *  this only reports whether the write landed. */
export function copyText(text: string): Promise<'copied' | 'failed'> {
  return navigator.clipboard.writeText(text).then(
    function () { return 'copied' as const; },
    function () { return 'failed' as const; },
  );
}

/**
 * Render `content` as sanitized markdown into `target`, replacing whatever was
 * there. Callers own the element: the streaming path appends plain text to the
 * same node, this path rebuilds it once the turn is final.
 */
export function renderMarkdown(target: HTMLElement, content: string): void {
  const parsed = markdownToHtml(expandThinkBlocks(content));
  if (!cdn.DOMPurify) {
    // No sanitizer: escape everything rather than trust marked's output.
    target.textContent = content;
    return;
  }
  const sanitized = cdn.DOMPurify.sanitize(parsed, { ADD_TAGS: ['details', 'summary'] });
  target.innerHTML = sanitized;
  highlightCodeBlocks(target);
  demoteHeadings(target);
  wrapTables(target);
  hardenLinks(target);
}

/** True when the markdown libraries are in place, so callers know whether a
 *  plain-text render is the final one. */
export function markdownReady(): boolean {
  return Boolean(cdn.marked && cdn.DOMPurify);
}

const idleQueue: (() => void)[] = [];
let idleDraining = false;

function scheduleIdle(fn: () => void): void {
  if (typeof requestIdleCallback === 'function') { requestIdleCallback(fn); }
  else { setTimeout(fn, 0); }
}

/** Responses that finished before the deferred libraries landed are rebuilt one
 *  per idle slot, so a restored history does not re-render in one task. */
export function onIdle(fn: () => void): void {
  idleQueue.push(fn);
  if (idleDraining) {return;}
  idleDraining = true;
  const drain = function () {
    const next = idleQueue.shift();
    if (!next) { idleDraining = false; return; }
    next();
    if (idleQueue.length > 0) { scheduleIdle(drain); }
    else { idleDraining = false; }
  };
  scheduleIdle(drain);
}
