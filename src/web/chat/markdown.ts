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

/** Give up on a CDN script that neither loads nor errors. A proxy that accepts
 *  the connection and then stalls fires no event, which would leave the load
 *  promise pending forever: the plain-text fallback rendered and the upgrade
 *  never ran. */
const CDN_SCRIPT_TIMEOUT_MS = 10_000;

/** How long a copy key keeps its "Copied" state before it returns to "Copy". */
const COPY_REVERT_MS = 2000;

/**
 * The deferred CDN libraries, read off `globalThis` rather than as bare
 * identifiers. A free `marked` reference throws a ReferenceError until the
 * script that declares it has run, and the first response arrives before that
 * on a cold page; a missing property on globalThis is simply undefined.
 */
/** `setTimeout` handle, which the Bun types call a Timeout object. */
type Timer = ReturnType<typeof setTimeout>;

// SAFETY: the shape is the ambient declaration in globals.d.ts.
// The cast only widens globalThis to carry those optional properties.
// oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- the declaration above is the contract
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
const loadCdnScript = async (url: string, integrity: string): Promise<boolean> => {
  const script = document.createElement('script');
  script.src = url;
  script.integrity = integrity;
  script.crossOrigin = 'anonymous';
  script.referrerPolicy = 'no-referrer';
  // oxlint-disable-next-line promise/avoid-new -- a script load is an event, so it needs a promise to await
  const result = await new Promise<'load' | 'error' | 'timeout'>((resolve) => {
    let settled = false;
    const handles: Array<Timer> = [];
    const settle = function (outcome: 'load' | 'error' | 'timeout') {
      if (settled) { return; }
      settled = true;
      for (const handle of handles) { clearTimeout(handle); }
      resolve(outcome);
    };
    handles.push(setTimeout(() => { settle('timeout'); }, CDN_SCRIPT_TIMEOUT_MS));
    script.addEventListener('load', () => { settle('load'); }, { once: true });
    script.addEventListener('error', () => { settle('error'); }, { once: true });
    document.head.append(script);
  });
  return result === 'load';
};

/** Fetch marked and DOMPurify. Resolves false when either fails; the renderers
 *  then fall back to escaped plain text. */
export const loadMarkdown = (): Promise<boolean> => {
  if (markdownLoad) {return markdownLoad;}
  if (cdn.marked && cdn.DOMPurify) {return Promise.resolve(true);}
  markdownLoad = Promise.all([loadCdnScript(MARKED_URL, MARKED_INTEGRITY), loadCdnScript(PURIFY_URL, PURIFY_INTEGRITY)]).then(
    () => Boolean(cdn.marked && cdn.DOMPurify),
  );
  return markdownLoad;
};

/** Fetch highlight.js on the first code block. Copy and language chrome do not
 *  wait for it. The token colors are theme tokens in src/web/ui/theme.css, so
 *  no highlight.js stylesheet is fetched. */
export const loadHighlightJs = (): Promise<boolean> => {
  if (highlightLoad) {return highlightLoad;}
  if (cdn.hljs) {return Promise.resolve(true);}
  highlightLoad = loadCdnScript(HLJS_URL, HLJS_INTEGRITY).then((ok) => ok && Boolean(cdn.hljs));
  return highlightLoad;
};

const escapeHtmlEntities = (text: string): string =>
  text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

const escapeHtmlText = (text: string): string =>
  escapeHtmlEntities(text).replaceAll('\n', '<br>');

const markdownToHtml = (source: string): string => {
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
};

/** Rewrite <think> blocks into a collapsed chain-of-thought disclosure. An
 *  unclosed tag is stripped and escaped so marked cannot treat the remainder as
 *  raw HTML (marked 11 passes HTML through; CWE-79). */
const expandThinkBlocks = (content: string): string => {
  if (!content.includes('<think>')) {return content;}
  let thinkIndex = 0;
  const expanded = content.replaceAll(/<think>(?<thought>[\s\S]*?)<\/think>\s*/gu, (_match, part: string) => {
    const thought = part.trim();
    if (!thought) {return '';}
    thinkIndex += 1;
    return `<details class="think-block"><summary>Chain of thought ${thinkIndex}</summary><div class="think-content">${escapeHtmlText(thought)}</div></details>`;
  });
  if (expanded.startsWith('<think>')) {return escapeHtmlEntities(expanded.slice(7));}
  return expanded;
};

/** Demote headings two levels so a model response cannot outrank the page
 *  structure. Attributes are not copied across (CWE-79): re-applying them
 *  could reintroduce handlers if a sanitizer gap exists. */
const demoteHeadings = (root: HTMLElement): void => {
  for (const heading of root.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
    const level = Number(heading.tagName.charAt(1));
    const next = Math.min(level + 2, 6);
    if (next === level) {continue;}
    const replacement = document.createElement(`h${next}`);
    while (heading.firstChild) {replacement.append(heading.firstChild);}
    heading.parentNode?.replaceChild(replacement, heading);
  }
};

const wrapTables = (root: HTMLElement): void => {
  for (const table of root.querySelectorAll('table')) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    wrapper.setAttribute('tabindex', '0');
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Data table');
    table.before(wrapper);
    wrapper.append(table);
    for (const header of table.querySelectorAll('th')) {
      if (header.getAttribute('scope') === null) {header.setAttribute('scope', 'col');}
    }
  }
};

/** Neutralize active-content URL schemes that may survive sanitizer gaps (CWE-79). */
const hardenLinks = (root: HTMLElement): void => {
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
};

/** Copy to the clipboard. The caller owns the label and its revert timer, so
 *  this only reports whether the write landed. */
export const copyText = async (text: string): Promise<'copied' | 'failed'> => {
  try {
    await navigator.clipboard.writeText(text);
    return 'copied';
  } catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- the caller renders the failure
    return 'failed';
  }
};

const decorateCodeBlock = (block: Element): void => {
  const pre = block.parentElement;
  if (!pre || pre.querySelector('.copy-btn')) {return;}
  const lang = /language-(?<lang>\w+)/u.exec(block.className)?.groups?.lang ?? '';
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
  /* The key reports its result by swapping its text, so it is a live region:
     A screen reader that is not reading the page still has to hear that the
     clipboard write landed (SC 4.1.3), and the name has to stop saying
     "Copy" once it did. */
  copy.setAttribute('aria-live', 'polite');
  const what = lang === '' ? 'code' : `${lang} code`;
  copy.setAttribute('aria-label', `Copy ${what}`);
  copy.addEventListener('click', () => {
    void copyText(block.textContent).then((result) => {
      const state = result === 'copied' ? 'copied' : 'failed';
      copy.textContent = result === 'copied' ? 'Copied' : 'Failed';
      copy.setAttribute('aria-label', `Copy ${what}, ${state}`);
      setTimeout(() => {
        copy.textContent = 'Copy';
        copy.setAttribute('aria-label', `Copy ${what}`);
      }, COPY_REVERT_MS);
    });
  });
  pre.append(copy);
};

const highlightCodeBlocks = (root: HTMLElement): void => {
  const blocks = root.querySelectorAll('pre code');
  if (blocks.length === 0) {return;}
  for (const block of blocks) {decorateCodeBlock(block);}
  const apply = function () {
    if (!cdn.hljs || !root.isConnected) {return;}
    for (const block of root.querySelectorAll('pre code')) {
      if (block.classList.contains('hljs')) {continue;}
      /* The CDN build carries the common languages only; a fence naming
         another one stays plain text instead of logging a highlight.js warning. */
      const lang = /language-(?<lang>[\w-]+)/u.exec(block.className)?.groups?.lang;
      if (lang !== undefined && cdn.hljs.getLanguage(lang) === undefined) {continue;}
      /* SAFETY: a `pre code` descendant is an HTMLElement, which is what
         highlight.js takes; a selector never returns an SVG. */
      // oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- narrowed by the query above
      cdn.hljs.highlightElement(block as HTMLElement);
    }
  };
  if (cdn.hljs) {apply(); return;}
  void loadHighlightJs().then((ok) => { if (ok) {apply();} });
};

/**
 * Render `content` as sanitized markdown into `target`, replacing whatever was
 * there. Callers own the element: the streaming path appends plain text to the
 * same node, this path rebuilds it once the turn is final.
 */
export const renderMarkdown = (target: HTMLElement, content: string): void => {
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
};

/** True when the markdown libraries are in place, so callers know whether a
 *  plain-text render is the final one. */
export const markdownReady = (): boolean =>
  Boolean(cdn.marked && cdn.DOMPurify);

const idleQueue: Array<() => void> = [];
let idleDraining = false;

const scheduleIdle = (fn: () => void): void => {
  // Safari has no requestIdleCallback until 18. A timeout is the fallback.
  if ('requestIdleCallback' in globalThis) { globalThis.requestIdleCallback(fn); return; }
  setTimeout(fn, 0);
};

const drainIdle = (): void => {
  const task = idleQueue.shift();
  if (!task) { idleDraining = false; return; }
  task();
  if (idleQueue.length > 0) { scheduleIdle(drainIdle); } else { idleDraining = false; }
};

/** Responses that finished before the deferred libraries landed are rebuilt one
 *  per idle slot, so a restored history does not re-render in one task. */
export const onIdle = (fn: () => void): void => {
  idleQueue.push(fn);
  if (idleDraining) {return;}
  idleDraining = true;
  scheduleIdle(drainIdle);
};
