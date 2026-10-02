/**
 * Runtime smoke test for the serve chat UI.
 *
 * The UI is a Preact tree that only exists once it is mounted, so this drives the
 * real entry point against a stubbed server in a DOM: it proves the bundle
 * mounts, the SSE stream paints, and the turn lands in the log. It also pins the
 * two properties the markdown path owes model output: a turn renders as text
 * until DOMPurify is in place, and a link that survives the sanitizer still
 * loses its script scheme. Everything else (Tailwind classes, Radix behavior) is
 * covered by the manual pass; this is the check that would catch a broken mount,
 * a dead import, or an unsanitized response before a browser ever sees the page.
 *
 * Run: bun test src/web (also wired into scripts/lint-web.sh).
 */

import { afterAll, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';

import { renderMarkdown } from './chat/markdown';
/* The stylesheet server.zig embeds, read as text. A text import declares no
   global and pulls in no Node module, which is what src/web's lint scope
   allows; see scripts/build-web.sh for the committed artifact's provenance. */
import SERVED_STYLESHEET from './style.css' with { type: 'text' };
import { fmtMegabytes, fmtNum, fmtPercent } from './chat/format';
import type { Bubble } from './chat/types';
import type { ComposerProps } from './chat/components/composer';

GlobalRegistrator.register({ url: 'http://127.0.0.1:49453' });

/** A turn body carrying both an injected tag and a script-scheme link. */
const UNSANITIZED_TEXT = '<img src=x onerror=alert(1)> and [a](javascript:alert(2))';

/** What a sanitizer that returns its input hands the DOM: the worst case
 *  `hardenLinks` exists to contain. */
const LINK_PAYLOAD = '<a id="bad" href="javascript:alert(1)">bad</a>'
  + '<a id="good" href="https://example.com/page">good</a>'
  + '<a id="data" href="data:text/html;base64,PHNjcmlwdD4=">data</a>';

const clearCdn = (): void => {
  Reflect.deleteProperty(globalThis, 'marked');
  Reflect.deleteProperty(globalThis, 'DOMPurify');
};

/** Standalone element for the markdown tests: `renderMarkdown` replaces
 *  whatever was in it, so one node serves both. */
const markdownTarget = document.createElement('div');

/** The value, or a thrown error naming what was missing, so a test body reads
 *  as straight-line assertions. */
const present = <T,>(value: T | null | undefined, what: string): T => {
  if (value === null || value === undefined) { throw new Error(`${what} missing`); }
  return value;
};

/** Let Preact flush its render and the stream throttle timer. */
const settle = (ms = 1): Promise<void> =>
  // oxlint-disable-next-line promise/avoid-new -- a timer is the only clock the test needs
  new Promise((resolve) => { setTimeout(resolve, ms); });

/** A callback the tests never assert on, so they share one sink rather than a
 *  dozen empty bodies the lint rules reject. */
const VOID = (..._args: ReadonlyArray<unknown>): void => {
  /* Nothing to record: no test below depends on a callback firing. */
};

const tick = async (times = 4): Promise<void> => {
  for (let index = 0; index < times; index += 1) { await settle(5); }
};

/** Poll for a condition. Preact commits on its own scheduler, so a fixed
 *  number of ticks is a guess the loaded machine can lose. */
const waitFor = async (condition: () => boolean): Promise<boolean> => {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (condition()) { return true; }
    await settle(10);
  }
  return condition();
};

/** `waitFor` for a node that may mount late: Radix renders dialog content
 *  through a Portal from an effect, so the element is absent on the first poll
 *  even when the tree is already mounted. Returns the node, not a boolean. */
const waitForNode = async <T,>(condition: () => T | null, what: string): Promise<T> => {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const found = condition();
    if (found !== null) { return found; }
    await settle(10);
  }
  throw new Error(`${what} never appeared`);
};

afterAll(async () => {
  /* Preact's scheduler drains through `window`, so let the pending work from
     the mounted trees land before the registrator takes the DOM globals away. */
  for (let index = 0; index < 40; index += 1) { await settle(5); }
  await GlobalRegistrator.unregister();
});

/** A host element for a tree the test drives by hand. */
const mountHost = (): HTMLDivElement => {
  document.body.append(document.createElement('div'));
  const host = document.body.lastElementChild;
  if (!(host instanceof HTMLDivElement)) { throw new Error('host element missing'); }
  return host;
};

const sseResponse = (frames: Array<string>): Response => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) { controller.enqueue(encoder.encode(frame)); }
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
};

const MODEL = { data: [{ id: 'test-model', backend: 'cpu', ctx_size: 4096, kv_seq_len: 5, vision: false }] };

/** One finished assistant turn, for the `MessageList` state tests. The literal
 *  is what fixes `role` and `phase`; the return type pins both. */
type StubTurn = { id: number; role: 'assistant'; text: string; phase: 'done' };
const stubTurn = (text: string): StubTurn => ({ id: text.length, role: 'assistant', text, phase: 'done' });

/** Mounting the real tree and waiting out the stream throttle costs more than
 *  the 5s default on a loaded machine; a timeout here is not a verdict. */
const MOUNT_TIMEOUT_MS = 30_000;

const stubServer = (): void => {
  // SAFETY: the stub answers only the three routes the UI calls on this path.
  globalThis.fetch = function (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
    const url = input instanceof Request ? input.url : String(input);
    if (url === '/v1/models') { return Promise.resolve(Response.json(MODEL)); }
    if (url === '/v1/conversations') { return Promise.resolve(Response.json([])); }
    if (url === '/v1/chat' && init.method === 'POST') {
      return Promise.resolve(sseResponse([
        'data: {"t":"The answer "}\n\n',
        'data: {"t":"is 4."}\n\n',
        'data: {"done":true,"n":4,"tps":12.5,"ms":320,"pn":8,"pms":40,"ptps":200.0}\n\n',
        'data: [DONE]\n\n',
      ]));
    }
    return Promise.resolve(new Response('not found', { status: 404 }));
  } as typeof fetch;
};

test('a prompt streams into the log and the model badge resolves', async () => {
  document.body.innerHTML = '<div id="root"></div>';
  stubServer();
  await import('./app');
  await tick();

  const log = document.querySelector('#chat');
  expect(log).not.toBeNull();
  expect(document.querySelector('a[href="#msg"]')).not.toBeNull();
  expect(document.body.textContent).toContain('test-model');

  const input = present(document.querySelector<HTMLTextAreaElement>('#msg'), 'composer input');

  /* The framework installs its own value setter on the node, so the native descriptor
     is the only way to write a value the controlled input will observe. */
  // oxlint-disable-next-line typescript-eslint/unbound-method -- `.call` below binds the node, which is the point
  const nativeSetter = present(Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set, 'textarea value setter');
  // SAFETY: the descriptor is the DOM value setter of a textarea.
  nativeSetter.call(input, 'What is 2+2?');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await tick();

  const form = document.querySelector<HTMLFormElement>('form');
  expect(form).not.toBeNull();
  form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

  // The stream paints on a 60ms throttle, so poll rather than guess a delay.
  expect(await waitFor(() => document.body.textContent.includes('is 4.'))).toBe(true);

  const text = document.body.textContent;
  expect(text).toContain('What is 2+2?');
  expect(text).toContain('The answer ');
  expect(text).toContain('is 4.');
  expect(text).toContain('agave');

  // The thinking placeholder is replaced by the stream, not prepended to it.
  const bodies = [...document.querySelectorAll('.agave-prose')];
  expect(bodies.at(-1)?.textContent).toBe('The answer is 4.');
}, MOUNT_TIMEOUT_MS);

test('a streaming turn replaces the thinking placeholder instead of extending it', async () => {
  /* Imported here, not at the top: react-dom reads the global document when
     it loads, so it has to load after the registrator. */
  const { createRoot } = await import('react-dom/client');
  const { MessageBody } = await import('./chat/components/message');
  const host = mountHost();
  const root = createRoot(host);
  const announced: Array<string> = [];
  const paint = (text: string, phase: Bubble['phase']): void => {
    root.render(<MessageBody text={text} phase={phase} onRendered={function (rendered) { announced.push(rendered); }} />);
  };
  paint('', 'thinking');
  expect(await waitFor(() => host.textContent === '…')).toBe(true);
  paint('The answer', 'streaming');
  expect(await waitFor(() => host.textContent === 'The answer')).toBe(true);
  paint('The answer is 4.', 'streaming');
  expect(await waitFor(() => host.textContent === 'The answer is 4.')).toBe(true);
  // A turn in flight is not a rendered turn, so it announces nothing.
  expect(announced).toEqual([]);
  root.unmount();
  host.remove();
});

test('model text renders as text when the sanitizer has not loaded', () => {
  clearCdn();
  renderMarkdown(markdownTarget, UNSANITIZED_TEXT);
  expect(markdownTarget.querySelector('img')).toBeNull();
  expect(markdownTarget.textContent).toBe(UNSANITIZED_TEXT);
});

/** Whether the conversation-list stub rejects its next call. Module scope, so
 *  the branch is the stub's rather than a test body's. */
const conversationsFetchState = { failing: false };

/** Answers `GET /v1/conversations` and nothing else: it is the single route
 *  `useConversations` calls, and it ignores the input rather than routing on
 *  it, so an unexpected call still reaches the same list. */
const conversationsFetch = function (_input: RequestInfo | URL): Promise<Response> {
  if (conversationsFetchState.failing) { return Promise.reject(new TypeError('Failed to fetch')); }
  return Promise.resolve(Response.json([{ id: 'a', title: 'Chat a', active: true }]));
};

test('a conversation list that fails to refresh keeps the last copy on screen', async () => {
  conversationsFetchState.failing = false;
  const { createRoot } = await import('react-dom/client');
  const { useBubbleLog, useConversations } = await import('./chat/hooks');
  // SAFETY: the stub answers only `GET /v1/conversations`, the one route this hook calls.
  globalThis.fetch = conversationsFetch as typeof fetch;
  const toasts: Array<string> = [];
  let api: { conversations: Array<unknown> | null; loadError: string | null; load: () => Promise<void> } | null = null;
  const host = mountHost();
  const root = createRoot(host);
  const Probe = function Probe(): null {
    /* The real bubble log rather than a stub: `useConversations` forwards it
       to the action callbacks, and this test drives none of them. */
    api = useConversations({
      log: useBubbleLog(),
      /* Nothing below depends on an announcement reaching the reader. */
      announce() { VOID(); },
      pushToast(text: string) { toasts.push(text); },
    });
    return null;
  };
  root.render(<Probe />);
  const live = () => present(api, 'useConversations result');
  expect(await waitFor(() => live().conversations?.length === 1)).toBe(true);

  /* `load` also runs after every turn, so a blip on the refresh must not
     replace a list the reader is already looking at with the retry state. */
  conversationsFetchState.failing = true;
  await live().load();
  expect(live().conversations).toHaveLength(1);
  expect(live().loadError).toBeNull();
  expect(toasts.at(-1)).toContain('Could not refresh the conversation list');
  root.unmount();
  host.remove();
});

test('switching conversations hides the outgoing transcript while it loads', async () => {
  const { createRoot } = await import('react-dom/client');
  const { MessageList } = await import('./chat/components/message-list');
  const host = mountHost();
  const root = createRoot(host);

  /* The outgoing conversation is still in the log when the fetch starts, so
     it stayed readable behind the loader and read as the selected one. */
  root.render(
    <MessageList
      bubbles={[stubTurn('the previous answer')]}
      toasts={[]}
      showStats={false}
      vision={false}
      streaming={false}
      loading
      lastAssistantId={null}
      onRegenerate={VOID}
      onRendered={VOID}
      onRunCommand={VOID}
      onDismissToast={VOID}
    />,
  );
  expect(await waitFor(() => host.textContent.includes('Loading conversation'))).toBe(true);
  /* `MessageBody` paints markdown, and the sanitizer is torn down by an
     earlier test, so the turn's words are not in the DOM. The message group
     is: with the outgoing transcript left on screen the loader would have a
     rendered turn above it. */
  expect(host.querySelectorAll('[role="group"]')).toHaveLength(0);

  /* Once the fetch lands, the transcript takes the column back: both turns
     are rendered and the loader is gone. */
  root.render(
    <MessageList
      bubbles={[stubTurn('the previous answer'), stubTurn('the selected answer')]}
      toasts={[]}
      showStats={false}
      vision={false}
      streaming={false}
      loading={false}
      lastAssistantId={null}
      onRegenerate={VOID}
      onRendered={VOID}
      onRunCommand={VOID}
      onDismissToast={VOID}
    />,
  );
  expect(await waitFor(() => host.querySelectorAll('[role="group"]').length === 2)).toBe(true);
  expect(host.textContent).not.toContain('Loading conversation');
  root.unmount();
  host.remove();
});

test('a link that survives the sanitizer loses its script scheme', () => {
  Object.assign(globalThis, {
    marked: { setOptions () { return undefined; }, parse () { return LINK_PAYLOAD; } },
    DOMPurify: { sanitize (dirty: string) { return dirty; } },
  });
  try {
    renderMarkdown(markdownTarget, LINK_PAYLOAD);
    expect(markdownTarget.querySelector('#bad')?.getAttribute('href')).toBeNull();
    expect(markdownTarget.querySelector('#data')?.getAttribute('href')).toBeNull();
    expect(markdownTarget.querySelector('#good')?.getAttribute('target')).toBe('_blank');
    expect(markdownTarget.querySelector('#good')?.getAttribute('rel')).toBe('noopener noreferrer');
  } finally {
    clearCdn();
  }
});

const COMPOSER_PROPS: ComposerProps = {
  /* The sampling the composer holds until a turn reports otherwise; the tests
     below assert the key's rendering, not a decode, so these are inert. */
  // oxlint-disable-next-line @rikalabs/no-hardcoded-secrets -- a decode bound, not a credential
  sampling: { temperature: 0.7, topP: 0.95, maxTokens: '512', system: '' },
  onSamplingChange(next) { VOID(next); },
  onSubmit(text, image) { VOID(text, image); },
  streaming: false,
  onStop() { VOID(); },
  vision: false,
  pendingImage: null,
  onImageFile(file, label) { VOID(file, label); },
  onRemoveImage() { VOID(); },
  onDropRejected(message) { VOID(message); },
  onClearSystem() { VOID(); },
  tps: null,
  settingsOpen: false,
  onToggleSettings() { VOID(); },
  focusToken: 0,
};

test('the send key wears the solid variant the two surfaces share', async () => {
  /* The brand guide (docs/brand/README.md) assigns `solid` to the send key.
     This mounted `primaryOutline` while the WASM shell wore `solid`, so one
     action looked different on two surfaces the guide holds to one product.
     The outline variant stays right for other actions, so what is pinned is
     the send key's own class, not the variant's existence. */
  const { createRoot } = await import('react-dom/client');
  const { Composer } = await import('./chat/components/composer');
  const host = mountHost();
  const root = createRoot(host);
  root.render(<Composer {...COMPOSER_PROPS} />);
  expect(await waitFor(() => host.querySelector('button[aria-label="Send message"]') !== null)).toBe(true);

  const send = present(host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]'), 'send key');
  expect(send.textContent).toBe('Send');
  expect(send.className).toContain('bg-primary');
  expect(send.className).not.toContain('border-primary');
  /* Still a submit inside the composer form, not a click handler, and disabled
     on an empty composer, which is the resting state a reader meets. */
  expect(send.type).toBe('submit');
  expect(send.form).toBe(present(host.querySelector('form'), 'composer form'));
  expect(send.disabled).toBe(true);
  root.unmount();
  host.remove();
});

test('a streaming turn swaps the send key for the stop key', async () => {
  /* The variant change is scoped to the idle key: the control that replaces it
     is the destructive one, and a composer mid-turn must not offer Send. */
  const { createRoot } = await import('react-dom/client');
  const { Composer } = await import('./chat/components/composer');
  const host = mountHost();
  const root = createRoot(host);
  root.render(<Composer {...COMPOSER_PROPS} streaming />);
  expect(await waitFor(() => host.querySelector('button[aria-label="Stop generation"]') !== null)).toBe(true);

  const stop = present(host.querySelector<HTMLButtonElement>('button[aria-label="Stop generation"]'), 'stop key');
  expect(stop.textContent).toBe('Stop');
  expect(stop.className).toContain('bg-destructive');
  expect(host.querySelector('button[aria-label="Send message"]')).toBeNull();
  root.unmount();
  host.remove();
});

test('the slider takes its radius from the pill token, not the framework default', async () => {
  /* `rounded-full` is a Tailwind default, not a token on the radius ramp, so
     editing --radius-pill could not move the track or the thumb. Both are the
     same 999px today, so this asserts the token's name in the class: that is
     what makes the slider follow the ramp if the ramp ever changes. */
  const { createRoot } = await import('react-dom/client');
  const { Slider } = await import('./ui/slider');
  const host = mountHost();
  const root = createRoot(host);
  root.render(<Slider value={1} onValueChange={VOID} />);
  expect(await waitFor(() => host.querySelector('input[type="range"]') !== null)).toBe(true);

  const track = present(host.querySelector<HTMLInputElement>('input[type="range"]'), 'slider');
  const classes = track.className.split(' ');
  expect(classes).toContain('rounded-pill');
  expect(classes).not.toContain('rounded-full');
  /* One arbitrary variant per class, so each thumb pseudo-element carries its
     own `rounded-pill`; both engines are named or one platform loses the ramp.
     The pseudo-elements differ (`-webkit-slider-thumb` and `-moz-range-thumb`),
     so each is looked up by its exact name. */
  expect(classes).toContain('[&::-webkit-slider-thumb]:rounded-pill');
  expect(classes).toContain('[&::-moz-range-thumb]:rounded-pill');
  /* No thumb fell back to the framework default. */
  expect(classes.filter((name) => name.endsWith('thumb]:rounded-full'))).toHaveLength(0);
  root.unmount();
  host.remove();
});

test('the served stylesheet gives a think block label a fill token, not an edge token', () => {
  /* The summary row painted its hover background with --color-border, which the
     brand guide reserves for control edges. As a fill it put --color-faint text
     at 1.57:1 dark and 1.56:1 light. This reads the stylesheet the server
     embeds and serves, so the assertion is about the rule a reader meets, not
     the source line it came from: editing theme.css without rebuilding fails
     here exactly as `scripts/check-web-artifacts.sh` reports it. */
  const matches = SERVED_STYLESHEET.match(/think-block summary:hover\{[^}]*\}/gu);
  /* Exactly one such rule in the served sheet, or a later one silently wins. */
  expect(matches).toHaveLength(1);
  expect(present(matches, 'think-block hover rule')[0]).toContain('background:var(--color-muted)');
  expect(present(matches, 'think-block hover rule')[0]).not.toContain('--color-border');
});

test('the served sheet insets both surfaces by one gutter token, not two literals', () => {
  /* The header, transcript and composer were `px-6` in the serve UI and
     `px-8` in the shell, so two surfaces meant to read as one product
     indented by different amounts. Both resolve through one token now; a
     literal step back in either sheet fails here. */
  const steps = SERVED_STYLESHEET.match(/\.px-gutter\{padding-inline:var\(--spacing-gutter\)\}/gu);
  expect(steps).toHaveLength(1);
  /* The token ships, or `px-gutter` silently drops the inset. */
  expect(SERVED_STYLESHEET).toContain('--spacing-gutter:24px');
});

test('the empty state sits on the transcript edge instead of centering the mark', () => {
  /* A centered rosette over a centered headline over a centered chip row is
     the shape every generated app opens on. This reads the served rule, so
     editing theme.css without rebuilding fails here the way
     `scripts/check-web-artifacts.sh` reports it. */
  const matches = SERVED_STYLESHEET.match(/\.mark-lg\{[^}]*\}/gu);
  /* Exactly one such rule in the served sheet, or a later one silently wins. */
  expect(matches).toHaveLength(1);
  /* `margin: 0 0 16px`, not `0 auto`: auto is what centers the block. */
  expect(present(matches, 'mark-lg rule')[0]).toContain('margin:0 0 16px');
  expect(present(matches, 'mark-lg rule')[0]).not.toContain('auto');
});

test("a percentage carries the locale's mark, not a pasted one", () => {
  /* German writes the mark tight, French puts a narrow no-break space before
     it, and Arabic-Indic digits carry their own sign: `fmtNum(x * 100) + "%"`
     produced "12,3%" for de-DE and "١٢٫٣%" for ar-EG, both wrong. */
  /* U+00A0, not an ASCII space: CLDR's percent pattern for both locales puts a
     no-break space before the mark, so a pasted "%" lost a separator the
     line wrapping and the screen reader both rely on. */
  expect(fmtPercent(0.123, 1, 'de-DE')).toBe('12,3\u00A0%');
  expect(fmtPercent(0.123, 1, 'fr-FR')).toBe('12,3\u00A0%');
  expect(fmtPercent(0.5, 0, 'en-US')).toBe('50%');
  /* A bidi locale supplies its own percent sign (U+066A) and the isolation
     mark (U+061C) that keeps a trailing Latin sign from being reordered
     around the digits; an appended ASCII "%" had neither. */
  const arabic = fmtPercent(0.5, 0, 'ar-EG');
  expect(arabic).toContain('\u066A');
  expect(arabic).toContain('\u061C');
  expect(arabic).not.toContain('%');
});

test("megabytes keep the locale's unit label and spacing", () => {
  expect(fmtMegabytes(10_485_760, 'de-DE')).toBe('10,0 MB');
  expect(fmtMegabytes(1_572_864, 'en-US')).toBe('1.5 MB');
  /* A grouped size beside a hardcoded " MB" read "1,572,864.0 MB". */
  expect(fmtMegabytes(1_572_864, 'en-US')).not.toContain(',');
});

test('the tok/s readout rounds like the rest of the stats', () => {
  expect(fmtNum(1234.56, 1, 'en-US')).toBe('1,234.6');
  expect(fmtNum(12.34, 2, 'de-DE')).toBe('12,34');
});

test('the About dialog reports the running configuration, not a feature list', async () => {
  /* Three capability bullets ('runs locally', 'opens models', 'streams
     responses') read the same in every generated app and told a reader
     standing at an OpenAI-compatible endpoint nothing they could not see.
     The context window and the KV cache in use are the facts the dialog
     has and the generic list did not. */
  const { createRoot } = await import('react-dom/client');
  const { AboutDialog } = await import('./chat/components/about-dialog');
  const host = mountHost();
  const root = createRoot(host);
  root.render(
    <AboutDialog
      open
      onOpenChange={VOID}
      modelName="qwen2.5-1.5b-instruct-q4k.gguf"
      backendName="metal"
      ctxSize={32_768}
      kvUsed={4096}
    />,
  );
  /* Radix renders the content through a Portal onto document.body, not into
     the host, so the assertion reads the dialog itself. */
  const dialog = await waitForNode(() => document.querySelector('[role="dialog"]'), 'about dialog');
  expect(dialog.textContent).toContain('metal');
  /* 32768 through fmtCtx is '32K', and 4096 is '4K', in the reader's own
     digit script rather than a hardcoded group separator. */
  expect(dialog.textContent).toContain('32K');
  expect(dialog.textContent).toContain('4K');
  expect(dialog.textContent).not.toContain('Runs locally on CPU or GPU');
  root.unmount();
  host.remove();
});

test('the About dialog says what it does not know rather than showing a bare zero', async () => {
  /* An offline server has no `ctx_size`. Rendering '0' next to Window read
     as a real, wrong number, so the row names the missing value instead. */
  const { createRoot } = await import('react-dom/client');
  const { AboutDialog } = await import('./chat/components/about-dialog');
  const host = mountHost();
  const root = createRoot(host);
  root.render(<AboutDialog open onOpenChange={VOID} modelName="" backendName="" ctxSize={0} kvUsed={0} />);
  const dialog = await waitForNode(() => document.querySelector('[role="dialog"]'), 'about dialog');
  expect(dialog.textContent).toContain('unknown');
  expect(dialog.textContent).toContain('n/a');
  root.unmount();
  host.remove();
});

/** The props a `MessageList` renders the toasts with, for the live-region test. */
const TOAST_PROPS = {
  bubbles: [],
  showStats: false,
  vision: false,
  streaming: false,
  loading: false,
  lastAssistantId: null,
  onRegenerate: VOID,
  onRendered: VOID,
  onRunCommand: VOID,
  onDismissToast: VOID,
};

test('a toast reaches a screen reader through a region that was already on the page', async () => {
  /* A live region added to the DOM together with the text it carries is the
     one case assistive tech reliably does not announce, so every toast and
     error notice reached a screen reader as silence. The overlay holding the
     toasts stays mounted and holds the live semantics itself; each toast is an
     addition to it. */
  const { createRoot } = await import('react-dom/client');
  const { MessageList } = await import('./chat/components/message-list');
  const host = mountHost();
  const root = createRoot(host);
  root.render(<MessageList {...TOAST_PROPS} toasts={[]} />);
  await tick(2);
  const region = present(host.querySelector('[role="status"]'), 'toast live region');
  expect(region.getAttribute('aria-live')).toBe('polite');
  expect(region.textContent).toBe('');

  root.render(<MessageList {...TOAST_PROPS} toasts={[{ id: 1, text: 'Could not clear on the server.', level: 'error' }]} />);
  expect(await waitFor(() => region.textContent.includes('Could not clear'))).toBe(true);
  /* One region, and not a second live role per toast. */
  expect(host.querySelectorAll('[role="status"]')).toHaveLength(1);
  expect(host.querySelectorAll('[role="alert"]')).toHaveLength(0);
  root.unmount();
  host.remove();
});

test('the copy key names the outcome it just reported', async () => {
  /* The visible label swapped to "Copied" or "Failed" while the accessible
     name stayed "Copy response", so a screen-reader user pressing the key
     heard no result at all (SC 4.1.3). */
  const { createRoot } = await import('react-dom/client');
  const { Message } = await import('./chat/components/message');
  const host = mountHost();
  const root = createRoot(host);
  root.render(
    <Message
      bubble={{ id: 1, role: 'assistant', text: 'An answer', phase: 'done' }}
      showStats={false}
      canRegenerate={false}
      onRegenerate={VOID}
      onRendered={VOID}
    />,
  );
  await tick(2);
  const copy = present(host.querySelector<HTMLButtonElement>('button[aria-label]'), 'copy key');
  expect(copy.getAttribute('aria-label')).toBe('Copy response');

  /* The stub stands in for the clipboard write the key awaits; the assertion
     is about the accessible name the key takes from the result. */
  Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.resolve() }, configurable: true });
  copy.click();
  expect(await waitFor(() => copy.getAttribute('aria-label') === 'Copy response: Copied')).toBe(true);
  root.unmount();
  host.remove();
});
