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
import { fmtMegabytes, fmtNum, fmtPercent } from './chat/format';
import type { Bubble } from './chat/types';

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

/** One finished assistant turn, for the `MessageList` state tests. */
const stubTurn = (text: string) => ({ id: text.length, role: 'assistant' as const, text, phase: 'done' as const });

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

test('a conversation list that fails to refresh keeps the last copy on screen', async () => {
  const { createRoot } = await import('react-dom/client');
  const { useConversations } = await import('./chat/hooks');
  let fail = false;
  // SAFETY: the stub answers only `GET /v1/conversations`, the one route
  // this hook calls.
  globalThis.fetch = function (input: RequestInfo | URL): Promise<Response> {
    void input;
    if (fail) { return Promise.reject(new TypeError('Failed to fetch')); }
    return Promise.resolve(Response.json([{ id: 'a', title: 'Chat a', active: true }]));
  } as unknown as typeof fetch;
  const toasts: Array<string> = [];
  let api: { conversations: Array<unknown> | null; loadError: string | null; load: () => Promise<void> } | null = null;
  const host = mountHost();
  const root = createRoot(host);
  function Probe(): null {
    api = useConversations({
      log: null as never,
      announce: function (): void {},
      pushToast: function (text: string) { toasts.push(text); },
    });
    return null;
  }
  root.render(<Probe />);
  const live = () => present(api, 'useConversations result');
  expect(await waitFor(() => live().conversations?.length === 1)).toBe(true);

  // `load` also runs after every turn, so a blip on the refresh must not
  // replace a list the reader is already looking at with the retry state.
  fail = true;
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
  const noop = function (): void {};

  // The outgoing conversation is still in the log when the fetch starts, so
  // it stayed readable behind the loader and read as the selected one.
  root.render(
    <MessageList
      bubbles={[stubTurn('the previous answer')]}
      toasts={[]}
      showStats={false}
      vision={false}
      streaming={false}
      loading
      lastAssistantId={null}
      onRegenerate={noop}
      onRendered={noop}
      onRunCommand={noop}
      onDismissToast={noop}
    />,
  );
  expect(await waitFor(() => host.textContent.includes('Loading conversation'))).toBe(true);
  // `MessageBody` paints markdown, and the sanitizer is torn down by an
  // earlier test, so the turn's words are not in the DOM. The message group
  // is: with the outgoing transcript left on screen the loader would have a
  // rendered turn above it.
  expect(host.querySelectorAll('[role="group"]')).toHaveLength(0);

  // Once the fetch lands, the transcript takes the column back: both turns
  // are rendered and the loader is gone.
  root.render(
    <MessageList
      bubbles={[stubTurn('the previous answer'), stubTurn('the selected answer')]}
      toasts={[]}
      showStats={false}
      vision={false}
      streaming={false}
      loading={false}
      lastAssistantId={null}
      onRegenerate={noop}
      onRendered={noop}
      onRunCommand={noop}
      onDismissToast={noop}
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
