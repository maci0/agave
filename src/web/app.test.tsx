/**
 * Runtime smoke test for the serve chat UI.
 *
 * The UI is a React tree that only exists once it is mounted, so this drives the
 * real entry point against a stubbed server in a DOM: it proves the bundle
 * mounts, the SSE stream paints, and the turn lands in the log. Everything else
 * (Tailwind classes, Radix behavior, the markdown pipeline) is covered by the
 * manual pass; this is the check that would catch a broken mount or a dead
 * import before a browser ever sees the page.
 *
 * Run: bun test src/web (also wired into scripts/lint-web.sh).
 */

import { afterAll, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';

GlobalRegistrator.register({ url: 'http://127.0.0.1:49453' });

afterAll(function () { void GlobalRegistrator.unregister(); });

/** Let React flush its concurrent render and the stream throttle timer. */
const wait = (ms: number): Promise<void> =>
  // oxlint-disable-next-line promise/avoid-new -- a timer is the only clock the test needs
  new Promise(function (resolve) { setTimeout(resolve, ms); });

const tick = async (times = 4): Promise<void> => {
  for (let index = 0; index < times; index += 1) { await wait(5); }
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

test('a prompt streams into the log and the model badge resolves', async function () {
  document.body.innerHTML = '<div id="root"></div>';
  stubServer();
  await import('./app');
  await tick();

  const log = document.querySelector('#chat');
  expect(log).not.toBeNull();
  expect(document.querySelector('a[href="#msg"]')).not.toBeNull();
  expect(document.body.textContent).toContain('test-model');

  const input = document.querySelector<HTMLTextAreaElement>('#msg');
  expect(input).not.toBeNull();
  if (!input) { throw new Error('composer input missing'); }

  // React installs its own value setter on the node, so the native descriptor
  // Is the only way to write a value the controlled input will observe.
  // oxlint-disable-next-line typescript-eslint/unbound-method -- `.call` below binds the node, which is the point
  const { set: nativeSetter } = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value') ?? {};
  if (nativeSetter === undefined) { throw new Error('no value setter'); }
  // SAFETY: the descriptor is the DOM value setter of a textarea.
  nativeSetter.call(input, 'What is 2+2?');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await tick();

  const form = document.querySelector<HTMLFormElement>('form');
  expect(form).not.toBeNull();
  form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

  // The stream paints on a 60ms throttle, so poll rather than guess a delay.
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (document.body.textContent.includes('is 4.')) { break; }
    await wait(10);
  }

  const text = document.body.textContent;
  expect(text).toContain('What is 2+2?');
  expect(text).toContain('The answer ');
  expect(text).toContain('is 4.');
  expect(text).toContain('agave');
});
