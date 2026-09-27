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

afterAll(function () { GlobalRegistrator.unregister(); });

/** Let React flush its concurrent render and the stream throttle timer. */
const tick = (times = 4): Promise<void> => {
  let chain = Promise.resolve();
  for (let index = 0; index < times; index++) {
    chain = chain.then(function () { return new Promise(function (resolve) { setTimeout(resolve, 5); }); });
  }
  return chain;
};

const sseResponse = (frames: string[]): Response => {
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
  globalThis.fetch = function (input: RequestInfo | URL, init?: RequestInit) {
    const url = String(input);
    if (url === '/v1/models') { return Promise.resolve(Response.json(MODEL)); }
    if (url === '/v1/conversations') { return Promise.resolve(Response.json([])); }
    if (url === '/v1/chat') {
      if (init?.method === 'POST' && String(init.body).includes('%2Fclear')) {
        return Promise.resolve(new Response('', { status: 200 }));
      }
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

  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
  setter?.call(input, 'What is 2+2?');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await tick();

  const form = document.querySelector<HTMLFormElement>('form');
  expect(form).not.toBeNull();
  form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  await tick(8);

  const text = document.body.textContent ?? '';
  expect(text).toContain('What is 2+2?');
  expect(text).toContain('The answer ');
  expect(text).toContain('is 4.');
  expect(text).toContain('agave');
});
