/** Stream decoding: the SSE body is a byte stream that only becomes text once
 *  a character is whole, so the reader has to hold partial sequences and
 *  release them at the right end.
 *
 *  These drive `streamChat` against a stubbed `fetch`, because the interesting
 *  failures are all about where the chunk boundaries fall: a 4-byte emoji split
 *  across two reads, a body that ends in the middle of a character, and a final
 *  frame whose terminating newline is missing.
 */

import { afterEach, expect, test } from 'bun:test';

import { streamChat } from './api';

const encoder = new TextEncoder();

/** One `read()` result: `value` is always present, so an exhausted reader
 *  reports an empty chunk alongside `done` rather than leaving it undefined. */
type ReadResult = { done: boolean; value: Uint8Array };

/** A 200 `Response` whose body hands out a reader yielding `chunks` one read at
 *  a time, then done. The stream underneath is empty; only the reader matters. */
const stubResponse = (chunks: Array<Uint8Array>): Response => {
  const pending = chunks[Symbol.iterator]();
  // The stream underneath is empty; only the reader below is exercised.
  return new Response(
    Object.defineProperty(
      new ReadableStream<Uint8Array>({
        start: (controller) => { controller.enqueue(new Uint8Array(0)); },
      }),
      'getReader',
      {
        value: () => ({
          read: (): Promise<ReadResult> => {
            const step = pending.next();
            if (step.done === true) {
              return Promise.resolve({ done: true, value: new Uint8Array(0) });
            }
            return Promise.resolve({ done: false, value: step.value });
          },
        }),
      },
    ),
    { status: 200 },
  );
};

const originalFetch = globalThis.fetch;

/** Stats frames seen since it was last zeroed, so the callback has a use and
 *  is not an empty stub. */
let statsSeen = 0;
const countStats = (): void => { statsSeen += 1; };

/** Feed `chunks` through `streamChat` and return the text it painted. */
const painted = async (chunks: Array<Uint8Array>): Promise<string> => {
  // SAFETY: the stub answers only `POST /v1/chat`, the one route under test,
  // And it ignores the request arguments, which this test does not vary.
  globalThis.fetch = function (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    void input;
    void init;
    return Promise.resolve(stubResponse(chunks));
  } as typeof fetch;
  let content = '';
  await streamChat(
    { body: 'message=hi', signal: new AbortController().signal, requestId: 'test' },
    { onText: (next) => { content = next; }, onStats: countStats },
  );
  return content;
};

afterEach(() => { globalThis.fetch = originalFetch; });

test('a character split across two reads is assembled, not dropped', async () => {
  const frame = encoder.encode('data: {"t":"hi \u{1F600}!"}\n');
  const at = frame.indexOf(0xF0);
  expect(await painted([frame.slice(0, at + 2), frame.slice(at + 2)])).toBe('hi \u{1F600}!');
});

test('a character split one byte at a time still arrives whole', async () => {
  const frame = encoder.encode('data: {"t":"\u{1F1FA}\u{1F1F8}"}\n');
  expect(await painted(Array.from(frame, (byte) => Uint8Array.of(byte)))).toBe('\u{1F1FA}\u{1F1F8}');
});

test('a final frame with no trailing newline is still parsed', async () => {
  const body = encoder.encode('data: {"t":"kept"}\ndata: {"t":"and this"}');
  expect(await painted([body])).toBe('keptand this');
});

test('text that arrived before a truncated final frame is not lost', async () => {
  // A body cut mid-character. The completed frame before it carries text, and
  // The trailing half-frame is not valid JSON, so it is dropped deliberately.
  const full = encoder.encode('data: {"t":"survived"}\ndata: {"t":"hi \u{1F600} tail"}\ndata: [DONE]\n');
  const at = full.indexOf(0xF0);
  expect(await painted([full.slice(0, at + 1)])).toBe('survived');
});

test('every two-way split of the body paints the same text', async () => {
  const full = encoder.encode('data: {"t":"café \u{1F600} \u{1F1FA}\u{1F1F8}"}\ndata: {"t":" ok"}\ndata: [DONE]\n');
  for (let cut = 1; cut < full.length; cut += 1) {
    expect(await painted([full.slice(0, cut), full.slice(cut)])).toBe('café \u{1F600} \u{1F1FA}\u{1F1F8} ok');
  }
});

test('the final stats frame is reported once, including without its newline', async () => {
  const body = encoder.encode('data: {"t":"a"}\ndata: {"done":true,"n":2,"tps":1,"ms":3}\ndata: [DONE]\n');

  statsSeen = 0;
  expect(await painted([body])).toBe('a');
  expect(statsSeen).toBe(1);

  // The same body with the stats frame's terminating newline cut off.
  // Cut by length, as `Uint8Array.indexOf` takes a byte, not a string.
  statsSeen = 0;
  const end = body.length - encoder.encode('\ndata: [DONE]\n').length;
  expect(await painted([body.slice(0, end)])).toBe('a');
  expect(statsSeen).toBe(1);
});

test('the stream still stops at [DONE]', async () => {
  const body = encoder.encode('data: {"t":"a"}\ndata: [DONE]\ndata: {"t":"never"}\n');
  expect(await painted([body])).toBe('a');
});