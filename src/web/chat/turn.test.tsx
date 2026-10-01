/**
 * Re-execution safety of the chat turn hook.
 *
 * `useChatTurn.send` mints a fresh idempotency key per call, so the server
 * treats each call as a distinct logical operation. That is correct for an
 * intentional second turn and destructive for a duplicate one: a second
 * `POST /v1/chat` under a new key appends a second user turn to the stored
 * conversation, and a second `POST /v1/chat/regenerate` pops a second
 * assistant message. The server ledger deduplicates a repeat of the *same*
 * key; nothing upstream of it collapses two clicks into one key.
 *
 * The guard used to be `turn.streaming`, which is React state and is still
 * false in the closure of a second click that lands before the re-render
 * commits. These tests drive the hook directly and call `send` twice in one
 * tick, which is exactly the window that guard missed.
 *
 * Run: bun test src/web (also wired into scripts/lint-web.sh).
 */

import { afterAll, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';

import type { LogApi } from './hooks';
import type { Bubble } from './types';

GlobalRegistrator.register({ url: 'http://127.0.0.1:49453' });

type ChatTurn = {
  streaming: boolean;
  busy: () => boolean;
  tps: number | null;
  send: (body: string, errorLabel: string, url?: string) => void;
  stop: () => void;
};

/** What the stub server saw, so a duplicate is observable rather than inferred. */
type Seen = { urls: Array<string>; keys: Array<string | null>; release: Array<() => void> };

/** A `stream=1` response held open until the test releases it, so the hook is
 *  still mid-turn when the duplicate call arrives. */
const heldStream = (seen: Seen): Response => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      seen.release.push(() => {
        controller.enqueue(encoder.encode('data: {"done":true,"n":1,"tps":1.0,"ms":1,"pn":1,"pms":1,"ptps":1.0}\n\n'));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      });
    },
  });
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
};

/** Counts every stream POST and records its idempotency key. */
const stubServer = (): Seen => {
  const seen: Seen = { urls: [], keys: [], release: [] };
  /* SAFETY: the stub answers only the routes this hook calls, so a request
     reaching it is one this test placed. */
  globalThis.fetch = function (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
    const url = input instanceof Request ? input.url : String(input);
    if (init.method === 'POST') {
      seen.urls.push(url);
      seen.keys.push(new Headers(init.headers ?? {}).get('X-Request-Id'));
      return Promise.resolve(heldStream(seen));
    }
    if (url === '/v1/conversations') { return Promise.resolve(Response.json([])); }
    return Promise.resolve(new Response('not found', { status: 404 }));
  } as typeof fetch;
  return seen;
};

const settle = (ms = 1): Promise<void> =>
  // oxlint-disable-next-line promise/avoid-new -- a timer is the only clock the test needs
  new Promise((resolve) => { setTimeout(resolve, ms); });

/** Poll for a condition: Preact commits on its own scheduler, so a fixed
 *  number of ticks is a guess a loaded machine can lose. */
const waitFor = async (condition: () => boolean): Promise<boolean> => {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (condition()) { return true; }
    await settle(10);
  }
  return condition();
};

const present = <T,>(value: T | null | undefined, what: string): T => {
  if (value === null || value === undefined) { throw new Error(`${what} missing`); }
  return value;
};

/** A recorder the stub log writes every paint into. `flushNow` records the
 *  final text and `onTurnEnd` records the turn finishing, so a run that ends
 *  with one of each has exactly one completed turn. */
type Paints = { calls: Array<string> };

const mountHost = (): HTMLDivElement => {
  document.body.append(document.createElement('div'));
  const host = document.body.lastElementChild;
  if (!(host instanceof HTMLDivElement)) { throw new Error('host element missing'); }
  return host;
};

/** A complete `LogApi`. The hook reads nothing off it that the stub has to
 *  answer, so every method is present; `flushNow` records the final paint so a
 *  test can assert a turn actually finished. */
const stubLog = (flushed: Array<string>): LogApi => {
  const blank: Bubble = { id: 1, role: 'assistant', text: '', phase: 'thinking' };
  return {
    bubbles: [],
    allocate (bubble) { return { ...blank, ...bubble }; },
    addTurn: () => blank.id,
    append () { flushed.push('append'); },
    replaceAll () { flushed.push('replaceAll'); },
    patch () { flushed.push('patch'); },
    clear () { flushed.push('clear'); },
    dropLastAssistant () { flushed.push('dropLastAssistant'); },
    schedulePaint () { flushed.push('schedulePaint'); },
    flushNow (_id, content) { flushed.push(content); },
    lastAssistantId: null,
  };
};

/** Mount the hook and hand back its `send`, its recorder and a teardown. */
const mountTurn = async (): Promise<{ send: ChatTurn['send']; busy: ChatTurn['busy']; paints: Paints; unmount: () => void }> => {
  const { createRoot } = await import('react-dom/client');
  const { useChatTurn } = await import('./hooks');
  const root = createRoot(mountHost());
  const painted: Paints = { calls: [] };
  const settled = (): void => { painted.calls.push('onTurnEnd'); };
  const spoke = (text: string): void => { painted.calls.push(`announce:${text}`); };
  let api: ChatTurn | null = null;
  const Probe = (): null => {
    api = useChatTurn({ log: stubLog(painted.calls), announce: spoke, onTurnEnd: settled });
    return null;
  };
  root.render(<Probe />);
  expect(await waitFor(() => api !== null)).toBe(true);
  return {
    send (body: string, errorLabel: string, url?: string) { present(api, 'turn').send(body, errorLabel, url); },
    busy () { return present(api, 'turn').busy(); },
    paints: painted,
    unmount () { root.unmount(); document.body.lastElementChild?.remove(); },
  };
};

afterAll(async () => {
  for (let index = 0; index < 40; index += 1) { await settle(5); }
  await GlobalRegistrator.unregister();
});

test('a duplicate send in one tick issues one request, not two', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  // Two clicks before the re-render commits: the window `turn.streaming` misses.
  turn.send('message=hello', 'Failed');
  turn.send('message=hello', 'Failed');

  expect(await waitFor(() => seen.urls.length > 0)).toBe(true);
  await settle(50);

  /* One POST. A second would have carried a second idempotency key, so the
     server would have appended a second user turn to the same conversation. */
  expect(seen.urls).toEqual(['/v1/chat']);
  turn.unmount();
});

test('the latch releases when the turn ends, so a later message still runs', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  turn.send('message=first', 'Failed');
  expect(await waitFor(() => seen.urls.length === 1)).toBe(true);
  present(seen.release[0], 'stream release')();
  expect(await waitFor(() => seen.release.length === 1)).toBe(true);
  await settle(80);

  /* The latch is per in-flight turn, not a one-shot: once the first turn ends
     a deliberate second message has to reach the server under its own key. */
  seen.urls.length = 0;
  turn.send('message=second', 'Failed');
  expect(await waitFor(() => seen.urls.length === 1)).toBe(true);
  expect(seen.keys.at(-1)).toBeTruthy();
  turn.unmount();
});

test('a duplicate regenerate is refused before it pops a second message', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  /* Regenerate is the destructive one: each accepted call pops an assistant
     message from the stored conversation, so two calls under two keys
     destroyed two turns. */
  turn.send('stream=1', 'Failed to regenerate', '/v1/chat/regenerate');
  turn.send('stream=1', 'Failed to regenerate', '/v1/chat/regenerate');

  expect(await waitFor(() => seen.urls.length > 0)).toBe(true);
  await settle(50);

  expect(seen.urls).toEqual(['/v1/chat/regenerate']);
  turn.unmount();
});

test('a refused duplicate leaves the in-flight turn to finish on its own', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  turn.send('message=hello', 'Failed');
  turn.send('message=hello', 'Failed');
  expect(await waitFor(() => seen.urls.length === 1)).toBe(true);
  await settle(20);

  /* The suppressed duplicate must not abort the turn the reader is watching:
     a second `send` allocates a second bubble and replaces the shared abort
     controller, so an unguarded duplicate truncated the first answer and left
     two bubbles behind. One `onTurnEnd` means exactly one turn finished. */
  expect(seen.release).toHaveLength(1);
  seen.release[0]();
  expect(await waitFor(() => turn.paints.calls.filter((call) => call === 'onTurnEnd').length === 1)).toBe(true);

  /* With the turn over the latch is free again, which is what the retry
     affordance on the finished bubble depends on. */
  turn.send('message=again', 'Failed');
  expect(await waitFor(() => seen.urls.length === 2)).toBe(true);
  turn.unmount();
});
test('busy answers inside the click that started the turn', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  /* `regenerate` reads this before it drops the finished reply, so a state
     flag would report false here and the drop would run against a turn the
     latch is about to refuse. */
  expect(turn.busy()).toBe(false);
  turn.send('message=hello', 'Failed');
  expect(turn.busy()).toBe(true);

  expect(await waitFor(() => seen.release.length === 1)).toBe(true);
  seen.release[0]();
  expect(await waitFor(() => turn.busy() === false)).toBe(true);
  turn.unmount();
});

test('a regenerate refused while busy never reaches the wire', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  /* Regenerate drops the finished reply before it sends, so the guard has to
     hold before the drop. Read here, it reports the in-flight turn and the
     caller leaves the log alone; a state flag reports false in this same tick
     and the drop runs against the answer now streaming into that slot. */
  turn.send('message=first', 'Failed');
  expect(await waitFor(() => seen.urls.length === 1)).toBe(true);
  seen.release[0]();
  expect(await waitFor(() => turn.busy() === false)).toBe(true);

  turn.send('stream=1', 'Failed to regenerate', '/v1/chat/regenerate');
  expect(turn.busy()).toBe(true);
  expect(await waitFor(() => seen.urls.filter((url) => url === '/v1/chat/regenerate').length === 1)).toBe(true);

  // The duplicate regenerate, arriving while the accepted one streams.
  turn.send('stream=1', 'Failed to regenerate', '/v1/chat/regenerate');
  await settle(50);

  /* Only the accepted regenerate reached the wire. The duplicate issued no
     second request, so the single reply it would have dropped has a single
     replacement rather than two turns racing for the same slot. */
  expect(seen.urls.filter((url) => url === '/v1/chat/regenerate')).toHaveLength(1);
  turn.unmount();
});
/** A server that answers the next POST with a status, so a turn fails the way
 *  a real one does: `streamChat` throws before it reads a body, which is the
 *  path the failure announcement is written for. */
const stubFailingServer = (status: number): Seen => {
  const seen: Seen = { urls: [], keys: [], release: [] };
  /* SAFETY: the stub answers only the routes this hook calls, so a request
     reaching it is one this test placed. */
  globalThis.fetch = function (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
    const url = input instanceof Request ? input.url : String(input);
    if (init.method === 'POST') {
      seen.urls.push(url);
      seen.keys.push(new Headers(init.headers ?? {}).get('X-Request-Id'));
      return Promise.resolve(new Response('busy', { status }));
    }
    if (url === '/v1/conversations') { return Promise.resolve(Response.json([])); }
    return Promise.resolve(new Response('not found', { status: 404 }));
  } as typeof fetch;
  return seen;
};

test('a failed turn announces the failure, not that it completed', async () => {
  const seen = stubFailingServer(503);
  const turn = await mountTurn();

  turn.send('message=hello', 'Send failed');

  expect(await waitFor(() => turn.paints.calls.some((call) => call.startsWith('announce:Send failed')))).toBe(true);

  /* "Response complete." on a turn that failed is the one sentence a screen
     reader is guaranteed to hear, and it contradicted the bubble beside it. The
     announcement is the message the bubble carries, so the region does not
     read a second copy of a paragraph nobody asked for either. */
  const said = turn.paints.calls.filter((call) => call.startsWith('announce:'));
  expect(said).not.toContain('announce:Response complete.');
  expect(said).toContain('announce:Send failed: The model is not ready yet. Try again shortly.');
  expect(seen.urls).toEqual(['/v1/chat']);
  turn.unmount();
});

test('a turn that succeeded still announces completion', async () => {
  const seen = stubServer();
  const turn = await mountTurn();

  /* The failure announcement replaced the success one, so the passing path
     needs its own guard: a run that announced nothing on success would leave
     a screen-reader user waiting on a turn that had already landed. */
  turn.send('message=hello', 'Send failed');
  expect(await waitFor(() => seen.release.length === 1)).toBe(true);
  seen.release[0]();

  expect(await waitFor(() => turn.paints.calls.includes('announce:Response complete.'))).toBe(true);
  const said = turn.paints.calls.filter((call) => call.startsWith('announce:'));
  expect(said).toContain('announce:Generating response…');
  expect(said).not.toContain('announce:Generation stopped.');
  turn.unmount();
});
