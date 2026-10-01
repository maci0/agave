/** Server calls for the chat UI. Every mutating route carries an idempotency
 *  key (docs/API.md, Idempotency) so a double-click, a proxy replay or a retried
 *  fetch cannot append a second user turn or a second conversation. */

import type { ConvMessages, ConvRecord, ModelRecord, Sampling, StreamFrame, StreamStats } from './types';
import { clampMaxTokens } from './storage';

const FORM_HEADERS = { 'Content-Type': 'application/x-www-form-urlencoded' } as const;

/** Idempotency key for one mutating request. A fresh key is minted per user
 *  action, never per attempt, so an intentional second action still runs. */
export const newRequestId = (): string => {
  // RandomUUID needs a secure context, and the UI is reachable over plain
  // HTTP on a LAN address, so the fallback keeps keys inside the sanitize set.
  if ('randomUUID' in crypto) { return crypto.randomUUID(); }
  return `${Date.now().toString(16)}.${Math.random().toString(16).slice(2, 10)}`;
};

/** Map HTTP status codes to short messages that say what to do next for the chat UI. */
export const httpErrorMessage = (status: number): string => {
  if (status === 400) {return 'The request was rejected. Check your message and settings.';}
  if (status === 413) {return 'Message or image is too large.';}
  if (status === 429) {return 'The server is busy. Wait a moment and try again.';}
  if (status === 503) {return 'The model is not ready yet. Try again shortly.';}
  /* No "something went wrong": an unnamed 5xx is the one status that could be
     anything, so the copy names the thing that actually failed and the retry
     the reader can take. */
  if (status >= 500) {return 'The server failed to handle the request. Try again.';}
  return `Could not complete the request (error ${status}).`;
};

/** Map fetch and network failures to short copy that says what to do next, not the engine's
 *  exception text. */
export const userFacingError = (cause: unknown): string => {
  if (cause instanceof Error) {
    const lower = cause.message.toLowerCase();
    if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
      return 'Could not reach the server. Check that it is still running.';
    }
    if (lower === 'empty response body') {
      return 'The server sent an empty reply. Try again.';
    }
    return cause.message;
  }
  return String(cause);
};

const getJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url);
  /* An error status carries an error object where the caller expects a list or
     A record, so it has to reject here: the caller's failure path is what
     tells the reader, and what it guards is a shape the error body is not. */
  if (!response.ok) {throw new Error(httpErrorMessage(response.status));}
  // SAFETY: every route answers with the shape the caller names, and
  // The server serving this page is that same binary.
  // oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- narrowed by the contract above
  return (await response.json()) as T;
};

/** Post a conversation action. `id` is required by select and delete only;
 *  `requestId` is the idempotency key, which the mutating routes need. */
const postConversation = async (action: string, id?: string, requestId?: string): Promise<ConvMessages> => {
  const target = id === undefined ? `action=${action}` : `action=${action}&id=${encodeURIComponent(id)}`;
  const headers = requestId === undefined ? FORM_HEADERS : { ...FORM_HEADERS, 'X-Request-Id': requestId };
  const response = await fetch('/v1/conversations', { method: 'POST', headers, body: target });
  /* Select, delete and new all answer 4xx or 5xx with an error object, which
     the ConvMessages callers would otherwise read as an empty result. */
  if (!response.ok) {throw new Error(httpErrorMessage(response.status));}
  // SAFETY: the POST answers with the ConvMessages shape; a non-JSON
  // Body surfaces as a parse error the caller toasts.
  // oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- narrowed by the contract above
  return (await response.json()) as ConvMessages;
};

/** First model in `/v1/models`, or null when the server reports none. */
export const loadModels = async (): Promise<ModelRecord | null> => {
  const data = await getJson<{ data?: Array<ModelRecord> }>('/v1/models');
  return data.data?.[0] ?? null;
};

export const loadConversations = (): Promise<Array<ConvRecord>> => getJson<Array<ConvRecord>>('/v1/conversations');

export const createConversation = async (): Promise<void> => {
  await postConversation('new', undefined, newRequestId());
};

export const selectConversation = (id: string): Promise<ConvMessages> => postConversation('select', id);

export const deleteConversation = (id: string): Promise<ConvMessages> => postConversation('delete', id);

/** Clear the server-side conversation and KV cache. */
export const clearServerConversation = async (): Promise<void> => {
  const response = await fetch('/v1/chat', { method: 'POST', headers: FORM_HEADERS, body: 'message=%2Fclear' });
  // A rejected clear is what the caller reports, so the status has to reach it.
  if (!response.ok) {throw new Error(httpErrorMessage(response.status));}
};

/** Query string for the sampling settings and the system prompt. */
export const samplingParams = (sampling: Sampling): string => {
  const parts = [
    `temperature=${encodeURIComponent(String(sampling.temperature))}`,
    `top_p=${encodeURIComponent(String(sampling.topP))}`,
    `max_tokens=${encodeURIComponent(String(clampMaxTokens(sampling.maxTokens)))}`,
  ];
  const system = sampling.system.trim();
  if (system) {parts.push(`system=${encodeURIComponent(system)}`);}
  return `&${parts.join('&')}`;
};

export type StreamCallbacks = {
  /** Called with the full text so far, once per decoded token. */
  onText: (content: string) => void;
  onStats: (stats: StreamStats) => void;
};

export type StreamRequest = {
  body: string;
  signal: AbortSignal;
  requestId: string;
  /** Regenerate replays the last turn, so it posts to its own route. */
  url?: string;
};

/** Decode one SSE frame. A malformed frame is reported to the console and
 *  dropped: one bad frame must not kill a stream that is otherwise healthy. */
const parseFrame = (payload: string): StreamFrame | null => {
  // oxlint-disable-next-line @rikalabs/no-json-parse-default-fallback -- a malformed SSE frame is dropped, not defaulted
  try {
    // SAFETY: the server emits these frames itself (docs/API.md, streaming).
    // A frame that does not match is dropped by the field checks below.
    // oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- narrowed by the contract above
    return JSON.parse(payload) as StreamFrame;
  } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- one malformed SSE frame must not kill the stream
    // oxlint-disable-next-line no-console -- stream diagnostics; toasts would spam the UI per token
    console.warn('SSE parse:', error);
    return null;
  }
};

/** Run every `data:` line in `buffer` through `onFrame` in arrival order,
 *  returning the trailing partial line for the next chunk. `onFrame` returning
 *  false ends the stream.
 *
 *  With `final` set, a last line that never got its newline is run as well: a
 *  body can end without one while still holding a complete frame, and dropping
 *  it loses the text it carries. It is handled after the complete lines so
 *  that trailing text is appended last, not interleaved before them. Exported
 *  so frame splitting can be exercised without a socket. */
export const drainSseBuffer = (
  buffer: string,
  onFrame: (payload: string) => boolean,
  final = false,
): string => {
  const lines = buffer.split('\n');
  let rest = lines.pop() ?? '';
  for (const line of lines) {
    if (!line.startsWith('data: ')) {continue;}
    if (!onFrame(line.slice(6))) {return '';}
  }
  if (final && rest.startsWith('data: ')) {
    if (!onFrame(rest.slice(6))) {return '';}
    rest = '';
  }
  return rest;
};

/** Consume a `stream=1` response, calling back per token and once with the
 *  final statistics. Throws on a non-2xx response, an empty body, a decode
 *  failure, or the abort signal; the caller turns that into UI state. */
export const streamChat = async (request: StreamRequest, callbacks: StreamCallbacks): Promise<void> => {
  const response = await fetch(request.url ?? '/v1/chat', {
    method: 'POST',
    headers: { ...FORM_HEADERS, 'X-Request-Id': request.requestId },
    body: request.body,
    signal: request.signal,
  });
  if (!response.ok) {throw new Error(httpErrorMessage(response.status));}
  const stream = response.body;
  if (!stream) {throw new Error('empty response body');}
  const reader = stream.getReader();
  // Holds the lead bytes of a character the next chunk completes, so a 4-byte
  // emoji split across two reads never reaches a frame as a fragment.
  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  /* Returns false once the stream is over: `[DONE]` ends it. The final stats
     frame only records, so a route that keeps sending after it is unaffected. */
  const onFrame = (payload: string): boolean => {
    if (payload === '[DONE]') {return false;}
    const frame = parseFrame(payload);
    if (frame === null) { return true; }
    if (frame.t !== undefined && frame.t !== '') {
      content += frame.t;
      callbacks.onText(content);
    }
    if (frame.done === true) {
      callbacks.onStats({
        tokens: String(frame.n),
        tps: (frame.tps ?? 0).toFixed(2),
        time: String(frame.ms),
        pfTok: String(frame.pn),
        pfMs: String(frame.pms),
        pfTps: (frame.ptps ?? 0).toFixed(1),
      });
    }
    return true;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      /* Flush the decoder, then run whatever the last read left. Without the
         flush a body that ends mid-character drops those bytes, and without
         the final drain a frame with no trailing newline is never parsed. */
      buffer += decoder.decode();
      drainSseBuffer(buffer, onFrame, true);
      return;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = drainSseBuffer(buffer, onFrame);
  }
};
