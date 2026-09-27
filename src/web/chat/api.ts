/** Server calls for the chat UI. Every mutating route carries an idempotency
 *  key (docs/API.md, Idempotency) so a double-click, a proxy replay or a retried
 *  fetch cannot append a second user turn or a second conversation. */

import type { ConvMessages, ConvRecord, ModelRecord, Sampling, StreamFrame, StreamStats } from './types';
import { clampMaxTokens } from './storage';

const FORM_HEADERS = { 'Content-Type': 'application/x-www-form-urlencoded' } as const;

/** Idempotency key for one mutating request. A fresh key is minted per user
 *  action, never per attempt, so an intentional second action still runs. */
export const newRequestId = (): string => {
  // RandomUUID needs a secure context; the UI is also reachable over plain
  // Http on a LAN address, where the fallback keeps every key inside the
  // Server's A-Za-z0-9-_. sanitize set.
  if (typeof crypto.randomUUID === 'function') {return crypto.randomUUID();}
  return `${Date.now().toString(16)}.${Math.random().toString(16).slice(2, 10)}`;
};

/** Map HTTP status codes to short, actionable messages for the chat UI. */
export const httpErrorMessage = (status: number): string => {
  if (status === 400) {return 'The request was rejected. Check your message and settings.';}
  if (status === 413) {return 'Message or image is too large.';}
  if (status === 429) {return 'The server is busy. Wait a moment and try again.';}
  if (status === 503) {return 'The model is not ready yet. Try again shortly.';}
  if (status >= 500) {return 'Something went wrong on the server. Try again.';}
  return `Could not complete the request (error ${status}).`;
};

/** Map fetch and network failures to short, actionable copy, not the engine's
 *  exception text. */
export const userFacingError = (error: unknown): string => {
  if (error instanceof Error) {
    const lower = error.message.toLowerCase();
    if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
      return 'Could not reach the server. Check that it is still running.';
    }
    if (lower === 'empty response body') {
      return 'The server sent an empty reply. Try again.';
    }
    return error.message;
  }
  return String(error);
};

const getJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url);
  // SAFETY: every route below answers with the JSON shape named by the caller,
  // And the server is the same binary that serves this page.
  return (await response.json()) as T;
};

/** Post a conversation action. `id` is required by select and delete only;
 *  `requestId` is the idempotency key, which the mutating routes need. */
const postConversation = async (action: string, id?: string, requestId?: string): Promise<ConvMessages> => {
  const target = id === undefined ? `action=${action}` : `action=${action}&id=${encodeURIComponent(id)}`;
  const headers = requestId === undefined ? FORM_HEADERS : { ...FORM_HEADERS, 'X-Request-Id': requestId };
  const response = await fetch('/v1/conversations', { method: 'POST', headers, body: target });
  // SAFETY: the /v1/conversations POST always answers with the ConvMessages
  // Shape; a non-JSON body surfaces as a parse error the caller toasts.
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
  await fetch('/v1/chat', { method: 'POST', headers: FORM_HEADERS, body: 'message=%2Fclear' });
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
  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {return;}
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) {continue;}
      const payload = line.slice(6);
      if (payload === '[DONE]') {return;}
      let frame: StreamFrame;
      try {
        // SAFETY: the server emits these frames itself (docs/API.md, streaming);
        // A frame that does not match is dropped by the field checks below.
        frame = JSON.parse(payload) as StreamFrame;
      } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- one malformed SSE frame must not kill the stream
        // Oxlint-disable-next-line no-console -- stream diagnostics; toasts would spam the UI per token
        console.warn('SSE parse:', error);
        continue;
      }
      if (frame.t) {
        content += frame.t;
        callbacks.onText(content);
      }
      if (frame.done) {
        callbacks.onStats({
          tokens: String(frame.n),
          tps: (frame.tps ?? 0).toFixed(2),
          time: String(frame.ms),
          pfTok: String(frame.pn),
          pfMs: String(frame.pms),
          pfTps: (frame.ptps ?? 0).toFixed(1),
        });
      }
    }
  }
};
