/** Shared wire types for the server chat UI. The server is OpenAI-compatible
 *  (`docs/API.md`); only the fields the UI reads are modeled here. */

/** One entry of `GET /v1/models`. */
export type ModelRecord = {
  id: string;
  backend?: string;
  kv_seq_len?: number;
  ctx_size?: number;
  vision?: boolean;
};

/** One entry of `GET /v1/conversations`. */
export type ConvRecord = {
  id: string;
  title?: string;
  active?: boolean;
};

/** One stored turn. */
export type StoredMessage = {
  role: string;
  content: string;
};

/** Body of `POST /v1/conversations` with `action=select`. */
export type ConvMessages = {
  messages?: StoredMessage[];
  cleared?: boolean;
};

/** Final generation statistics, decoded from the last SSE stats frame. */
export type StreamStats = {
  tokens: string;
  tps: string;
  time: string;
  pfTok: string;
  pfMs: string;
  pfTps: string;
};

/** One `data:` frame of a `stream=1` response. */
export type StreamFrame = {
  t?: string;
  done?: boolean;
  n?: number;
  tps?: number;
  ms?: number;
  pn?: number;
  pms?: number;
  ptps?: number;
};

/** A chat turn in the UI. `id` is local and monotonic; the server keys turns by
 *  conversation, not by id. */
export type Bubble = {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  /** Data URL for an attached image on a user turn. */
  image?: string | null;
  /** `thinking` before the first token, `streaming` while it grows, `done`
   *  after the final render, `error` when the request failed. */
  phase: 'thinking' | 'streaming' | 'done' | 'error';
  error?: string;
  stats?: StreamStats;
};

/** A transient message under the log. Errors are alerts, everything else is a
 *  status, which is what decides the live-region politeness. */
export type Toast = {
  id: number;
  text: string;
  level: 'error' | 'info';
  /** Label on the retry control; absent means the toast is dismiss-only. */
  action?: 'Retry';
  /** Called with the action button, after the toast leaves the log. */
  onAction?: () => void;
};

/** Sampling settings the composer and the settings panel share. */
export type Sampling = {
  temperature: number;
  topP: number;
  maxTokens: string;
  system: string;
};
