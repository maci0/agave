/** The chat UI's state, split by concern.
 *
 *  Each hook owns one slice of the surface so the component that assembles them
 *  stays a layout rather than a store. The logic is the logic the pre-React
 *  the pre-React `app.ts` held at module scope; nothing here changes what the
 *  UI does.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  type StreamRequest,
  clearServerConversation,
  createConversation,
  deleteConversation,
  loadConversations,
  loadModels,
  newRequestId,
  samplingParams,
  selectConversation,
  streamChat,
  userFacingError,
} from './api';
import { loadMarkdown } from './markdown';
import { clearStoredSystemPrompt, readSampling, readShowStats, writeSampling, writeShowStats } from './storage';
import type { Bubble, ConvRecord, ModelRecord, Sampling, StreamStats, Toast } from './types';

/** One paint per window: faster than the eye, far cheaper than one per token. */
const STREAM_FLUSH_MS = 60;
/** The screen reader needs the region emptied before the same text re-announces. */
const ANNOUNCE_DELAY_MS = 100;
/** The tok/s counter is a live region, so it refreshes at most once a second. */
const TPS_UPDATE_MS = 1000;
const MAX_IMAGE_BYTES = 10_485_760;
const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp']);

/** `setTimeout` handle. The DOM lib types it as a number, the Bun types as a
 *  Timeout object; the UI is browser code, so it needs both. */
type Timer = ReturnType<typeof setTimeout>;

export type Announce = (text: string) => void;
export type PushToast = (text: string, level?: Toast['level'], retry?: () => void) => void;

type Announcer = { announcement: string; announce: Announce };

export const useAnnouncer = (): Announcer => {
  const [announcement, setAnnouncement] = useState('');
  const timer = useRef<Timer | null>(null);
  const announce = useCallback((text: string) => {
    if (timer.current !== null) { clearTimeout(timer.current); }
    setAnnouncement('');
    timer.current = setTimeout(function () { setAnnouncement(text); }, ANNOUNCE_DELAY_MS);
  }, []);
  return { announcement, announce };
};

type Toaster = {
  toasts: Array<Toast>;
  pushToast: PushToast;
  dismissToast: (id: number) => void;
};

export const useToasts = (): Toaster => {
  const [toasts, setToasts] = useState<Array<Toast>>([]);
  const nextId = useRef(1);
  const pushToast = useCallback(function (text: string, level: Toast['level'] = 'error', retry?: () => void) {
    const id = nextId.current;
    nextId.current += 1;
    const toast: Toast = retry === undefined ? { id, text, level } : { id, text, level, action: 'Retry', onAction: retry };
    setToasts(function (previous) { return [...previous, toast]; });
  }, []);
  const dismissToast = useCallback(function (id: number) {
    setToasts(function (previous) { return previous.filter(function (toast) { return toast.id !== id; }); });
  }, []);
  return { toasts, pushToast, dismissToast };
};

/** The model the server has loaded, and the context window it reports. */
export const useModelInfo = () => {
  const [model, setModel] = useState<ModelRecord | null>(null);
  const [modelResolved, setModelResolved] = useState(false);
  const [context, setContext] = useState<{ used: number; max: number } | null>(null);
  const [vision, setVision] = useState(false);
  const [visionKnown, setVisionKnown] = useState(false);
  const resolved = useRef(false);

  const refresh = useCallback(async function (): Promise<ModelRecord | null> {
    try {
      const record = await loadModels();
      if (record === null) { return null; }
      resolved.current = true;
      setModel(record);
      setModelResolved(true);
      setVision(record.vision === true);
      setVisionKnown(true);
      setContext({ used: record.kv_seq_len ?? 0, max: record.ctx_size ?? 0 });
      return record;
    } catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- the offline badge already reports it
      return null;
    }
  }, []);

  const markOffline = useCallback(function () {
    // A failed refresh after a name is known only means the numbers
    // Are stale, so a live model never becomes an offline badge.
    if (resolved.current) { return; }
    setModelResolved(false);
  }, []);

  useEffect(function () {
    void refresh().then(function (record) { if (record === null) { markOffline(); } });
  }, [refresh, markOffline]);

  const retry = useCallback(function () {
    setModelResolved(false);
    void refresh().then(function (record) { if (record === null) { markOffline(); } });
  }, [markOffline, refresh]);

  return {
    model,
    modelName: model?.id ?? '',
    backendName: model?.backend ?? '',
    modelResolved,
    context,
    vision,
    visionKnown,
    refresh,
    markOffline,
    retry,
  };
};

export type LogApi = {
  bubbles: Array<Bubble>;
  /** Build a bubble with the next id. */
  allocate: (bubble: Omit<Bubble, 'id'>) => Bubble;
  /** Append a new bubble and hand back its id, for the caller that then
   *  patches it as the turn streams. */
  addTurn: (bubble: Omit<Bubble, 'id'>) => number;
  append: (bubble: Bubble) => void;
  replaceAll: (bubbles: ReadonlyArray<Omit<Bubble, 'id'>>) => void;
  patch: (id: number, change: Partial<Bubble>) => void;
  clear: () => void;
  dropLastAssistant: () => void;
  schedulePaint: (id: number, content: string) => void;
  flushNow: (id: number, content: string, change: Partial<Bubble>) => void;
  lastAssistantId: number | null;
};

/** The bubbles in the log, and the streaming paint queue. */
export const useBubbleLog = (): LogApi => {
  const [bubbles, setBubbles] = useState<Array<Bubble>>([]);
  const nextId = useRef(1);
  const flushTimer = useRef<Timer | null>(null);
  const pendingText = useRef('');

  const allocate = useCallback(function (bubble: Omit<Bubble, 'id'>): Bubble {
    const id = nextId.current;
    nextId.current += 1;
    return { ...bubble, id };
  }, []);

  const append = useCallback(function (bubble: Bubble) {
    setBubbles(function (previous) { return [...previous, bubble]; });
  }, []);

  const addTurn = useCallback(function (bubble: Omit<Bubble, 'id'>): number {
    const id = nextId.current;
    nextId.current += 1;
    setBubbles(function (previous) { return [...previous, { ...bubble, id }]; });
    return id;
  }, []);

  const replaceAll = useCallback(function (next: ReadonlyArray<Omit<Bubble, 'id'>>) {
    nextId.current = 1;
    setBubbles(next.map(function (bubble) { return allocate(bubble); }));
  }, [allocate]);

  const patch = useCallback(function (id: number, change: Partial<Bubble>) {
    setBubbles(function (previous) {
      return previous.map(function (bubble) { return bubble.id === id ? { ...bubble, ...change } : bubble; });
    });
  }, []);

  const clear = useCallback(function () { setBubbles([]); }, []);

  const dropLastAssistant = useCallback(function () {
    setBubbles(function (previous) {
      const index = previous.map(function (bubble) { return bubble.role === 'assistant'; }).lastIndexOf(true);
      return index === -1 ? previous : previous.filter(function (_bubble, at) { return at !== index; });
    });
  }, []);

  /** Queue a streaming paint. The timer arms once and then repaints at a fixed
   *  cadence with whatever text has arrived, so a burst of tokens costs one
   *  render per window rather than one per token. */
  const schedulePaint = useCallback(function (id: number, content: string) {
    pendingText.current = content;
    if (flushTimer.current !== null) {return;}
    flushTimer.current = setTimeout(function () {
      flushTimer.current = null;
      const next = pendingText.current;
      pendingText.current = '';
      patch(id, { text: next, phase: 'streaming' });
    }, STREAM_FLUSH_MS);
  }, [patch]);

  const flushNow = useCallback(function (id: number, content: string, change: Partial<Bubble>) {
    if (flushTimer.current !== null) { clearTimeout(flushTimer.current); flushTimer.current = null; }
    pendingText.current = '';
    patch(id, { text: content, ...change });
  }, [patch]);

  const lastAssistantId = useMemo(function () {
    for (let index = bubbles.length - 1; index >= 0; index -= 1) {
      if (bubbles[index]?.role === 'assistant') { return bubbles[index]?.id ?? null; }
    }
    return null;
  }, [bubbles]);

  return { bubbles, allocate, addTurn, append, replaceAll, patch, clear, dropLastAssistant, schedulePaint, flushNow, lastAssistantId };
};

/** Consume one `stream=1` response into a bubble. */
const runStream = async (
  log: LogApi,
  id: number,
  request: { body: string; url: string | undefined; signal: AbortSignal; errorLabel: string },
  onFinish: (stopped: boolean) => void,
  onToken: (elapsedSeconds: number) => void,
): Promise<void> => {
  let content = '';
  const tokenCount = { value: 0 };
  const started = performance.now();
  let lastTick = 0;
  // Regenerate replays the last turn, so it posts to its own route.
  const target: StreamRequest = { body: request.body, signal: request.signal, requestId: newRequestId() };
  if (request.url !== undefined) { target.url = request.url; }
  try {
    await streamChat(target, {
      onText: function (next) {
        content = next;
        tokenCount.value += 1;
        const now = performance.now();
        if (now - lastTick >= TPS_UPDATE_MS) {
          lastTick = now;
          onToken(tokenCount.value / ((now - started) / 1000));
        }
        log.schedulePaint(id, next);
      },
      onStats: function (stats: StreamStats) { log.patch(id, { stats }); },
    });
    log.flushNow(id, content || 'No response.', { phase: 'done' });
    onFinish(false);
  } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- rendered as an alert in the log
    const stopped = error instanceof Error && error.name === 'AbortError';
    log.flushNow(id, stopped ? (content || 'Stopped.') : `${request.errorLabel}: ${userFacingError(error)}`, {
      phase: stopped ? 'done' : 'error',
      error: `${request.errorLabel}: ${userFacingError(error)}`,
    });
    onFinish(stopped);
  }
};

export type ChatTurn = {
  streaming: boolean;
  tps: number | null;
  send: (body: string, errorLabel: string, url?: string) => void;
  stop: () => void;
};

/** One assistant turn: the request, the abort handle and the rate counter. */
export const useChatTurn = ({ log, announce, onTurnEnd }: { log: LogApi; announce: Announce; onTurnEnd: () => void }): ChatTurn => {
  const [streaming, setStreaming] = useState(false);
  const [tps, setTps] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stopped = useRef(false);

  const stop = useCallback(function () { abortRef.current?.abort(); }, []);

  const send = useCallback(function (body: string, errorLabel: string, url?: string) {
    const turnId = log.addTurn({ role: 'assistant', text: '', phase: 'thinking' });
    // Start the markdown fetch alongside the request: by the last
    // Chunk, marked and DOMPurify are normally already in place.
    void loadMarkdown();
    const controller = new AbortController();
    abortRef.current = controller;
    stopped.current = false;
    setStreaming(true);
    setTps(0);
    announce('Generating response…');
    void runStream(log, turnId, { body, url, signal: controller.signal, errorLabel },
      function (wasStopped) {
        stopped.current = wasStopped;
        abortRef.current = null;
        setStreaming(false);
        setTps(null);
        onTurnEnd();
        announce(wasStopped ? 'Generation stopped.' : 'Response complete.');
      },
      setTps);
  }, [announce, log, onTurnEnd]);

  return { streaming, tps, send, stop };
};

type ConversationsInput = {
  log: LogApi;
  announce: Announce;
  pushToast: PushToast;
  load: () => Promise<void>;
  setLoading: (loading: boolean) => void;
};

/** Selecting, creating, clearing and deleting a conversation. */
const useConversationActions = ({ log, announce, pushToast, load, setLoading }: ConversationsInput) => {
  const seq = useRef(0);
  const open = useCallback(function (id: string) {
    seq.current += 1;
    const mine = seq.current;
    setLoading(true);
    announce('Loading conversation…');
    void selectConversation(id).then(
      function (data) {
        if (mine !== seq.current) {return;}
        setLoading(false);
        log.replaceAll((data.messages ?? []).map(function (message) {
          return { role: message.role === 'user' ? 'user' : 'assistant', text: message.content, phase: 'done' };
        }));
        void load();
      },
      function () {
        if (mine !== seq.current) {return;}
        setLoading(false);
        pushToast('Failed to load conversation. Check that the server is running.', 'error', function () { open(id); });
      },
    );
  }, [announce, load, log, pushToast, setLoading]);

  const startNew = useCallback(function () {
    void createConversation().then(
      function () { log.clear(); void load(); announce('New conversation started'); },
      function () { pushToast('Could not create a new conversation. Check that the server is running.'); },
    );
  }, [announce, load, log, pushToast]);

  const clearAll = useCallback(function () {
    if (log.bubbles.length > 0 && globalThis.confirm('Clear this conversation?') === false) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    void clearServerConversation().then(
      function () { log.clear(); void load(); announce('Conversation cleared'); },
      function () {
        log.clear();
        // Failure styling: every other failed action in the UI toasts in red.
        pushToast('Could not clear on the server. The view was reset locally, but the conversation is still stored.');
      },
    );
  }, [announce, load, log, pushToast]);

  const remove = useCallback(function (id: string) {
    if (!globalThis.confirm('Delete this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    void deleteConversation(id).then(
      function (data) { void load(); if (data.cleared === true) { log.clear(); } },
      function () { pushToast('Could not delete that conversation. Check that the server is running.'); },
    );
  }, [load, log, pushToast]);

  return { open, startNew, clearAll, remove };
};

/** The conversation list, and the actions that mutate it. */
export const useConversations = ({ log, announce, pushToast }: {
  log: LogApi;
  announce: Announce;
  pushToast: PushToast;
}) => {
  const [conversations, setConversations] = useState<Array<ConvRecord> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async function () {
    try {
      setConversations(await loadConversations());
      setLoadError(null);
    } catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- the sidebar shows the retry state
      setLoadError('Could not load conversations.');
    }
  }, []);

  useEffect(function () { void load(); }, [load]);

  const actions = useConversationActions({ log, announce, pushToast, load, setLoading });

  return { conversations, loadError, loading, load, ...actions };
};

export const useSettings = (announce: Announce) => {
  const [sampling, setSampling] = useState<Sampling>(readSampling);
  const [showStats, setShowStats] = useState(readShowStats);
  const [open, setOpen] = useState(false);

  const change = useCallback(function (next: Sampling) {
    setSampling(next);
    writeSampling(next);
  }, []);

  const clearSystem = useCallback(function () {
    setSampling(function (previous) { return { ...previous, system: '' }; });
    clearStoredSystemPrompt();
    announce('System prompt cleared');
  }, [announce]);

  const toggleStats = useCallback(function () {
    setShowStats(function (previous) {
      writeShowStats(!previous);
      return !previous;
    });
  }, []);

  const togglePanel = useCallback(function () {
    setOpen(function (wasOpen) {
      announce(`Settings panel ${wasOpen ? 'closed' : 'opened'}`);
      return !wasOpen;
    });
  }, [announce]);

  return { sampling, change, showStats, toggleStats, open, togglePanel, clearSystem };
};

export const useImageAttachment = ({ vision, visionKnown, announce, pushToast }: {
  vision: boolean;
  visionKnown: boolean;
  announce: Announce;
  pushToast: PushToast;
}) => {
  const [pending, setPending] = useState<string | null>(null);

  useEffect(function () {
    if (visionKnown && !vision && pending !== null) {
      setPending(null);
      pushToast('This model cannot view images.');
    }
  }, [pending, pushToast, vision, visionKnown]);

  const attach = useCallback(function (file: File, label: string) {
    if (visionKnown && !vision) { pushToast('This model cannot view images.'); return; }
    if (!ALLOWED_IMAGE_TYPES.has(file.type)) { pushToast('Unsupported image format. Use JPEG, PNG, GIF, or WebP.'); return; }
    if (file.size > MAX_IMAGE_BYTES) { pushToast('Image too large (max 10 MB).'); return; }
    const reader = new FileReader();
    reader.addEventListener('load', function (event) {
      const encoded = event.target?.result;
      if (encoded === null || encoded === undefined || encoded instanceof ArrayBuffer) {return;}
      setPending(encoded);
      announce(label);
    });
    reader.addEventListener('error', function () { pushToast('Could not read that image. Try another file.'); });
    reader.readAsDataURL(file);
  }, [announce, pushToast, vision, visionKnown]);

  const clear = useCallback(function () {
    setPending(null);
    announce('Image removed');
  }, [announce]);

  return { pending, attach, clear };
};

/** The request body for a user turn, including any attached image. */
export const chatRequestBody = (text: string, image: string | null, sampling: Sampling): string => {
  let body = `message=${encodeURIComponent(text)}&stream=1${samplingParams(sampling)}`;
  if (image !== null) { body += `&image=${encodeURIComponent(image)}`; }
  return body;
};

export const regenerateBody = (sampling: Sampling): string => `stream=1${samplingParams(sampling)}`;
