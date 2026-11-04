/** Model loading for the browser WASM shell.
 *
 *  Owns everything between "the reader asked for a model" and "the engine can
 *  take a prompt": the URL field, the file drop zone, the download progress
 *  line, and the error copy. The conversation itself lives in `shell.tsx`.
 */

/* global AgaveEngine */

import { useCallback, useRef, useState, type DragEvent } from 'react';
import { friendlyLoadError } from './load-errors';
import { fmtMegabytes, fmtPercent } from '../src/web/chat/format';

const IDLE_HINT = 'Load a GGUF model to begin';
/* The status names the state the engine is actually in. "Ready" read as "a
   chat is ready"; the browser build tokenizes prompts and does not generate,
   so the line says which one it is. */
const READY_HINT = 'Model loaded. Prompts are tokenized, not answered.';
const GGUF_MAGIC = [0x47, 0x47, 0x55, 0x46] as const;

export type Report = (text: string, level: 'error' | 'info') => void;

export type ModelLoader = {
  status: string;
  ready: boolean;
  loading: boolean;
  url: string;
  urlError: string | null;
  /** File name of the model that loaded, or null before the first one does. */
  modelName: string | null;
  busy: boolean;
  /** Aborts the in-flight model load, when one is running. */
  cancelLoad: () => void;
  /** True while a file hovers the drop zone, for its highlighted state. */
  dragOver: boolean;
  onDragOver: (event: DragEvent) => void;
  onDragLeave: (event: DragEvent) => void;
  onDrop: (event: DragEvent) => void;
  setUrl: (next: string) => void;
  loadFromUrl: () => void;
  loadFromBuffer: (file: File) => void;
  onReport: Report;
  focusPrompt: () => void;
};

const isGgufName = (name: string): boolean => name.toLowerCase().endsWith('.gguf');

const isGgufBuffer = (modelBytes: ArrayBuffer): boolean => {
  if (modelBytes.byteLength < GGUF_MAGIC.length) {return false;}
  const head = new Uint8Array(modelBytes, 0, GGUF_MAGIC.length);
  return head[0] === GGUF_MAGIC[0] && head[1] === GGUF_MAGIC[1] && head[2] === GGUF_MAGIC[2] && head[3] === GGUF_MAGIC[3];
};

const isHttpUrl = (candidate: string): boolean => {
  try {
    const parsed = new URL(candidate);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- an unparseable URL is the error path, reported by the caller
    return false;
  }
};

/** The file name at the end of a model URL, so a download and a dropped file
 *  read the same in the bar that names the loaded model. */
const nameFromUrl = (candidate: string): string => {
  const tail = new URL(candidate).pathname.split('/').pop() ?? '';
  return tail === '' ? candidate : tail;
};

type LoadState = {
  setStatus: (text: string) => void;
  setReady: (ready: boolean) => void;
  setLoading: (loading: boolean) => void;
  setUrlError: (message: string | null) => void;
};

/** The engine handshake and the failure report, shared by both load paths. */
const useEngineInit = (engine: AgaveEngine, state: LoadState, onReport: Report, focusPrompt: () => void, abortRef: { current: AbortController | null }) => {
  const hadModel = useRef(false);
  /* `load` receives the in-flight `init()` promise instead of waiting on it up
     front: fetching and compiling agave.wasm does not need the model bytes, so
     the (often multi-hundred-MB) model download starts immediately and overlaps
     the WASM round trip instead of queueing behind it. The caller awaits
     `engineReady()` right before `loadModel`, which is the first step that
     actually needs an instantiated module. */
  return useCallback(async (load: (engineReady: () => Promise<void>, signal: AbortSignal) => Promise<void>, fromUrl: boolean) => {
    hadModel.current = engine.hasModel;
    state.setLoading(true);
    state.setReady(false);
    state.setStatus('Initializing engine…');
    const controller = new AbortController();
    abortRef.current = controller;
    /* Mark the eager `init()` promise handled so a load that throws before it
       reaches `engineReady` (a dead model URL, say) cannot leave an unhandled
       rejection behind; the original promise still rejects for the await. */
    const engineReady = engine.ready ? Promise.resolve() : engine.init();
    engineReady.catch(() => undefined);
    try {
      await load(() => engineReady, controller.signal);
      onReport(engine.initMessage || 'Model loaded', 'info');
      state.setStatus(READY_HINT);
      state.setReady(true);
      /* After the commit that enables #prompt: focusing while disabled
         drops to body in browsers, so the focus must land post-render. */
      setTimeout(() => { focusPrompt(); }, 0);
    } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- reported to the reader, not swallowed
      const cancelled = error instanceof Error && error.name === 'AbortError';
      state.setStatus(cancelled ? 'Model load cancelled.' : friendlyLoadError(error));
      onReport(cancelled ? 'Model load cancelled.' : friendlyLoadError(error), cancelled ? 'info' : 'error');
      if (fromUrl && !cancelled) { state.setUrlError(friendlyLoadError(error)); }
      // A failed reload must not disable chat if the previous model is still loaded.
      if (hadModel.current && engine.hasModel) { state.setReady(true); }
    } finally {
      if (abortRef.current === controller) { abortRef.current = null; }
      state.setLoading(false);
    }
  }, [abortRef, engine, focusPrompt, onReport, state]);
};

/** The two ways a model arrives: a URL to fetch, or a file the reader dropped. */
const useModelSources = (
  engine: AgaveEngine,
  state: LoadState,
  onReport: Report,
  failUrl: (message: string) => void,
  focusPrompt: () => void,
  abortRef: { current: AbortController | null },
) => {
  const initAndLoad = useEngineInit(engine, state, onReport, focusPrompt, abortRef);
  const [url, setUrl] = useState('');
  /* Set only after the engine accepts the bytes, so the bar never names a model
     that failed to load: a stale name would read as a working model. */
  const [modelName, setModelName] = useState<string | null>(null);

  const loadFromUrl = useCallback(() => {
    const target = url.trim();
    if (!target) { failUrl('Enter a model URL first'); return; }
    if (!isHttpUrl(target)) { failUrl('Enter a valid http(s) URL to a GGUF file'); return; }
    state.setUrlError(null);
    void initAndLoad(async (engineReady: () => Promise<void>, signal: AbortSignal) => {
      state.setStatus('Downloading model…');
      const modelBytes = await engine.fetchModel(target, {
        signal,
        onProgress ({ received, total }) {
          /* Both figures go through the locale formatter: the size keeps its
             own unit label and the percent its own mark, instead of a "MB"
             and a "%" pasted onto numbers that already carry a grouped
             thousands separator. */
          if (total > 0) {
            state.setStatus(`Downloading model… ${fmtMegabytes(received)} / ${fmtMegabytes(total)} (${fmtPercent(received / total, 0)})`);
          } else {
            state.setStatus(`Downloading model… ${fmtMegabytes(received)}`);
          }
        },
      });
      if (!isGgufBuffer(modelBytes)) { throw new Error('This file is not a valid GGUF model.'); }
      await engineReady();
      await engine.loadModel(modelBytes);
      setModelName(nameFromUrl(target));
    }, true);
  }, [engine, failUrl, initAndLoad, state, url]);

  const loadFromBuffer = useCallback((file: File) => {
    if (file.name && !isGgufName(file.name)) {
      onReport('This is not a GGUF model file. Choose a file ending in .gguf.', 'error');
      return;
    }
    void initAndLoad(async (engineReady: () => Promise<void>, signal: AbortSignal) => {
      if (signal.aborted) { throw new DOMException('Aborted', 'AbortError'); }
      state.setStatus(`Reading ${file.name} (${fmtMegabytes(file.size)})…`);
      const modelBytes = await file.arrayBuffer();
      if (!isGgufBuffer(modelBytes)) { throw new Error('This file is not a valid GGUF model.'); }
      await engineReady();
      await engine.loadModel(modelBytes);
      setModelName(file.name);
    }, false);
  }, [engine, initAndLoad, onReport, state]);

  return { url, setUrl, modelName, loadFromUrl, loadFromBuffer };
};

export const useModelLoader = (engine: AgaveEngine, onReport: Report): ModelLoader => {
  const [status, setStatus] = useState(IDLE_HINT);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  /** Focus the composer. The prompt element is owned by the shell's composer,
   *  not by the loader: looking it up by id keeps it working straight after a
   *  load lands, when focus would otherwise drop to the body. */
  const focusPrompt = useCallback((): void => { document.querySelector<HTMLInputElement>('#prompt')?.focus(); }, []);
  /* The reader is looking at the URL field, and the error renders beside it,
     so focus stays where it is: the prompt is disabled until a model loads,
     and moving focus there drops it on the body instead. */
  const failUrl = useCallback((message: string) => {
    setStatus(message);
    setUrlError(message);
  }, []);

  const state: LoadState = { setStatus, setReady, setLoading, setUrlError };
  const abortRef = useRef<AbortController | null>(null);
  const sources = useModelSources(engine, state, onReport, failUrl, focusPrompt, abortRef);
  const cancelLoad = useCallback(() => { abortRef.current?.abort(); }, []);
  const setUrl = useCallback((next: string) => { setUrlError(null); sources.setUrl(next); }, [sources]);

  const onDragOver = useCallback((event: DragEvent) => { event.preventDefault(); setDragOver(true); }, []);
  const onDragLeave = useCallback((event: DragEvent) => {
    // Ignore a leave that stays inside the zone (child to child flicker).
    if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {return;}
    setDragOver(false);
  }, []);
  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    if (event.dataTransfer.files.length > 0) { sources.loadFromBuffer(event.dataTransfer.files[0]); }
  }, [sources]);

  return {
    status,
    ready,
    loading,
    url: sources.url,
    urlError,
    modelName: sources.modelName,
    busy: loading,
    cancelLoad,
    dragOver,
    onDragOver,
    onDragLeave,
    onDrop,
    setUrl,
    loadFromUrl: sources.loadFromUrl,
    loadFromBuffer: sources.loadFromBuffer,
    onReport,
    focusPrompt,
  };
};
