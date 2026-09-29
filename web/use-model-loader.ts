/** Model loading for the browser WASM shell.
 *
 *  Owns everything between "the reader asked for a model" and "the engine can
 *  take a prompt": the URL field, the file drop zone, the download progress
 *  line, and the error copy. The conversation itself lives in `shell.tsx`.
 */

/* global AgaveEngine */

import { useCallback, useRef, useState, type DragEvent } from 'react';
import { friendlyLoadError } from './load-errors';

const IDLE_HINT = 'Load a GGUF model to begin';
const READY_HINT = 'Ready';
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

const fmtMb = (bytes: number): string =>
  (bytes / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });

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
const useEngineInit = (engine: AgaveEngine, state: LoadState, onReport: Report, focusPrompt: () => void) => {
  const hadModel = useRef(false);
  return useCallback(async (load: () => Promise<void>, fromUrl: boolean) => {
    hadModel.current = engine.hasModel;
    state.setLoading(true);
    state.setReady(false);
    state.setStatus('Initializing engine…');
    try {
      if (!engine.ready) { await engine.init(); }
      await load();
      onReport(engine.initMessage || 'Model loaded', 'info');
      state.setStatus(READY_HINT);
      state.setReady(true);
      focusPrompt();
    } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- reported to the reader, not swallowed
      state.setStatus(friendlyLoadError(error));
      onReport(friendlyLoadError(error), 'error');
      if (fromUrl) { state.setUrlError(friendlyLoadError(error)); }
      // A failed reload must not disable chat if the previous model is still loaded.
      if (hadModel.current && engine.hasModel) { state.setReady(true); }
    } finally {
      state.setLoading(false);
    }
  }, [engine, focusPrompt, onReport, state]);
};

/** The two ways a model arrives: a URL to fetch, or a file the reader dropped. */
const useModelSources = (
  engine: AgaveEngine,
  state: LoadState,
  onReport: Report,
  failUrl: (message: string) => void,
  focusPrompt: () => void,
) => {
  const initAndLoad = useEngineInit(engine, state, onReport, focusPrompt);
  const [url, setUrl] = useState('');
  // Set only after the engine accepts the bytes, so the bar never names a model
  // That failed to load: a stale name would read as a working model.
  const [modelName, setModelName] = useState<string | null>(null);

  const loadFromUrl = useCallback(() => {
    const target = url.trim();
    if (!target) { failUrl('Enter a model URL first'); return; }
    if (!isHttpUrl(target)) { failUrl('Enter a valid http(s) URL to a GGUF file'); return; }
    state.setUrlError(null);
    void initAndLoad(async () => {
      state.setStatus('Downloading model…');
      const modelBytes = await engine.fetchModel(target, {
        onProgress ({ received, total }) {
          if (total > 0) {
            const percent = Math.round((received / total) * 100);
            state.setStatus(`Downloading model… ${fmtMb(received)} / ${fmtMb(total)} MB (${String(percent)}%)`);
          } else {
            state.setStatus(`Downloading model… ${fmtMb(received)} MB`);
          }
        },
      });
      if (!isGgufBuffer(modelBytes)) { throw new Error('This file is not a valid GGUF model.'); }
      await engine.loadModel(modelBytes);
      setModelName(nameFromUrl(target));
    }, true);
  }, [engine, failUrl, initAndLoad, state, url]);

  const loadFromBuffer = useCallback((file: File) => {
    if (file.name && !isGgufName(file.name)) {
      onReport('This is not a GGUF model file. Choose a file ending in .gguf.', 'error');
      return;
    }
    void initAndLoad(async () => {
      state.setStatus(`Reading ${file.name} (${fmtMb(file.size)} MB)…`);
      const modelBytes = await file.arrayBuffer();
      if (!isGgufBuffer(modelBytes)) { throw new Error('This file is not a valid GGUF model.'); }
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
  const prompt = useRef<HTMLInputElement>(null);

  const focusPrompt = useCallback(() => { prompt.current?.focus(); }, []);
  // The reader is looking at the URL field, and the error renders beside it,
  // So focus stays where it is: the prompt is disabled until a model loads,
  // And moving focus there drops it on the body instead.
  const failUrl = useCallback((message: string) => {
    setStatus(message);
    setUrlError(message);
  }, []);

  const state: LoadState = { setStatus, setReady, setLoading, setUrlError };
  const sources = useModelSources(engine, state, onReport, failUrl, focusPrompt);
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
