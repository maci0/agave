/**
 * Standalone WASM chat shell. React tree mounted by `web/index.html` after
 * `agave.js` has put `AgaveEngine` and `AgaveError` on `globalThis`.
 * Distinct from `src/web/` (HTTP --serve chat UI).
 *
 * Shares the agave theme (src/web/ui/theme.css) and the shadcn primitives with
 * the server UI, so both chat surfaces read as one product.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from '../src/web/ui/button';
import { Input } from '../src/web/ui/input';
import { cn } from '../src/web/ui/utils';

const engine = new AgaveEngine();

/** GGUF files start with this 4-byte magic (`GGUF`). */
const GGUF_MAGIC = [0x47, 0x47, 0x55, 0x46] as const;
const ANNOUNCE_DELAY_MS = 100;
const MAX_TOKENS = 200;
const IDLE_HINT = 'Load a GGUF model to begin';
const READY_HINT = 'Ready';

type Role = 'user' | 'assistant' | 'system' | 'error';

type Message = {
  id: number;
  role: Role;
  text: string;
};

const ROLE_LABELS: Record<Role, string> = { user: 'You', assistant: 'Agave', system: 'System', error: 'Error' };

function fmtMb(bytes: number): string {
  return (bytes / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });
}

function truncateAnnounce(text: string, maxChars: number): string {
  const chars = Array.from(text);
  if (chars.length <= maxChars) {return text;}
  return `${chars.slice(0, maxChars).join('')}...`;
}

function isGgufName(name: string): boolean {
  return name.toLowerCase().endsWith('.gguf');
}

function isGgufBuffer(data: ArrayBuffer): boolean {
  if (data.byteLength < GGUF_MAGIC.length) {return false;}
  const head = new Uint8Array(data, 0, GGUF_MAGIC.length);
  return head[0] === GGUF_MAGIC[0] && head[1] === GGUF_MAGIC[1] && head[2] === GGUF_MAGIC[2] && head[3] === GGUF_MAGIC[3];
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

/** Map engine and network failures to short, actionable copy. */
function friendlyLoadError(error: unknown): string {
  if (error instanceof AgaveError) {
    switch (error.code) {
      case 'wasm_fetch_failed':
        return 'Could not load the inference engine. Reload the page, or check that agave.wasm is being served.';
      case 'wasm_invalid':
        return 'The inference engine failed to start. Reload the page.';
      case 'download_failed':
        if (error.httpStatus === 404) {return 'The model URL was not found. Check the link.';}
        if (error.httpStatus === 403 || error.httpStatus === 401) {
          return 'The model URL refused the download. Try dropping a GGUF file instead.';
        }
        return 'Could not download the model. Check the URL, or drop a GGUF file instead. Some hosts block browser downloads.';
      case 'gguf_parse':
        return 'This file is not a valid GGUF model.';
      case 'unsupported_arch':
        return 'This model architecture is not supported in the browser.';
      case 'no_vocab':
        return 'This GGUF file has no vocabulary and cannot be used.';
      case 'tokenizer':
        return 'Could not read the tokenizer from this model file.';
      case 'init_failed':
        return 'Could not initialize this model in the browser.';
      case 'alloc_failed':
        return 'The model is too large to fit in this browser.';
      case 'not_initialized':
        return 'The engine is not ready. Reload the page and try again.';
      default:
        return error.message.startsWith('Could not') ? error.message : `Could not load model: ${error.message}`;
    }
  }
  const message = error instanceof Error ? error.message : String(error);
  const lower = message.toLowerCase();
  if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
    return 'Could not download the model. Check the URL, or drop a GGUF file instead. Some hosts block browser downloads.';
  }
  if (lower.includes('http 404')) {return 'The model URL was not found. Check the link.';}
  if (lower.includes('http 403') || lower.includes('http 401')) {
    return 'The model URL refused the download. Try dropping a GGUF file instead.';
  }
  if (lower.startsWith('gguf parse error') || lower.includes('not a valid gguf')) {
    return 'This file is not a valid GGUF model.';
  }
  if (lower.startsWith('unsupported arch')) {
    return 'This model architecture is not supported in the browser.';
  }
  if (lower.startsWith('no vocab')) {return 'This GGUF file has no vocabulary and cannot be used.';}
  if (lower.startsWith('tok error')) {return 'Could not read the tokenizer from this model file.';}
  if (lower.startsWith('model init error') || lower === 'failed to initialize model') {
    return 'Could not initialize this model in the browser.';
  }
  if (lower.includes('failed to allocate')) {return 'The model is too large to fit in this browser.';}
  if (lower === 'engine not initialized') {return 'The engine is not ready. Reload the page and try again.';}
  return message.startsWith('Could not') ? message : `Could not load model: ${message}`;
}

/** Map generation failures to short, actionable copy, mirroring friendlyLoadError. */
function friendlyGenerateError(error: unknown): string {
  if (error instanceof AgaveError) {
    switch (error.code) {
      case 'not_initialized':
      case 'no_model':
        return 'Load a GGUF model first.';
      case 'alloc_failed':
        return 'Not enough memory to generate a reply. Try a smaller model.';
      case 'invalid_argument':
        return 'Generation settings are out of range. Reload the page and try again.';
      case 'tokenize':
        return 'That message could not be encoded for this model. Try shorter or plain text.';
      case 'wasm_invalid':
        return 'The inference engine failed. Reload the page.';
      default:
        return error.message.startsWith('Could not') ? error.message : `Could not generate a reply: ${error.message}`;
    }
  }
  const message = error instanceof Error ? error.message : String(error);
  const lower = message.toLowerCase();
  if (lower === 'no model loaded' || lower === 'model not initialized') {return 'Load a GGUF model first.';}
  if (lower.includes('out of memory') || lower.includes('failed to allocate')) {
    return 'Not enough memory to generate a reply. Try a smaller model.';
  }
  if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
    return 'The connection dropped while generating. Try again.';
  }
  return message.startsWith('Could not') ? message : `Could not generate a reply: ${message}`;
}

function Shell() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState(IDLE_HINT);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [url, setUrl] = useState('');
  const [urlError, setUrlError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [announcement, setAnnouncement] = useState('');

  const nextId = useRef(1);
  const announceTimer = useRef<number | null>(null);
  const promptRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const hadModel = useRef(false);

  const announce = useCallback(function (text: string) {
    if (announceTimer.current !== null) { window.clearTimeout(announceTimer.current); }
    setAnnouncement('');
    announceTimer.current = window.setTimeout(function () { setAnnouncement(text); }, ANNOUNCE_DELAY_MS);
  }, []);

  const addMessage = useCallback(function (role: Role, text: string) {
    const id = nextId.current;
    nextId.current += 1;
    setMessages(function (previous) { return [...previous, { id, role, text }]; });
    if (role === 'error' || role === 'system') { announce(text); }
    else if (role === 'assistant') { announce(`Agave responded: ${truncateAnnounce(text, 200)}`); }
  }, [announce]);

  useEffect(function () {
    const log = logRef.current;
    if (log) { log.scrollTop = log.scrollHeight; }
  }, [messages, sending]);

  const failUrl = useCallback(function (message: string) {
    setStatus(message);
    setUrlError(message);
    announce(message);
    promptRef.current?.focus();
  }, [announce]);

  const downloadModel = useCallback(function (modelUrl: string): Promise<ArrayBuffer> {
    return engine.fetchModel(modelUrl, {
      onProgress: function ({ received, total }) {
        if (total > 0) {
          const percent = Math.round((received / total) * 100);
          setStatus(`Downloading model… ${fmtMb(received)} / ${fmtMb(total)} MB (${String(percent)}%)`);
        } else {
          setStatus(`Downloading model… ${fmtMb(received)} MB`);
        }
      },
    });
  }, []);

  const initAndLoad = useCallback(function (load: () => Promise<void>, fromUrl: boolean) {
    hadModel.current = engine.hasModel;
    setLoading(true);
    setReady(false);
    setStatus('Initializing engine…');
    return (async function () {
      try {
        if (!engine.ready) { await engine.init(); }
        await load();
        addMessage('system', engine.initMessage || 'Model loaded');
        setStatus(READY_HINT);
        announce('Model loaded. Ready to chat.');
        setReady(true);
        promptRef.current?.focus();
      } catch (error) {
        const message = friendlyLoadError(error);
        setStatus(message);
        addMessage('error', message);
        if (fromUrl) { setUrlError(message); }
        // A failed reload must not disable chat if the previous model is still loaded.
        if (hadModel.current && engine.hasModel) { setReady(true); }
      } finally {
        setLoading(false);
      }
    })();
  }, [addMessage, announce]);

  const loadFromUrl = useCallback(function () {
    const target = url.trim();
    if (!target) { failUrl('Enter a model URL first'); return; }
    if (!isHttpUrl(target)) { failUrl('Enter a valid http(s) URL to a GGUF file'); return; }
    setUrlError(null);
    void initAndLoad(async function () {
      setStatus('Downloading model…');
      const data = await downloadModel(target);
      if (!isGgufBuffer(data)) { throw new Error('This file is not a valid GGUF model.'); }
      await engine.loadModel(data);
    }, true);
  }, [downloadModel, failUrl, initAndLoad, url]);

  const loadFromBuffer = useCallback(function (file: File) {
    if (file.name && !isGgufName(file.name)) {
      const message = 'This is not a GGUF model file. Choose a file ending in .gguf.';
      setStatus(message);
      addMessage('error', message);
      return;
    }
    void initAndLoad(async function () {
      setStatus(`Reading ${file.name} (${fmtMb(file.size)} MB)…`);
      const data = await file.arrayBuffer();
      if (!isGgufBuffer(data)) { throw new Error('This file is not a valid GGUF model.'); }
      await engine.loadModel(data);
    }, false);
  }, [addMessage, initAndLoad]);

  const send = useCallback(function () {
    const text = prompt.trim();
    if (!text || sending || !ready) {return;}
    setSending(true);
    addMessage('user', text);
    setPrompt('');
    setStatus('Generating…');
    announce('Generating response…');
    void engine.generate(text, { maxTokens: MAX_TOKENS }).then(
      function (output) { addMessage('assistant', output); },
      function (error) { addMessage('error', friendlyGenerateError(error)); },
    ).then(function () {
      setSending(false);
      setStatus(engine.hasModel ? READY_HINT : IDLE_HINT);
      announce('Response complete');
      promptRef.current?.focus();
    });
  }, [addMessage, announce, prompt, ready, sending]);

  const clearChat = useCallback(function () {
    if (sending) {return;}
    if (messages.length === 0) { setStatus('Nothing to clear'); return; }
    if (!globalThis.confirm('Clear this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    setMessages([]);
    setStatus(engine.hasModel ? READY_HINT : IDLE_HINT);
    announce('Conversation cleared');
    promptRef.current?.focus();
  }, [announce, messages.length, sending]);

  const busy = loading || sending;

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <a
        href="#prompt"
        className="skip-link absolute start-4 top-[-100%] z-50 rounded-lg bg-primary px-4 py-2 font-mono text-sm font-medium text-primary-foreground no-underline transition-[top] duration-200 focus:top-2"
      >
        Skip to message input
      </a>
      <header className="flex items-center justify-between gap-4 border-b border-border bg-card px-8 py-4 max-drawer:flex-wrap max-drawer:px-4 max-drawer:py-3">
        <div>
          <h1 className="inline-flex items-center gap-2 font-mono text-lg font-semibold tracking-tight text-primary">
            <span className="mark" aria-hidden="true" />
            agave
          </h1>
          <small className="text-faint">LLM inference in the browser via WebAssembly</small>
        </div>
        {ready ? (
          <Button type="button" size="sm" onClick={clearChat} title="Clear conversation" disabled={busy}>
            Clear
          </Button>
        ) : null}
      </header>

      <div role="region" aria-label="Model loading" className="flex flex-wrap items-center gap-2 bg-card px-8 py-4 max-drawer:flex-col max-drawer:items-stretch max-drawer:px-4">
        <label htmlFor="model-url" className="flex-none font-mono text-sm text-faint">Model URL</label>
        <Input
          id="model-url"
          type="url"
          placeholder="https://example.com/model.gguf"
          autoComplete="url"
          spellCheck={false}
          className="min-w-[200px] flex-1 font-mono"
          value={url}
          disabled={busy}
          aria-invalid={urlError !== null}
          aria-describedby={urlError === null ? undefined : 'url-error'}
          onChange={function (event) { setUrl(event.target.value); setUrlError(null); }}
          onKeyDown={function (event) {
            if (event.key === 'Enter') { event.preventDefault(); loadFromUrl(); }
          }}
        />
        <Button type="button" variant="solid" size="lg" onClick={loadFromUrl} disabled={busy} aria-busy={loading}>
          {loading ? 'Loading…' : 'Load model'}
        </Button>
        <label
          id="drop-zone"
          htmlFor="file-input"
          onDragOver={function (event) { event.preventDefault(); setDragOver(true); }}
          onDragLeave={function (event) {
            if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {return;}
            setDragOver(false);
          }}
          onDrop={function (event) {
            event.preventDefault();
            setDragOver(false);
            const file = event.dataTransfer?.files[0];
            if (file) { loadFromBuffer(file); }
          }}
          className={cn(
            'inline-flex min-h-11 cursor-pointer items-center rounded-lg border-2 border-dashed border-border px-4 py-2 font-mono text-sm text-faint transition-colors hover:border-primary hover:text-primary',
            dragOver && 'border-primary text-primary',
            busy && 'pointer-events-none opacity-50',
          )}
          aria-disabled={busy}
        >
          Drop GGUF file or click
        </label>
        <input
          id="file-input"
          type="file"
          accept=".gguf"
          className="sr-only"
          disabled={busy}
          onChange={function (event) {
            const file = event.target.files?.[0];
            if (file) { loadFromBuffer(file); }
          }}
        />
        {urlError === null ? null : (
          <div id="url-error" role="alert" className="w-full text-sm text-destructive-foreground">{urlError}</div>
        )}
      </div>

      <main aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
        <p id="status" role="status" aria-live="polite" className="bg-background px-8 py-2 font-mono text-sm text-faint max-drawer:px-4">
          {status}
        </p>
        <div
          id="chat"
          ref={logRef}
          role="log"
          aria-label="Chat messages"
          aria-live="off"
          aria-busy={sending}
          tabIndex={0}
          className="agave-scroll flex-1 overflow-y-auto px-8 py-4 max-drawer:px-4"
        >
          {messages.length === 0 ? (
            <div className="mx-auto my-8 max-w-prose text-center text-faint">
              <span className="mark mark-lg" aria-hidden="true" />
              {engine.hasModel ? 'Send a prompt.' : 'Load a GGUF model above, then send a prompt.'}
            </div>
          ) : null}
          {messages.map(function (message) {
            return (
              <div
                key={message.id}
                role={message.role === 'error' ? 'alert' : 'group'}
                aria-labelledby={message.role === 'error' ? undefined : `msg-role-${String(message.id)}`}
                className={cn(
                  'mx-auto my-2 flex w-full max-w-[80%] flex-col gap-1',
                  message.role === 'user' ? 'ms-auto items-end' : 'me-auto items-start',
                )}
              >
                <span id={`msg-role-${String(message.id)}`} className="sr-only">{ROLE_LABELS[message.role]}</span>
                <div
                  dir="auto"
                  className={cn(
                    'w-full rounded-lg border px-4 py-3 text-base whitespace-pre-wrap [overflow-wrap:break-word] [word-break:break-word] [line-break:loose]',
                    message.role === 'user' && 'rounded-ee-[2px] border-border bg-card',
                    message.role === 'assistant' && 'rounded-es-[2px] border-border bg-popover',
                    message.role === 'system' && 'border-border-strong bg-primary/10 text-sm text-muted-foreground',
                    message.role === 'error' && 'border-destructive bg-destructive/10 text-sm text-destructive-foreground',
                  )}
                >
                  {message.text}
                </div>
              </div>
            );
          })}
          {sending ? (
            <div role="status" aria-label="Generating response" className="mx-auto my-2 max-w-[80%] animate-pulse-soft rounded-lg rounded-es-[2px] border border-border bg-popover px-4 py-3 text-base text-faint">
              …
            </div>
          ) : null}
        </div>
        <form
          aria-label="Send message"
          onSubmit={function (event) { event.preventDefault(); send(); }}
          className="flex gap-2 border-t border-border bg-card px-8 py-4 max-drawer:px-4"
        >
          <Input
            ref={promptRef}
            id="prompt"
            placeholder={ready ? 'Type a message...' : 'Load a model to start...'}
            aria-label="Message input"
            enterKeyHint="send"
            autoComplete="off"
            dir="auto"
            autoFocus
            className="flex-1 text-base max-drawer:text-[16px]"
            value={prompt}
            disabled={!ready || busy}
            aria-describedby={ready ? 'input-hint' : undefined}
            onChange={function (event) { setPrompt(event.target.value); }}
            onKeyDown={function (event) {
              // Ignore Enter during IME composition (CJK input): Enter there
              // confirms the conversion, it must not send.
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                send();
              }
            }}
          />
          <Button type="submit" variant="solid" size="lg" aria-label="Send message" aria-busy={sending} disabled={!ready || busy || !prompt.trim()}>
            {sending ? 'Generating…' : 'Send'}
          </Button>
        </form>
        <p id="input-hint" hidden={!ready} className="px-8 text-center font-mono text-xs text-faint max-drawer:px-4">
          Enter to send
        </p>
      </main>
      <div id="sr-announce" className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
    </div>
  );
}

const root = document.getElementById('root');
if (!root) { throw new Error('missing #root'); }
createRoot(root).render(<Shell />);
