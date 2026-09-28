/* global AgaveEngine */

/**
 * Standalone WASM chat shell. Preact tree mounted by `web/index.html` after
 * `agave.js` has put `AgaveEngine` on `globalThis`.
 * Distinct from `src/web/` (HTTP --serve chat UI).
 *
 * Shares the agave theme (src/web/ui/theme.css) and the shadcn primitives with
 * the server UI, so both chat surfaces read as one product.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from '../src/web/ui/button';
import { EmptyState } from '../src/web/ui/empty-state';
import { HintChip } from '../src/web/ui/hint-chip';
import { Input } from '../src/web/ui/input';
import { cn } from '../src/web/ui/cn';
import { friendlyGenerateError } from './load-errors';
import { useModelLoader, type ModelLoader } from './use-model-loader';

const MAX_TOKENS = 200;
const ANNOUNCE_DELAY_MS = 100;

type Role = 'user' | 'assistant' | 'system' | 'error';

type Message = {
  id: number;
  role: Role;
  text: string;
};

const ROLE_LABELS = { user: 'You', assistant: 'Agave', system: 'System', error: 'Error' } as const;

const BUBBLE = {
  user: 'rounded-ee-[2px] border-border bg-card',
  assistant: 'rounded-es-[2px] border-border bg-popover',
  system: 'border-border-strong bg-primary/10 text-sm text-muted-foreground',
  error: 'border-destructive bg-destructive/10 text-sm text-destructive-foreground',
} as const;

/** Replace the live-region text, emptying it first so the same sentence
 *  announces twice. Returns the cancel handle for the pending timer. */
const announceLater = (setter: (text: string) => void, text: string): void => {
  setter('');
  setTimeout(function () { setter(text); }, ANNOUNCE_DELAY_MS);
};

const ShellMessage = ({ message }: { message: Message }) => (
  <div
    role={message.role === 'error' ? 'alert' : 'group'}
    aria-labelledby={message.role === 'error' ? undefined : `msg-role-${String(message.id)}`}
    className={cn(
      'mx-auto my-2 flex w-full agave-measure flex-col gap-1',
      message.role === 'user' ? 'ms-auto items-end' : 'me-auto items-start',
    )}
  >
    <span id={`msg-role-${String(message.id)}`} className="sr-only">{ROLE_LABELS[message.role]}</span>
    <div
      dir="auto"
      className={cn(
        'w-full rounded-lg border px-4 py-3 text-base whitespace-pre-wrap [overflow-wrap:break-word] [word-break:break-word] [line-break:loose]',
        BUBBLE[message.role],
      )}
    >
      {message.text}
    </div>
  </div>
);

/** The file drop zone. A label wraps the file input, so the same control takes a
 *  click and a drop, and the highlight follows the drag. */
const ModelDropZone = ({ loader }: { loader: ModelLoader }) => (
  <>
    <label
      htmlFor="file-input"
      onDragOver={loader.onDragOver}
      onDragLeave={loader.onDragLeave}
      onDrop={loader.onDrop}
      className={cn(
        'inline-flex min-h-11 cursor-pointer items-center rounded-lg border-2 border-dashed px-4 py-2 font-mono text-sm transition-colors hover:border-primary hover:text-primary aria-disabled:pointer-events-none aria-disabled:opacity-50',
        loader.dragOver ? 'border-primary text-primary' : 'border-border text-faint',
      )}
    >
      Drop GGUF file or click
    </label>
    <input
      id="file-input"
      type="file"
      accept=".gguf"
      className="sr-only"
      disabled={loader.busy}
      onChange={function (event) {
        const file = event.target.files?.[0];
        // Clear the field so choosing the same file again is still a change
        // Event: a rejected file is often re-picked after the reader fixes it.
        event.target.value = '';
        if (file !== undefined) { loader.loadFromBuffer(file); }
      }}
    />
  </>
);

/** The model bar. A loaded model needs only its name here: the URL field, the
 *  load button and the drop zone stay three rows tall on a phone, so they fold
 *  away behind "Change model" once there is a model to prompt. */
const ModelBar = ({ loader }: { loader: ModelLoader }) => {
  const [editing, setEditing] = useState(false);
  const name = loader.modelName ?? 'Model loaded';
  if (loader.ready && !editing) {
    return (
      <div
        role="region"
        aria-label="Loaded model"
        className="flex items-center gap-2 bg-card px-8 py-2 max-drawer:px-4"
      >
        <span className="min-w-0 flex-1 truncate font-mono text-sm text-faint" title={name}>{name}</span>
        <Button type="button" size="sm" onClick={function () { setEditing(true); }}>Change model</Button>
      </div>
    );
  }
  return (
  <div
    role="region"
    aria-label="Model loading"
    className="flex flex-wrap items-center gap-2 bg-card px-8 py-4 max-drawer:flex-col max-drawer:items-stretch max-drawer:px-4"
  >
    <label htmlFor="model-url" className="flex-none font-mono text-sm text-faint">Model URL</label>
    <Input
      id="model-url"
      type="url"
      placeholder="https://example.com/model.gguf"
      autoComplete="url"
      spellCheck={false}
      className="min-w-[200px] flex-1 font-mono"
      value={loader.url}
      disabled={loader.busy}
      aria-invalid={loader.urlError !== null}
      aria-describedby={loader.urlError === null ? undefined : 'url-error'}
      onChange={function (event) { loader.setUrl(event.target.value); }}
      onKeyDown={function (event) {
        if (event.key === 'Enter') { event.preventDefault(); loader.loadFromUrl(); }
      }}
    />
    <Button
      type="button"
      variant="solid"
      size="lg"
      onClick={loader.loadFromUrl}
      disabled={loader.busy}
      aria-busy={loader.loading}
    >
      {loader.loading ? 'Loading…' : 'Load model'}
    </Button>
    <ModelDropZone loader={loader} />
    {loader.ready ? (
      <Button type="button" size="lg" onClick={function () { setEditing(false); }}>Done</Button>
    ) : null}
    {loader.urlError === null ? null : (
      <div id="url-error" role="alert" className="w-full text-sm text-destructive-foreground">{loader.urlError}</div>
    )}
  </div>
  );
};

/** Before the first prompt. Same anchor as the serve UI's empty state, drawn by
 *  the same component, so the two surfaces read as one product rather than a
 *  chat page and a demo page. */
const ShellEmptyState = ({ ready }: { ready: boolean }) => (
  <EmptyState
    title={ready ? 'Prompt the model' : 'Load a model to start'}
    line={ready
      ? 'Inference runs in this tab. Nothing leaves the browser.'
      : 'Drop a GGUF file or paste a model URL above, then prompt it.'}
    hints={
      <>
        <HintChip>GGUF only</HintChip>
        <HintChip>Enter to send</HintChip>
        <HintChip>{`Replies capped at ${String(MAX_TOKENS)} tokens`}</HintChip>
      </>
    }
  />
);

/** The log: the transcript, the empty hint, and the pending bubble. */
const ChatLog = ({ messages, sending, ready }: { messages: Array<Message>; sending: boolean; ready: boolean }) => {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(function () {
    const log = logRef.current;
    if (log) { log.scrollTop = log.scrollHeight; }
  }, [messages, sending]);
  return (
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
      {messages.length === 0 ? <ShellEmptyState ready={ready} /> : null}
      {messages.map(function (message) { return <ShellMessage key={message.id} message={message} />; })}
      {sending ? (
        <div className="mx-auto flex w-full agave-measure flex-col items-start">
          <div
            role="status"
            aria-label="Generating response"
            className="w-full animate-pulse-soft rounded-lg rounded-es-[2px] border border-border bg-popover px-4 py-3 text-base text-faint"
          >
            …
          </div>
        </div>
      ) : null}
    </div>
  );
};

/** The product line, and the control that clears the transcript. */
const ShellHeader = ({ ready, busy, onClear }: { ready: boolean; busy: boolean; onClear: () => void }) => (
  <header className="flex items-center justify-between gap-4 border-b border-border bg-card px-8 py-4 max-drawer:flex-wrap max-drawer:px-4 max-drawer:py-3">
    <div>
      <h1 className="inline-flex items-center gap-2 font-mono text-lg font-semibold tracking-tight text-primary">
        <span className="mark" aria-hidden="true" />
        agave
      </h1>
      <small className="text-faint">LLM inference in the browser via WebAssembly</small>
    </div>
        {ready ? (
          <Button type="button" size="sm" onClick={onClear} title="Clear conversation" disabled={busy}>
            Clear
          </Button>
        ) : null}
    </header>
);

type ComposerProps = {
  prompt: string;
  onPrompt: (next: string) => void;
  onSend: () => void;
  ready: boolean;
  sending: boolean;
  busy: boolean;
  focus: () => void;
};

const Composer = ({ prompt, onPrompt, onSend, ready, sending, busy, focus }: ComposerProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(function () { focus(); }, [focus]);
  return (
    <>
      <form
        aria-label="Send message"
        onSubmit={function (event) { event.preventDefault(); onSend(); }}
        className="flex gap-2 border-t border-border bg-card px-8 py-4 max-drawer:px-4"
      >
        <Input
          ref={inputRef}
          id="prompt"
          placeholder={ready ? 'Type a message...' : 'Load a model to start...'}
          aria-label="Message input"
          enterKeyHint="send"
          autoComplete="off"
          dir="auto"
          className="flex-1 text-base max-drawer:text-[16px]"
          value={prompt}
          disabled={!ready || busy}
          aria-describedby={ready ? 'input-hint' : undefined}
          onChange={function (event) { onPrompt(event.target.value); }}
          onKeyDown={function (event) {
            // Ignore Enter during IME composition (CJK input).
            // There, Enter confirms the conversion; it must not send.
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <Button
          type="submit"
          variant="solid"
          size="lg"
          aria-label="Send message"
          aria-busy={sending}
          disabled={!ready || busy || !prompt.trim()}
        >
          {sending ? 'Generating…' : 'Send'}
        </Button>
      </form>
      <p id="input-hint" hidden={!ready} className="px-8 text-center font-mono text-xs text-faint max-drawer:px-4">
        Enter to send
      </p>
    </>
  );
};

/** The transcript, the composer, and the send and clear actions. */
const useShellChat = (engine: AgaveEngine) => {
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [prompt, setPrompt] = useState('');
  const [sending, setSending] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const nextId = useRef(1);
  const announce = useCallback((text: string) => { announceLater(setAnnouncement, text); }, []);

  const addMessage = useCallback(function (role: Role, text: string) {
    const id = nextId.current;
    nextId.current += 1;
    setMessages(function (previous) { return [...previous, { id, role, text }]; });
    if (role === 'error' || role === 'system') { announce(text); }
    else if (role === 'assistant') { announce(`Agave responded: ${text.slice(0, 200)}`); }
  }, [announce]);

  const send = useCallback(function (ready: boolean, focus: () => void) {
    const text = prompt.trim();
    if (!text || sending || !ready) {return;}
    setSending(true);
    addMessage('user', text);
    setPrompt('');
    announce('Generating response…');
    void (async function () {
      try {
        addMessage('assistant', await engine.generate(text, { maxTokens: MAX_TOKENS }));
      } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- rendered as an alert in the log
        addMessage('error', friendlyGenerateError(error));
      } finally {
        setSending(false);
        announce('Response complete');
        focus();
      }
    })();
  }, [addMessage, announce, engine, prompt, sending]);

  const clearChat = useCallback(function (focus: () => void) {
    if (sending) {return;}
    if (messages.length === 0) { return; }
    if (!globalThis.confirm('Clear this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    setMessages([]);
    announce('Conversation cleared');
    focus();
  }, [announce, messages.length, sending]);

  return { messages, prompt, setPrompt, sending, announcement, addMessage, send, clearChat };
};

const Shell = () => {
  const engine = new AgaveEngine();
  const chat = useShellChat(engine);
  const loader = useModelLoader(engine, function (text, level) {
    chat.addMessage(level === 'error' ? 'error' : 'system', text);
  });

  const busy = loader.busy || chat.sending;
  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <a
        href="#prompt"
        className="absolute start-4 top-[-100%] z-50 rounded-lg bg-primary px-4 py-2 font-mono text-sm font-medium text-primary-foreground no-underline transition-[top] duration-200 focus:top-2"
      >
        Skip to message input
      </a>
      <ShellHeader ready={loader.ready} busy={busy} onClear={function () { chat.clearChat(loader.focusPrompt); }} />
      <ModelBar loader={loader} />
      <main aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
        <p
          id="status"
          role="status"
          aria-live="polite"
          className="bg-background px-8 py-2 font-mono text-sm text-faint max-drawer:px-4"
        >
          {loader.status}
        </p>
        <ChatLog messages={chat.messages} sending={chat.sending} ready={loader.ready} />
        <Composer
          prompt={chat.prompt}
          onPrompt={chat.setPrompt}
          onSend={function () { chat.send(loader.ready, loader.focusPrompt); }}
          ready={loader.ready}
          sending={chat.sending}
          busy={busy}
          focus={loader.focusPrompt}
        />
      </main>
      <div id="sr-announce" className="sr-only" aria-live="polite" aria-atomic="true">{chat.announcement}</div>
    </div>
  );
};

const root = document.querySelector('#root');
if (root === null) { throw new Error('missing #root'); }
createRoot(root).render(<Shell />);
