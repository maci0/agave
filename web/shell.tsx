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
import { Mark } from '../src/web/ui/icons';
import { SkipLink } from '../src/web/ui/skip-link';
import { cn } from '../src/web/ui/cn';
import { friendlyGenerateError } from './load-errors';
import { useModelLoader, type ModelLoader } from './use-model-loader';
import { writingDirection } from '../src/web/chat/format';

const MAX_TOKENS = 200;
const ANNOUNCE_DELAY_MS = 100;

/* There is no `assistant` role: the wasm32 forward pass is blocked (see the
   note in src/wasm_entry.zig), so `generate()` resolves with the engine's
   tokenization report, not model output. Everything the engine says is
   labelled `engine`, which keeps a report from reading as a reply and makes a
   regression to an `assistant` bubble a type error. */
type Role = 'user' | 'engine' | 'error';

type Message = {
  id: number;
  role: Role;
  text: string;
};

const ROLE_LABELS = { user: 'You', engine: 'Engine', error: 'Error' } as const;

const BUBBLE = {
  user: 'w-auto max-w-4/5 rounded-ee-xs border-transparent bg-primary/10',
  engine: 'border-border bg-card text-sm text-muted-foreground',
  error: 'border-destructive bg-destructive/10 text-sm text-destructive-foreground',
} as const;

/** Replace the live-region text, emptying it first so the same sentence
 *  announces twice. Returns the cancel handle for the pending timer. */
const announceLater = (setter: (text: string) => void, text: string): void => {
  setter('');
  setTimeout(() => { setter(text); }, ANNOUNCE_DELAY_MS);
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
        'agave-wrap w-full rounded-lg border px-4 py-3 text-base whitespace-pre-wrap',
        BUBBLE[message.role],
      )}
    >
      {message.text}
    </div>
  </div>
);

/** The file drop zone. A label wraps the file input, so the same control takes a
 *  click and a drop, and the highlight follows the drag.
 *
 *  The input is the label's target and carries its text as its name, and it is
 *  moved off the visual layout with a clip rather than `display: none`: a
 *  display-hidden input is not focusable, so the one keyboard route to the
 *  drop zone (tab to it, press Enter) disappeared, and a clipped one still
 *  takes focus and opens the picker. The label draws the focus ring from
 *  `:focus-within`, which the clip preserves. */
const ModelDropZone = ({ loader }: { loader: ModelLoader }) => (
  <>
    <label
      htmlFor="file-input"
      onDragOver={loader.onDragOver}
      onDragLeave={loader.onDragLeave}
      onDrop={loader.onDrop}
      className={cn(
        'inline-flex min-h-11 cursor-pointer items-center rounded-lg border-2 border-dashed px-4 py-2 font-mono text-sm transition-colors hover:border-primary hover:text-primary focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-primary aria-disabled:pointer-events-none aria-disabled:opacity-50',
        loader.dragOver ? 'border-primary text-primary' : 'border-border text-faint',
      )}
    >
      Choose or drop a GGUF file
    </label>
    <input
      id="file-input"
      type="file"
      accept=".gguf"
      /* The visually-hidden pattern, not Tailwind's `sr-only`: that utility is
         a fixed 1px box rather than a clip, so the control stayed in the tab
         order with no visible indication of where focus was. */
      className="absolute size-px overflow-hidden border-0 p-0 opacity-0"
      disabled={loader.busy}
      onChange={function (event) {
        const file = event.target.files?.[0];
        /* Clear the field so choosing the same file again is still a change
           event: a rejected file is often re-picked after the reader fixes it. */
        event.target.value = '';
        if (file !== undefined) { loader.loadFromBuffer(file); }
      }}
    />
  </>
);

/** A failed model load, announced where it lands rather than only where the
 *  field is.
 *
 *  The region is mounted with the page and holds the live semantics, because a
 *  `role="alert"` element that appears already holding its text is the case
 *  assistive tech stays silent about — and a failed load leaves focus on the
 *  Load key, not in the field, so `aria-describedby` on the field alone would
 *  never be read. The empty state carries no text, so the region announces
 *  nothing until it has something to say. */
const UrlError = ({ message }: { message: string | null }) => (
  <div id="url-error" role="alert" aria-live="assertive" className="empty:hidden w-full text-sm text-destructive-foreground">
    {message ?? ''}
  </div>
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
        className="flex items-center gap-2 border-b border-divider bg-card px-8 py-2 max-drawer:px-4"
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
    className="flex flex-wrap items-center gap-2 border-b border-divider bg-card px-8 py-4 max-drawer:flex-col max-drawer:items-stretch max-drawer:px-4"
  >
    <label htmlFor="model-url" className="flex-none font-mono text-sm text-faint">Model URL</label>
    <Input
      id="model-url"
      type="url"
      placeholder="https://example.com/model.gguf"
      autoComplete="url"
      spellCheck={false}
      font="mono"
      className="min-w-50 flex-1"
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
    <UrlError message={loader.urlError} />
  </div>
  );
};

/** Before the first prompt. Same anchor as the serve UI's empty state, drawn by
 *  the same component, so the two surfaces read as one product rather than a
 *  chat page and a demo page.
 *
 *  The states name what the browser build does, which is init, parse and
 *  tokenize: docs/brand/README.md ("Honest status") forbids presenting the
 *  WASM shell as a working chat, and the reader has to learn the limit before
 *  the first prompt, not from the report it returns. */
const ShellEmptyState = ({ ready }: { ready: boolean }) => (
  <EmptyState
    title={ready ? 'Prompt the model' : 'Load a model to start'}
    line={ready
      ? 'The engine tokenizes the prompt and reports the token count. It does not generate text in this build.'
      : 'Drop a GGUF file or paste a model URL above, then prompt it.'}
    hints={
      <>
        <HintChip>GGUF only</HintChip>
        <HintChip>Enter to send</HintChip>
        <HintChip>Forward pass pending a Zig wasm32 fix</HintChip>
      </>
    }
  />
);

/** The log: the transcript, the empty hint, and the pending bubble. */
const ChatLog = ({ messages, sending, ready }: { messages: Array<Message>; sending: boolean; ready: boolean }) => {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
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
      {messages.map((message) => <ShellMessage key={message.id} message={message} />)}
      {sending ? (
        <div className="mx-auto flex w-full agave-measure flex-col items-start">
          <div className="w-full animate-pulse-soft px-1 py-3 text-base text-faint">
            <span className="sr-only">Tokenizing the prompt</span>
            <span aria-hidden="true">…</span>
          </div>
        </div>
      ) : null}
    </div>
  );
};

/** The product line, and the control that clears the transcript. */
const ShellHeader = ({ ready, busy, onClear }: { ready: boolean; busy: boolean; onClear: () => void }) => (
  <header className="flex items-center justify-between gap-4 border-b border-divider bg-card px-8 py-4 max-drawer:flex-wrap max-drawer:px-4 max-drawer:py-3">
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <h1 className="inline-flex items-center gap-2 font-mono text-lg font-semibold tracking-tight text-primary">
        <Mark />
        agave
      </h1>
      <small className="text-sm text-faint">Init, parse and tokenize only. Text generation is blocked in this build.</small>
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
  useEffect(() => { focus(); }, [focus]);
  return (
    <>
      <form
        aria-label="Send message"
        onSubmit={function (event) { event.preventDefault(); onSend(); }}
        className="border-t border-divider bg-card px-8 py-4 max-drawer:px-4"
      >
        <div className="mx-auto flex w-full agave-measure gap-2">
          <Input
            ref={inputRef}
            id="prompt"
            placeholder={ready ? 'Prompt' : 'Load a model first'}
            aria-label="Message input"
            enterKeyHint="send"
            autoComplete="off"
            dir="auto"
            size="lg"
            className="flex-1"
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
            {sending ? 'Tokenizing…' : 'Send'}
          </Button>
        </div>
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

  const addMessage = useCallback((role: Role, text: string) => {
    const id = nextId.current;
    nextId.current += 1;
    setMessages((previous) => [...previous, { id, role, text }]);
    if (role === 'error' || role === 'engine') { announce(text); }
  }, [announce]);

  const send = useCallback((ready: boolean, focus: () => void) => {
    const text = prompt.trim();
    if (!text || sending || !ready) {return;}
    setSending(true);
    addMessage('user', text);
    setPrompt('');
    announce('Tokenizing the prompt…');
    void (async function () {
      try {
        addMessage('engine', await engine.generate(text, { maxTokens: MAX_TOKENS }));
      } catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- rendered as an alert in the log
        addMessage('error', friendlyGenerateError(error));
      } finally {
        setSending(false);
        announce('Tokenization report ready');
        focus();
      }
    })();
  }, [addMessage, announce, engine, prompt, sending]);

  const clearChat = useCallback((focus: () => void) => {
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
  /* One engine per mount, not per render: `new AgaveEngine()` in the component
     body threw away the instantiated module and the loaded model on every state
     change, so the composer came back "Load a GGUF model first." after a load.
     A lazy ref keeps the instance stable and still constructs it during render,
     which is when the first instance was built. */
  const engineRef = useRef<AgaveEngine | null>(null);
  engineRef.current ??= new AgaveEngine();
  const engine = engineRef.current;
  const chat = useShellChat(engine);
  const loader = useModelLoader(engine, (text, level) => {
    chat.addMessage(level === 'error' ? 'error' : 'engine', text);
  });

  const busy = loader.busy || chat.sending;
  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <SkipLink href="#prompt">Skip to message input</SkipLink>
      <ShellHeader ready={loader.ready} busy={busy} onClear={function () { chat.clearChat(loader.focusPrompt); }} />
      <ModelBar loader={loader} />
      <main aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
        {/* A live region rather than `role="status"`: this element is mounted
            with the page and its first text ("Load a GGUF model to begin")
            lands in the same commit, which is exactly the case a live region
            introduced together with its content is not announced for. */}
        <p
          id="status"
          aria-live="polite"
          aria-atomic="true"
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
/* The shell stylesheet uses logical properties throughout, so this attribute
   is the whole of its right-to-left support. Set before the first render so
   the panel starts on the reader's own side instead of flipping after paint. */
document.documentElement.dir = writingDirection(navigator.language);
createRoot(root).render(<Shell />);
