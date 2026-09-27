// Server chat UI (embedded by src/server/server.zig). Not the WASM browser shell in web/.
// Source of truth: compile with `scripts/build-web.sh` (bun + Tailwind) to refresh
// the committed app.js and style.css.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AboutDialog } from './chat/components/about-dialog';
import { AppHeader } from './chat/components/app-header';
import { Composer } from './chat/components/composer';
import { MessageList } from './chat/components/message-list';
import { Sidebar, type SidebarProps } from './chat/components/sidebar';
import { Dialog, DialogContent } from './ui/dialog';
import {
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
} from './chat/api';
import { fmtInt, fmtNum, localDateYmd } from './chat/format';
import { loadMarkdown } from './chat/markdown';
import {
  clearSystemPrompt as clearStoredSystemPrompt,
  isMaxTokensValid,
  readMaxTokens,
  readShowStats,
  readSystemPrompt,
  readTemperature,
  readTopP,
  writeMaxTokens,
  writeShowStats,
  writeSystemPrompt,
  writeTemperature,
  writeTopP,
} from './chat/storage';
import type { Bubble, ConvRecord, ModelRecord, Sampling, Toast } from './chat/types';

/** One paint per window: faster than the eye, far cheaper than one per token. */
const STREAM_FLUSH_MS = 60;
/** The screen reader needs the region emptied before the same text re-announces. */
const ANNOUNCE_DELAY_MS = 100;
/** The tok/s counter is a live region, so it refreshes at most once a second. */
const TPS_UPDATE_MS = 1000;
const MAX_IMAGE_BYTES = 10_485_760;
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
const DRAWER_BREAKPOINT = '(max-width: 700px)';

const HELP_TEXT = [
  '**Commands:**',
  '- `/clear` / `/reset`: Clear conversation and KV cache',
  '- `/stats`: Toggle generation statistics',
  '- `/context` / `/ctx`: Show context window usage',
  '- `/model`: Show model name',
  '- `/help`: Show this help',
  '',
  '**Shortcuts:**',
  '- `Enter`: Send message',
  '- `Shift+Enter`: New line',
  '- `Escape`: Stop generation or close dialog',
  '',
  'Use the settings panel to configure temperature, top-p, max tokens, and system prompt.',
].join('\n');

function ChatApp() {
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [conversations, setConversations] = useState<ConvRecord[] | null>(null);
  const [convError, setConvError] = useState<string | null>(null);
  const [model, setModel] = useState<ModelRecord | null>(null);
  const [modelResolved, setModelResolved] = useState(false);
  const [context, setContext] = useState<{ used: number; max: number } | null>(null);
  const [vision, setVision] = useState(false);
  const [visionKnown, setVisionKnown] = useState(false);
  const [pendingImage, setPendingImage] = useState<string | null>(null);
  const [sampling, setSampling] = useState<Sampling>({
    temperature: readTemperature(),
    topP: readTopP(),
    maxTokens: readMaxTokens(),
    system: readSystemPrompt(),
  });
  const [showStats, setShowStats] = useState(readShowStats);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isDrawer, setIsDrawer] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [tps, setTps] = useState<number | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const [focusToken, setFocusToken] = useState(0);

  const nextBubbleId = useRef(1);
  const nextToastId = useRef(1);
  const abortRef = useRef<AbortController | null>(null);
  const flushTimer = useRef<number | null>(null);
  const pendingText = useRef('');
  const announceTimer = useRef<number | null>(null);
  const tokenCount = useRef(0);
  const streamStart = useRef(0);
  const lastTokenAt = useRef(0);
  const stoppedRef = useRef(false);
  const modelResolvedRef = useRef(false);
  const selectSeq = useRef(0);
  const autoScroll = useRef(true);

  const modelName = model?.id ?? '';
  const backendName = model?.backend ?? '';
  const lastAssistantId = useMemo(function () {
    for (let index = bubbles.length - 1; index >= 0; index--) {
      if (bubbles[index]?.role === 'assistant') { return bubbles[index]?.id ?? null; }
    }
    return null;
  }, [bubbles]);

  const focusComposer = useCallback(function () {
    setFocusToken(function (token) { return token + 1; });
  }, []);

  const announce = useCallback(function (text: string) {
    if (announceTimer.current !== null) { window.clearTimeout(announceTimer.current); }
    setAnnouncement('');
    announceTimer.current = window.setTimeout(function () { setAnnouncement(text); }, ANNOUNCE_DELAY_MS);
  }, []);

  const pushToast = useCallback(function (text: string, level: Toast['level'] = 'error', retry?: () => void) {
    const id = nextToastId.current;
    nextToastId.current += 1;
    const toast: Toast = retry === undefined ? { id, text, level } : { id, text, level, action: 'Retry', onAction: retry };
    setToasts(function (previous) { return [...previous, toast]; });
  }, []);

  const dismissToast = useCallback(function (id: number) {
    setToasts(function (previous) { return previous.filter(function (toast) { return toast.id !== id; }); });
  }, []);

  // ── model metadata ─────────────────────────────────────────────

  /** Refresh `/v1/models`. Resolves with the record, or null when the server
   *  reports no model or the call fails; the caller decides what to announce. */
  const refreshModel = useCallback(function (): Promise<ModelRecord | null> {
    return loadModels().then(
      function (record) {
        if (!record) { return null; }
        modelResolvedRef.current = true;
        setModel(record);
        setModelResolved(true);
        setVision(record.vision === true);
        setVisionKnown(true);
        setContext({ used: record.kv_seq_len ?? 0, max: record.ctx_size ?? 0 });
        return record;
      },
      function () { return null; },
    );
  }, []);

  const offline = useCallback(function () {
    // A failed refresh after a name is known only means the numbers are stale,
    // so a live model is never replaced by an offline badge.
    if (!modelResolvedRef.current) { setModelResolved(false); }
  }, []);

  // A loaded model without a vision encoder cannot read an attached image, so
  // the pending one is dropped rather than silently sent.
  useEffect(function () {
    if (visionKnown && !vision && pendingImage) {
      setPendingImage(null);
      pushToast('This model cannot view images.');
    }
  }, [visionKnown, vision, pendingImage, pushToast]);

  // ── conversation list ──────────────────────────────────────────

  const loadConvs = useCallback(function () {
    return loadConversations().then(
      function (records) {
        setConvError(null);
        setConversations(records);
      },
      function () { setConvError('Could not load conversations.'); },
    );
  }, []);

  useEffect(function () {
    void refreshModel().then(function (record) { if (!record) { offline(); } });
  }, [refreshModel, offline]);
  useEffect(function () { void loadConvs(); }, [loadConvs]);

  // ── viewport ───────────────────────────────────────────────────

  useEffect(function () {
    const query = globalThis.matchMedia(DRAWER_BREAKPOINT);
    const sync = function () {
      setIsDrawer(query.matches);
      // Leaving the drawer breakpoint must drop the sheet, or the chat stays
      // behind a scrim with no visible way to close it.
      if (!query.matches) { setDrawerOpen(false); }
    };
    sync();
    query.addEventListener('change', sync);
    return function () { query.removeEventListener('change', sync); };
  }, []);

  // ── keyboard ───────────────────────────────────────────────────

  const stopGeneration = useCallback(function () { abortRef.current?.abort(); }, []);

  useEffect(function () {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== 'Escape') {return;}
      if (aboutOpen) { setAboutOpen(false); }
      else if (drawerOpen) { setDrawerOpen(false); }
      else if (settingsOpen) { setSettingsOpen(false); }
      else if (abortRef.current) { stopGeneration(); }
    }
    document.addEventListener('keydown', onKeyDown);
    return function () { document.removeEventListener('keydown', onKeyDown); };
  }, [aboutOpen, drawerOpen, settingsOpen, stopGeneration]);

  // ── turn lifecycle ─────────────────────────────────────────────

  const patchBubble = useCallback(function (id: number, patch: Partial<Bubble>) {
    setBubbles(function (previous) {
      return previous.map(function (bubble) { return bubble.id === id ? { ...bubble, ...patch } : bubble; });
    });
  }, []);

  const flushNow = useCallback(function (id: number, content: string, patch: Partial<Bubble>) {
    if (flushTimer.current !== null) { window.clearTimeout(flushTimer.current); flushTimer.current = null; }
    pendingText.current = '';
    patchBubble(id, { text: content, ...patch });
  }, [patchBubble]);

  /** Queue a streaming paint. The timer arms once and then repaints at a fixed
   *  cadence with whatever text has arrived, so a burst of tokens costs one
   *  render per window rather than one per token. */
  const schedulePaint = useCallback(function (id: number, content: string) {
    pendingText.current = content;
    if (flushTimer.current !== null) {return;}
    flushTimer.current = window.setTimeout(function () {
      flushTimer.current = null;
      const next = pendingText.current;
      pendingText.current = '';
      patchBubble(id, { text: next, phase: 'streaming' });
    }, STREAM_FLUSH_MS);
  }, [patchBubble]);

  const runTurn = useCallback(function (body: string, errorLabel: string, url?: string) {
    const id = nextBubbleId.current;
    nextBubbleId.current += 1;
    setBubbles(function (previous) { return [...previous, { id, role: 'assistant', text: '', phase: 'thinking' }]; });
    // Start the markdown fetch alongside the request: by the time the last chunk
    // renders, marked and DOMPurify are normally already in place.
    void loadMarkdown();

    const controller = new AbortController();
    abortRef.current = controller;
    stoppedRef.current = false;
    setStreaming(true);
    setTps(0);
    tokenCount.current = 0;
    streamStart.current = performance.now();
    lastTokenAt.current = 0;
    announce('Generating response…');

    let content = '';
    const request = { body, signal: controller.signal, requestId: newRequestId(), ...(url === undefined ? {} : { url }) };
    void streamChat(request, {
      onText: function (next) {
        content = next;
        tokenCount.current += 1;
        const now = performance.now();
        if (now - lastTokenAt.current >= TPS_UPDATE_MS) {
          lastTokenAt.current = now;
          setTps(tokenCount.current / ((now - streamStart.current) / 1000));
        }
        schedulePaint(id, next);
      },
      onStats: function (stats) { patchBubble(id, { stats }); },
    }).then(
      function () { flushNow(id, content || 'No response.', { phase: 'done' }); },
      function (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          stoppedRef.current = true;
          flushNow(id, content || 'Stopped.', { phase: 'done' });
          return;
        }
        const message = `${errorLabel}: ${userFacingError(error)}`;
        flushNow(id, message, { phase: 'error', error: message });
        announce(message);
      },
    ).then(function () {
      abortRef.current = null;
      setStreaming(false);
      setTps(null);
      // One context refresh per turn, not one per finalize.
      void refreshModel().then(function (record) { if (!record) { offline(); } });
      void loadConvs();
      focusComposer();
      announce(stoppedRef.current ? 'Generation stopped.' : 'Response complete.');
    });
  }, [announce, flushNow, focusComposer, loadConvs, offline, patchBubble, refreshModel, schedulePaint]);

  // ── conversation actions ───────────────────────────────────────

  const clearChat = useCallback(function () {
    if (bubbles.length > 0 && !globalThis.confirm('Clear this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    abortRef.current?.abort();
    setPendingImage(null);
    void clearServerConversation().then(
      function () {
        setBubbles([]);
        setDrawerOpen(false);
        focusComposer();
        void loadConvs();
        announce('Conversation cleared');
      },
      function () {
        setBubbles([]);
        setDrawerOpen(false);
        focusComposer();
        // Failure styling: every other failed action in the UI toasts in red.
        pushToast('Could not clear on the server. The view was reset locally, but the conversation is still stored.');
      },
    );
  }, [announce, bubbles.length, focusComposer, loadConvs, pushToast]);

  const newConversation = useCallback(function () {
    abortRef.current?.abort();
    setPendingImage(null);
    void createConversation().then(
      function () {
        setBubbles([]);
        setDrawerOpen(false);
        focusComposer();
        void loadConvs();
        announce('New conversation started');
      },
      function () {
        void loadConvs();
        pushToast('Could not create a new conversation. Check that the server is running.');
      },
    );
  }, [announce, focusComposer, loadConvs, pushToast]);

  const openConversation = useCallback(function (id: string) {
    abortRef.current?.abort();
    setPendingImage(null);
    selectSeq.current += 1;
    const seq = selectSeq.current;
    setLoadingConversation(true);
    setBubbles([]);
    announce('Loading conversation…');
    void selectConversation(id).then(
      function (data) {
        if (seq !== selectSeq.current) {return;}
        setLoadingConversation(false);
        setDrawerOpen(false);
        const messages = data.messages ?? [];
        if (messages.length === 0) {
          focusComposer();
          void loadConvs();
          return;
        }
        setBubbles(messages.map(function (message) {
          return {
            id: nextBubbleId.current++,
            role: message.role === 'user' ? 'user' : 'assistant',
            text: message.content,
            phase: 'done',
          };
        }));
        void loadConvs();
        focusComposer();
      },
      function () {
        if (seq !== selectSeq.current) {return;}
        setLoadingConversation(false);
        setDrawerOpen(false);
        pushToast('Failed to load conversation. Check that the server is running.', 'error', function () { openConversation(id); });
      },
    );
  }, [announce, focusComposer, loadConvs, pushToast]);

  const removeConversation = useCallback(function (id: string) {
    if (!globalThis.confirm('Delete this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    void deleteConversation(id).then(
      function (data) {
        void loadConvs();
        if (data.cleared) { setBubbles([]); }
        focusComposer();
        // Deleting another conversation changes nothing on screen, so the row
        // vanishing is the only feedback; say it landed.
        pushToast('Conversation deleted.', 'info');
      },
      function () { pushToast('Could not delete that conversation. Check that the server is running.'); },
    );
  }, [focusComposer, loadConvs, pushToast]);

  const exportConversation = useCallback(function () {
    if (bubbles.length === 0) { pushToast('Nothing to export.', 'info'); return; }
    let markdown = '';
    for (const bubble of bubbles) {
      markdown += `## ${bubble.role === 'user' ? 'User' : 'Assistant'}\n\n${bubble.text.trim()}\n\n`;
    }
    const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `agave-chat-${localDateYmd()}.md`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    pushToast('Conversation exported.', 'info');
  }, [bubbles, pushToast]);

  // ── commands ───────────────────────────────────────────────────

  const runCommand = useCallback(function (command: string) {
    const reply = function (text: string) {
      const id = nextBubbleId.current;
      nextBubbleId.current += 1;
      setBubbles(function (previous) { return [...previous, { id, role: 'assistant', text, phase: 'done' }]; });
    };
    if (command === '/help') { reply(HELP_TEXT); return; }
    if (command === '/stats') {
      setShowStats(function (previous) {
        const next = !previous;
        writeShowStats(next);
        reply(`Statistics ${next ? 'enabled' : 'disabled'}.`);
        return next;
      });
      return;
    }
    if (command === '/context' || command === '/ctx') {
      void refreshModel().then(function (record) {
        const used = record?.kv_seq_len ?? 0;
        const max = record?.ctx_size ?? 0;
        if (!record || max <= 0) { reply('Could not retrieve context info.'); return; }
        const percent = fmtNum((used / max) * 100, 1);
        reply(`Context: **${fmtInt(used)} / ${fmtInt(max)}** tokens (${percent}% used)`);
      });
      return;
    }
    if (command === '/model') { reply(`Model: **${modelName || 'unknown'}**`); return; }
    if (command === '/reset' || command === '/clear') { clearChat(); return; }
    // Unknown command: give feedback like the REPL does, instead of silently
    // sending the "/..." text to the model as a chat message.
    reply(`Unknown command: \`${command}\`\n\nType \`/help\` to see the available commands.`);
    announce(`Unknown command ${command}`);
  }, [announce, clearChat, modelName, refreshModel]);

  // ── sending ────────────────────────────────────────────────────

  const sendMessage = useCallback(function (text: string, image: string | null) {
    let body = `message=${encodeURIComponent(text)}&stream=1${samplingParams(sampling)}`;
    if (image) { body += `&image=${encodeURIComponent(image)}`; }
    runTurn(body, 'Failed to get response');
  }, [runTurn, sampling]);

  const submitMessage = useCallback(function (text: string, image: string | null) {
    if (streaming) {return;}
    // Commands run client-side and never post an image, so the bubble would
    // show an attachment the model never received. Refuse and keep both.
    if (image && text.startsWith('/')) {
      pushToast('Slash commands do not send images. Remove the image or send it as a message.');
      return;
    }
    const id = nextBubbleId.current;
    nextBubbleId.current += 1;
    setBubbles(function (previous) {
      return [...previous, { id, role: 'user', text: text || '(image)', image, phase: 'done' }];
    });
    setPendingImage(null);
    if (text.startsWith('/')) { runCommand(text); } else { sendMessage(text, image); }
  }, [pushToast, runCommand, sendMessage, streaming]);

  const regenerate = useCallback(function () {
    if (streaming) {return;}
    setBubbles(function (previous) {
      const index = previous.map(function (bubble) { return bubble.role === 'assistant'; }).lastIndexOf(true);
      return index === -1 ? previous : previous.filter(function (_bubble, at) { return at !== index; });
    });
    runTurn(`stream=1${samplingParams(sampling)}`, 'Failed to regenerate', '/v1/chat/regenerate');
  }, [runTurn, sampling, streaming]);

  const attachImage = useCallback(function (file: File, label: string) {
    if (visionKnown && !vision) {
      pushToast('This model cannot view images.');
      return;
    }
    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      pushToast('Unsupported image format. Use JPEG, PNG, GIF, or WebP.');
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      pushToast('Image too large (max 10 MB).');
      return;
    }
    const reader = new FileReader();
    reader.addEventListener('load', function (event) {
      const result = event.target?.result;
      if (typeof result !== 'string') {return;}
      setPendingImage(result);
      announce(label);
    });
    reader.addEventListener('error', function () { pushToast('Could not read that image. Try another file.'); });
    reader.readAsDataURL(file);
  }, [announce, pushToast, vision, visionKnown]);

  const updateSampling = useCallback(function (next: Sampling) {
    setSampling(next);
    writeTemperature(next.temperature);
    writeTopP(next.topP);
    // Only a value that will be sent is stored, so a half-typed field cannot
    // survive a reload.
    if (isMaxTokensValid(next.maxTokens)) { writeMaxTokens(next.maxTokens); }
    writeSystemPrompt(next.system);
  }, []);

  const clearSystem = useCallback(function () {
    setSampling(function (previous) { return { ...previous, system: '' }; });
    clearStoredSystemPrompt();
    announce('System prompt cleared');
  }, [announce]);

  const onRendered = useCallback(function (_id: number, rendered: string) {
    announce(`Agave responded: ${rendered}`);
  }, [announce]);

  // ── render ─────────────────────────────────────────────────────

  const sidebarProps: SidebarProps = {
    conversations,
    loadError: convError,
    onSelect: openConversation,
    onDelete: removeConversation,
    onNew: newConversation,
    onRetryLoad: function () { void loadConvs(); },
  };

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <a
        href="#msg"
        className="skip-link absolute start-4 top-[-100%] z-50 rounded-lg bg-primary px-4 py-2 font-mono text-sm font-medium text-primary-foreground no-underline transition-[top] duration-200 focus:top-2"
      >
        Skip to message input
      </a>
      <AppHeader
        model={model}
        modelResolved={modelResolved}
        ctx={context}
        sidebarOpen={drawerOpen}
        onOpenSidebar={function () { setDrawerOpen(true); }}
        onRetryModel={function () { setModelResolved(false); void refreshModel().then(function (record) { if (!record) { offline(); } }); }}
        onNew={newConversation}
        onExport={exportConversation}
        onClear={clearChat}
        onAbout={function () { setAboutOpen(true); }}
      />
      <div className="relative z-1 flex flex-1 overflow-hidden">
        {isDrawer ? null : (
          <aside id="sidebar" aria-label="Conversations" className="z-20 w-sidebar shrink-0 border-e border-border">
            <Sidebar {...sidebarProps} />
          </aside>
        )}
        <main className="flex min-w-0 flex-1 flex-col">
          <MessageList
            bubbles={bubbles}
            toasts={toasts}
            showStats={showStats}
            vision={vision}
            streaming={streaming}
            loading={loadingConversation}
            lastAssistantId={lastAssistantId}
            onRegenerate={regenerate}
            onRendered={onRendered}
            onRunCommand={runCommand}
            onDismissToast={dismissToast}
            onScroll={function (nearBottom) { autoScroll.current = nearBottom; }}
          />
          <Composer
            sampling={sampling}
            onSamplingChange={updateSampling}
            onSubmit={submitMessage}
            streaming={streaming}
            onStop={stopGeneration}
            vision={vision}
            pendingImage={pendingImage}
            onImageFile={attachImage}
            onRemoveImage={function () { setPendingImage(null); announce('Image removed'); }}
            onClearSystem={clearSystem}
            tps={tps}
            settingsOpen={settingsOpen}
            onToggleSettings={function () {
              setSettingsOpen(function (open) {
                announce(`Settings panel ${open ? 'closed' : 'opened'}`);
                return !open;
              });
            }}
            focusToken={focusToken}
          />
        </main>
      </div>
      <div id="sr-announce" className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
      <AboutDialog open={aboutOpen} onOpenChange={setAboutOpen} modelName={modelName} backendName={backendName} />
      {isDrawer ? (
        // Radix supplies the scrim, the focus trap and the inert backdrop the
        // hand-rolled drawer reimplemented.
        <Dialog open={drawerOpen} onOpenChange={setDrawerOpen}>
          <DialogContent side="left" hideClose className="p-0">
            <Sidebar {...sidebarProps} onClose={function () { setDrawerOpen(false); }} />
          </DialogContent>
        </Dialog>
      ) : null}
    </div>
  );
}

const root = document.getElementById('root');
if (!root) { throw new Error('missing #root'); }
createRoot(root).render(<ChatApp />);
