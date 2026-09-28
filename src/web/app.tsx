// Server chat UI, embedded by src/server/server.zig, not the WASM shell.
// Source of truth is the .tsx sources: `scripts/build-web.sh`
// Refreshes the committed app.js and style.css.

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { AboutDialog } from './chat/components/about-dialog';
import { AppHeader } from './chat/components/app-header';
import { Composer as ComposerForm } from './chat/components/composer';
import { MessageList } from './chat/components/message-list';
import { Sidebar, type SidebarProps } from './chat/components/sidebar';
import { Dialog, DialogContent } from './ui/dialog';
import {
  type Announce,
  type PushToast,
  chatRequestBody,
  regenerateBody,
  useAnnouncer,
  useBubbleLog,
  useChatTurn,
  useConversations,
  useImageAttachment,
  useModelInfo,
  useSettings,
  useToasts,
} from './chat/hooks';
import { fmtInt, fmtNum, localDateYmd } from './chat/format';
import type { Toast } from './chat/types';

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

type ChatShellProps = {
  header: ReactNode;
  sidebar: ReactNode;
  log: ReactNode;
  composer: ReactNode;
  announcement: string;
  about: ReactNode;
};

/** The page frame: skip link, header, the two columns, and the live region. */
const ChatShell = ({ header, sidebar, log, composer, announcement, about }: ChatShellProps) => (
  <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
    <a
      href="#msg"
      className="absolute start-4 top-[-100%] z-50 rounded-lg bg-primary px-4 py-2 font-mono text-sm font-medium text-primary-foreground no-underline transition-[top] duration-200 focus:top-2"
    >
      Skip to message input
    </a>
    {header}
    <div className="relative z-1 flex flex-1 overflow-hidden">
      {sidebar}
      <main className="flex min-w-0 flex-1 flex-col">
        {log}
        {composer}
      </main>
    </div>
    <div id="sr-announce" className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
    {about}
  </div>
);

/** Render the sidebar, as a desktop column or as a sheet below the breakpoint. */
const ConversationPanel = ({
  isDrawer,
  drawerOpen,
  onDrawerChange,
  sidebarProps,
}: {
  isDrawer: boolean;
  drawerOpen: boolean;
  onDrawerChange: (open: boolean) => void;
  sidebarProps: SidebarProps;
}) => {
  if (!isDrawer) {
    return (
      <aside id="sidebar" aria-label="Conversations" className="z-20 w-sidebar shrink-0 border-e border-border">
        <Sidebar {...sidebarProps} />
      </aside>
    );
  }
  // Radix supplies the scrim, the focus trap and the inert backdrop.
  return (
    <Dialog open={drawerOpen} onOpenChange={onDrawerChange}>
      <DialogContent side="left" hideClose className="p-0">
        <Sidebar {...sidebarProps} onClose={function () { onDrawerChange(false); }} />
      </DialogContent>
    </Dialog>
  );
};

/** Answer `/context` from the last `/v1/models` refresh. */
const reportContext = (reply: (text: string) => void, model: ReturnType<typeof useModelInfo>): void => {
  void model.refresh().then(function (record) {
    const used = record?.kv_seq_len ?? 0;
    const max = record?.ctx_size ?? 0;
    if (record === null || max <= 0) { reply('Could not retrieve context info.'); return; }
    reply(`Context: **${fmtInt(used)} / ${fmtInt(max)}** tokens (${fmtNum((used / max) * 100, 1)}% used)`);
  });
};

/** A counter the composer watches to pull focus back after a turn. */
const useFocusToken = () => {
  const [focusToken, setFocusToken] = useState(0);
  const focusComposer = useCallback(function () {
    setFocusToken(function (token) { return token + 1; });
  }, []);
  return { focusToken, focusComposer };
};

/** The chrome around the conversation: the About dialog, the mobile drawer, the
 *  viewport breakpoint, and the Escape ladder that closes one layer at a time. */
const useShellChrome = (settings: ReturnType<typeof useSettings>, stopGeneration: () => void) => {
  const [aboutOpen, setAboutOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isDrawer, setIsDrawer] = useState(false);

  useEffect(function () {
    const query = globalThis.matchMedia(DRAWER_BREAKPOINT);
    const sync = function () {
      setIsDrawer(query.matches);
      // Leaving the drawer breakpoint drops the sheet; otherwise the chat
      // Stays behind a scrim with no visible way to close it.
      if (!query.matches) { setDrawerOpen(false); }
    };
    sync();
    query.addEventListener('change', sync);
    return function () { query.removeEventListener('change', sync); };
  }, []);

  const { togglePanel: closeSettings, open: settingsOpen } = settings;
  const onEscapeKey = useCallback(function (event: KeyboardEvent) {
    if (event.key !== 'Escape') {return;}
    if (aboutOpen) { setAboutOpen(false); }
    else if (drawerOpen) { setDrawerOpen(false); }
    else if (settingsOpen) { closeSettings(); }
    else { stopGeneration(); }
  }, [aboutOpen, closeSettings, drawerOpen, settingsOpen, stopGeneration]);

  useEffect(function () {
    document.addEventListener('keydown', onEscapeKey);
    return function () { document.removeEventListener('keydown', onEscapeKey); };
  }, [onEscapeKey]);

  return { aboutOpen, setAboutOpen, drawerOpen, setDrawerOpen, isDrawer };
};

/** Every state slice the shell renders, wired together. */
const useChatState = (announce: Announce, focusComposer: () => void) => {
  const { toasts, pushToast, dismissToast } = useToasts();
  const model = useModelInfo();
  const settings = useSettings(announce);
  const log = useBubbleLog();
  const convs = useConversations({ log, announce, pushToast });
  const image = useImageAttachment({ vision: model.vision, visionKnown: model.visionKnown, announce, pushToast });
  const turn = useChatTurn({
    log,
    announce,
    onTurnEnd: function () { void model.refresh(); void convs.load(); focusComposer(); },
  });
  return { toasts, pushToast, dismissToast, model, log, convs, image, turn, settings };
};

/** Everything the reader can do from the log or the composer: send, retry,
 *  export, and the client-side slash commands. */
const useChatCommands = ({ log, model, convs, image, turn, settings, announce, pushToast, focusComposer }: {
  log: ReturnType<typeof useBubbleLog>;
  model: ReturnType<typeof useModelInfo>;
  convs: ReturnType<typeof useConversations>;
  image: ReturnType<typeof useImageAttachment>;
  turn: ReturnType<typeof useChatTurn>;
  settings: ReturnType<typeof useSettings>;
  announce: Announce;
  pushToast: PushToast;
  focusComposer: () => void;
}) => {
  const runSlash = useCallback(function (text: string) {
    const reply = function (message: string) {
      log.append(log.allocate({ role: 'assistant', text: message, phase: 'done' }));
    };
    if (text === '/help') { reply(HELP_TEXT); return; }
    if (text === '/stats') { settings.toggleStats(); reply(`Statistics ${settings.showStats ? 'disabled' : 'enabled'}.`); return; }
    if (text === '/model') { reply(`Model: **${model.modelName || 'unknown'}**`); return; }
    if (text === '/reset' || text === '/clear') { convs.clearAll(); focusComposer(); return; }
    if (text === '/context' || text === '/ctx') { reportContext(reply, model); return; }
    // Unknown command: give feedback like the REPL does, rather than
    // Silently sending the "/..." text to the model as a message.
    reply(`Unknown command: \`${text}\`\n\nType \`/help\` to see the available commands.`);
    announce(`Unknown command ${text}`);
  }, [announce, convs, focusComposer, log, model, settings]);

  const submitMessage = useCallback(function (text: string, attached: string | null) {
    if (turn.streaming) {return;}
    // Commands run client-side and never post an image, so the bubble
    // Would show an attachment the model never received. Keep both.
    if (attached !== null && text.startsWith('/')) {
      pushToast('Slash commands do not send images. Remove the image or send it as a message.');
      return;
    }
    const bubble = log.allocate({ role: 'user', text: text || '(image)', phase: 'done' });
    if (attached !== null) { bubble.image = attached; }
    log.append(bubble);
    image.clear();
    focusComposer();
    if (text.startsWith('/')) { runSlash(text); } else { turn.send(chatRequestBody(text, attached, settings.sampling), 'Failed to get response'); }
  }, [focusComposer, image, log, pushToast, settings.sampling, turn]);

  const regenerate = useCallback(function () {
    if (turn.streaming) {return;}
    log.dropLastAssistant();
    turn.send(regenerateBody(settings.sampling), 'Failed to regenerate', '/v1/chat/regenerate');
  }, [log, settings.sampling, turn]);

  const exportConversation = useCallback(function () {
    const markdown = log.bubbles.map(function (bubble) {
      return `## ${bubble.role === 'user' ? 'User' : 'Assistant'}\n\n${bubble.text.trim()}\n\n`;
    }).join('');
    if (log.bubbles.length === 0) { pushToast('Nothing to export.', 'info'); return; }
    const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `agave-chat-${localDateYmd()}.md`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    pushToast('Conversation exported.', 'info');
  }, [log.bubbles, pushToast]);

  return { runSlash, submitMessage, regenerate, exportConversation };
};

/** The composer form, with the sampling and turn state folded in. */
const Composer = ({ settings, turn, model, image, submit, focusToken, reject }: {
  settings: ReturnType<typeof useSettings>;
  turn: ReturnType<typeof useChatTurn>;
  model: ReturnType<typeof useModelInfo>;
  image: ReturnType<typeof useImageAttachment>;
  submit: (text: string, image: string | null) => void;
  focusToken: number;
  reject: PushToast;
}) => (
  <ComposerForm
    sampling={settings.sampling}
    onSamplingChange={settings.change}
    onSubmit={submit}
    streaming={turn.streaming}
    onStop={turn.stop}
    vision={model.vision}
    pendingImage={image.pending}
    onImageFile={image.attach}
    onRemoveImage={image.clear}
    onDropRejected={reject}
    onClearSystem={settings.clearSystem}
    tps={turn.tps}
    settingsOpen={settings.open}
    onToggleSettings={settings.togglePanel}
    focusToken={focusToken}
  />
);

/** The header, with the conversation actions folded in. */
const ChatHeader = ({ model, convs, turn, chrome, focusComposer, exportConversation }: {
  model: ReturnType<typeof useModelInfo>;
  convs: ReturnType<typeof useConversations>;
  turn: ReturnType<typeof useChatTurn>;
  chrome: ReturnType<typeof useShellChrome>;
  focusComposer: () => void;
  exportConversation: () => void;
}) => (
  <AppHeader
    model={model.model}
    modelResolved={model.modelResolved}
    ctx={model.context}
    streaming={turn.streaming}
    onRetryModel={model.retry}
    onExport={exportConversation}
    onNew={function () { convs.startNew(); focusComposer(); }}
    onClear={function () { convs.clearAll(); focusComposer(); }}
    onAbout={function () { chrome.setAboutOpen(true); }}
    sidebarOpen={chrome.drawerOpen}
    onOpenSidebar={function () { chrome.setDrawerOpen(true); }}
  />
);

/** The transcript, with the log's own state folded in. */
const ChatLogPanel = ({ log, toasts, settings, model, turn, convs, commands, dismissToast }: {
  log: ReturnType<typeof useBubbleLog>;
  toasts: Array<Toast>;
  settings: ReturnType<typeof useSettings>;
  model: ReturnType<typeof useModelInfo>;
  turn: ReturnType<typeof useChatTurn>;
  convs: ReturnType<typeof useConversations>;
  commands: ReturnType<typeof useChatCommands> & { announce: Announce };
  dismissToast: (id: number) => void;
}) => (
  <MessageList
    bubbles={log.bubbles}
    toasts={toasts}
    showStats={settings.showStats}
    vision={model.vision}
    streaming={turn.streaming}
    loading={convs.loading}
    lastAssistantId={log.lastAssistantId}
    onRegenerate={commands.regenerate}
    onRendered={function (_id, rendered) { commands.announce(`Agave responded: ${rendered}`); }}
    onRunCommand={commands.runSlash}
    onDismissToast={dismissToast}
  />
);

const ChatApp = () => {
  const { announcement, announce } = useAnnouncer();
  const { focusToken, focusComposer } = useFocusToken();
  const { toasts, pushToast, dismissToast, model, log, convs, image, turn, settings } = useChatState(announce, focusComposer);
  const chrome = useShellChrome(settings, turn.stop);
  const commands = { ...useChatCommands({ log, model, convs, image, turn, settings, announce, pushToast, focusComposer }), announce };
  const sidebarProps: SidebarProps = {
    conversations: convs.conversations,
    loadError: convs.loadError,
    streaming: turn.streaming,
    onSelect: function (id) { convs.open(id); focusComposer(); },
    onDelete: function (id) { convs.remove(id); focusComposer(); },
    onNew: function () { convs.startNew(); focusComposer(); },
    onRetryLoad: function () { void convs.load(); },
  };

  return (
    <ChatShell
      header={<ChatHeader model={model} convs={convs} turn={turn} chrome={chrome} focusComposer={focusComposer} exportConversation={commands.exportConversation} />}
      sidebar={<ConversationPanel isDrawer={chrome.isDrawer} drawerOpen={chrome.drawerOpen} onDrawerChange={chrome.setDrawerOpen} sidebarProps={sidebarProps} />}
      log={<ChatLogPanel log={log} toasts={toasts} settings={settings} model={model} turn={turn} convs={convs} commands={commands} dismissToast={dismissToast} />}
      composer={<Composer settings={settings} turn={turn} model={model} image={image} submit={commands.submitMessage} focusToken={focusToken} reject={pushToast} />}
      announcement={announcement}
      about={<AboutDialog open={chrome.aboutOpen} onOpenChange={chrome.setAboutOpen} modelName={model.modelName} backendName={model.backendName} />}
    />
  );
};

const root = document.querySelector('#root');
if (root === null) { throw new Error('missing #root'); }
createRoot(root).render(<ChatApp />);
