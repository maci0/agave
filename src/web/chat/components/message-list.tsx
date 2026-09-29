import { CloseIcon, JumpToLatestIcon } from '../../ui/icons';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Message } from './message';
import { Button } from '../../ui/button';
import type { Bubble, Toast } from '../types';
import { EmptyState } from '../../ui/empty-state';
import { HintAction, HintChip } from '../../ui/hint-chip';
import { cn } from '../../ui/cn';

const TOAST_INFO_MS = 5000;
const TOAST_ERROR_MS = 12_000;
/** A message the user is reading gets twice as long before it leaves. */
const REDUCED_MOTION_FACTOR = 2;

const ToastItem = ({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) => {
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    if (paused) { return undefined; }
    const base = toast.level === 'error' ? TOAST_ERROR_MS : TOAST_INFO_MS;
    const doubled = globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const timer = setTimeout(() => { onDismiss(toast.id); }, doubled ? base * REDUCED_MOTION_FACTOR : base);
    return function () { clearTimeout(timer); };
  }, [paused, toast.id, toast.level, onDismiss]);

  return (
    <div
      role={toast.level === 'error' ? 'alert' : 'status'}
      onMouseEnter={function () { setPaused(true); }}
      onMouseLeave={function () { setPaused(false); }}
      onFocus={function () { setPaused(true); }}
      onBlur={function () { setPaused(false); }}
      className={cn(
        'mx-auto flex w-full agave-measure items-center gap-2 rounded-lg border px-4.5 py-3 text-sm shadow-overlay',
        toast.level === 'error'
          ? 'border-destructive bg-destructive/10 text-destructive-foreground'
          : 'border-border-strong bg-primary/10 text-muted-foreground',
      )}
    >
      <span className="flex-1">{toast.text}</span>
      {toast.action ? (
        <button
          type="button"
          onClick={function () { onDismiss(toast.id); toast.onAction?.(); }}
          aria-label={`Retry: ${toast.text}`}
          className="inline-flex min-h-11 items-center rounded-sm border border-current px-3 font-mono text-xs"
        >
          {toast.action}
        </button>
      ) : null}
      <button
        type="button"
        onClick={function () { onDismiss(toast.id); }}
        aria-label="Dismiss"
        className="inline-flex size-11 shrink-0 items-center justify-center opacity-70 transition-opacity hover:opacity-100"
      >
        <CloseIcon className="size-5" aria-hidden="true" />
      </button>
    </div>
  );
};

const TranscriptEmptyState = ({ vision, onRunCommand }: { vision: boolean; onRunCommand: (command: string) => void }) => (
  <EmptyState
    title="Prompt the model"
    line="Runs locally. Conversations stay on this machine."
    hints={
      <>
        <HintAction label="Show the command list" onClick={function () { onRunCommand('/help'); }}>/help for commands</HintAction>
        <HintChip>Enter to send</HintChip>
        {vision ? <HintChip>Paste or drop an image</HintChip> : null}
      </>
    }
  />
);
type MessageListProps = {
bubbles: Array<Bubble>;
toasts: Array<Toast>;
showStats: boolean;
vision: boolean;
streaming: boolean;
/** A conversation fetch is in flight; the log says so instead of guessing. */
loading: boolean;
lastAssistantId: number | null;
onRegenerate: () => void;
onRendered: (id: number, rendered: string) => void;
onRunCommand: (command: string) => void;
onDismissToast: (id: number) => void;
/** Reports whether the log is still within a thumb of the newest content. */
onScroll?: (nearBottom: boolean) => void;
};
/** The toasts, oldest last, in the same column as the transcript. */
const ToastList = ({ toasts, onDismiss }: { toasts: Array<Toast>; onDismiss: (id: number) => void }) => (
  <>
    {toasts.map((toast) =>
      <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
    )}
  </>
);

/** Shown only while the reader is scrolled away from the newest turn. */
const JumpToLatest = ({ onClick }: { onClick: () => void }) => (
  <Button type="button" variant="floating" size="sm" onClick={onClick}>
    <JumpToLatestIcon className="size-4" aria-hidden="true" />
    Jump to latest
  </Button>
);

/** Toasts and the jump key ride above the transcript instead of inside it: a
 *  reader who has scrolled up would otherwise never see a failure, and the log
 *  used to yank them back to the bottom to make one visible. */
const LogOverlay = ({ toasts, showJump, onJump, onDismiss }: {
  toasts: Array<Toast>;
  showJump: boolean;
  onJump: () => void;
  onDismiss: (id: number) => void;
}) => {
  if (toasts.length === 0 && !showJump) { return null; }
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex flex-col items-center gap-2 p-4">
      {showJump ? <div className="pointer-events-auto"><JumpToLatest onClick={onJump} /></div> : null}
      {toasts.length === 0 ? null : (
        <div className="pointer-events-auto flex w-full flex-col">
          <ToastList toasts={toasts} onDismiss={onDismiss} />
        </div>
      )}
    </div>
  );
};

/** Pixels from the bottom that still count as "following the stream". */
const STICK_SLACK_PX = 80;
/** The chat log. It owns its scroll position: a reader who scrolls up is left
 *  alone, and a reader at the bottom is carried along by every new turn. */
export const MessageList = memo((props: MessageListProps) => {
const ref = useRef<HTMLDivElement>(null);
const nearBottom = useRef(true);
const [showJump, setShowJump] = useState(false);
useEffect(() => {
  const log = ref.current;
  if (log && nearBottom.current) { log.scrollTop = log.scrollHeight; }
}, [props.bubbles]);
const jumpToLatest = useCallback(() => {
  const log = ref.current;
  if (!log) { return; }
  log.scrollTop = log.scrollHeight;
  nearBottom.current = true;
  setShowJump(false);
}, []);
return (
  <div className="relative flex min-h-0 flex-1 flex-col">
  <div
    id="chat"
    ref={ref}
    role="log"
    aria-label="Chat messages"
    aria-live="off"
    aria-busy={props.streaming}
    tabIndex={0}
    onScroll={function (event) {
      const log = event.currentTarget;
      nearBottom.current = log.scrollHeight - log.scrollTop - log.clientHeight < STICK_SLACK_PX;
      // A reader who scrolls up during a stream otherwise gets no sign that
      // The turn kept growing below the fold, and no way back to it.
      setShowJump(!nearBottom.current);
      props.onScroll?.(nearBottom.current);
    }}
    className={cn(
      'agave-scroll flex flex-1 flex-col gap-6 overflow-y-auto px-6 pt-6 focus-visible:-outline-offset-2 focus-visible:outline-2 focus-visible:outline-primary max-drawer:px-4 max-drawer:pt-4',
      // Room for the toast overlay, so it never covers the newest turn.
      props.toasts.length > 0 ? 'pb-28' : 'pb-2',
    )}
  >
    {props.bubbles.length === 0 && !props.loading ? (
      <TranscriptEmptyState vision={props.vision} onRunCommand={props.onRunCommand} />
    ) : null}
    {props.loading ? <div role="status" className="m-auto font-mono text-xs text-faint">Loading conversation…</div> : null}
    {props.bubbles.map((bubble) =>
      (
        <Message
          key={bubble.id}
          bubble={bubble}
          showStats={props.showStats}
          canRegenerate={bubble.id === props.lastAssistantId && !props.streaming}
          onRegenerate={props.onRegenerate}
          onRendered={props.onRendered}
        />
      )
    )}
  </div>
  <LogOverlay toasts={props.toasts} showJump={showJump} onJump={jumpToLatest} onDismiss={props.onDismissToast} />
  </div>
);
});