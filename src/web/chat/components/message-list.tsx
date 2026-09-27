import { X } from 'lucide-react';
import { memo, useEffect, useRef, useState } from 'react';
import { Message } from './message';
import type { Bubble, Toast } from '../types';
import { cn } from '../../ui/cn';

const TOAST_INFO_MS = 5000;
const TOAST_ERROR_MS = 12_000;
/** A message the user is reading gets twice as long before it leaves. */
const REDUCED_MOTION_FACTOR = 2;

const ToastItem = ({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) => {
  const [paused, setPaused] = useState(false);
  useEffect(function () {
    if (paused) { return undefined; }
    const base = toast.level === 'error' ? TOAST_ERROR_MS : TOAST_INFO_MS;
    const doubled = globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const timer = setTimeout(function () { onDismiss(toast.id); }, doubled ? base * REDUCED_MOTION_FACTOR : base);
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
        'mx-auto my-2 flex w-full max-w-prose items-center gap-2 rounded-lg border px-[18px] py-3 text-sm',
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
        <X className="size-5" aria-hidden="true" />
      </button>
    </div>
  );
};

const EmptyState = ({ vision, onRunCommand }: { vision: boolean; onRunCommand: (command: string) => void }) => (
  <div className="m-auto px-5 py-10 text-center">
    <span className="mark mark-lg" aria-hidden="true" />
    <h2 className="mb-2 font-mono text-lg font-semibold text-foreground">Prompt the model</h2>
    <p className="mb-6 text-base text-muted-foreground">Runs locally. Conversations stay on this machine.</p>
    <div className="flex flex-wrap justify-center gap-2">
      <button
        type="button"
        onClick={function () { onRunCommand('/help'); }}
        className="inline-flex min-h-11 items-center justify-center rounded-lg border border-border bg-card px-2.5 py-1 font-mono text-2xs text-faint underline decoration-primary decoration-2 underline-offset-2 transition-colors hover:border-primary hover:text-primary"
      >
        /help for commands
      </button>
      <span className="inline-flex min-h-11 items-center rounded-lg border border-border bg-card px-2.5 py-1 font-mono text-2xs text-faint">
        Enter to send
      </span>
      {vision ? (
        <span className="inline-flex min-h-11 items-center rounded-lg border border-border bg-card px-2.5 py-1 font-mono text-2xs text-faint">
          Paste or drop an image
        </span>
      ) : null}
    </div>
  </div>
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
/** Pixels from the bottom that still count as "following the stream". */
const STICK_SLACK_PX = 80;
/** The chat log. It owns its scroll position: a reader who scrolls up is left
 *  alone, and a reader at the bottom is carried along by every new turn. */
export const MessageList = memo(function MessageList(props: MessageListProps) {
const ref = useRef<HTMLDivElement>(null);
const nearBottom = useRef(true);
useEffect(function () {
  const log = ref.current;
  if (log && nearBottom.current) { log.scrollTop = log.scrollHeight; }
}, [props.bubbles, props.toasts]);
return (
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
      props.onScroll?.(nearBottom.current);
    }}
    className="agave-scroll flex flex-1 flex-col gap-6 overflow-y-auto px-6 pt-6 pb-2 focus-visible:-outline-offset-2 focus-visible:outline-2 focus-visible:outline-primary max-drawer:px-4 max-drawer:pt-4"
  >
    {props.bubbles.length === 0 && props.toasts.length === 0 && !props.loading ? (
      <EmptyState vision={props.vision} onRunCommand={props.onRunCommand} />
    ) : null}
    {props.loading ? <div role="status" className="m-auto font-mono text-xs text-faint">Loading conversation…</div> : null}
    {props.bubbles.map(function (bubble) {
      return (
        <Message
          key={bubble.id}
          bubble={bubble}
          showStats={props.showStats}
          canRegenerate={bubble.id === props.lastAssistantId && !props.streaming}
          onRegenerate={props.onRegenerate}
          onRendered={props.onRendered}
        />
      );
    })}
    {props.toasts.map(function (toast) {
      return <ToastItem key={toast.id} toast={toast} onDismiss={props.onDismissToast} />;
    })}
  </div>
);
});