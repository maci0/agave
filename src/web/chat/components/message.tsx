import { Copy, RefreshCw } from 'lucide-react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { copyText, markdownReady, onIdle, renderMarkdown } from '../markdown';
import { fmtInt, fmtNum, truncateAnnounce } from '../format';
import type { Bubble, StreamStats } from '../types';
import { cn } from '../../ui/cn';

const THINKING_GLYPH = '…';

/** Write `next` into an element, appending only what is new. Rebuilding
 *  textContent on every token re-copied the whole string and re-ran line
 *  wrapping; appending the delta keeps a flush proportional to the tokens that
 *  arrived. A stream that no longer extends what is on screen (a rebuild, a
 *  regenerated turn) refills instead of appending. */
const appendDelta = (element: HTMLElement, painted: string, next: string): string => {
  const node = element.firstChild;
  if (node instanceof Text && next.startsWith(painted)) {
    node.appendData(next.slice(painted.length));
    return next;
  }
  element.textContent = next;
  return next;
};

type MessageBodyProps = {
  text: string;
  phase: Bubble['phase'];
  /** Called once per finished turn with the rendered response text, for the
   *  screen-reader announcement. A late markdown upgrade must not repeat it. */
  onRendered: (rendered: string) => void;
};

/**
 * The body of an assistant turn.
 *
 * Preact owns the element, not its children: the streaming path appends text
 * nodes and the final path fills the same node from sanitized markdown, so a
 * turn that goes from hundreds of partial tokens to a full markdown render
 * costs one reconcile rather than a re-render per token.
 */
export const MessageBody = memo(({ text, phase, onRendered }: MessageBodyProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const painted = useRef('');
  const announced = useRef<string | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {return;}
    if (phase === 'streaming') {
      painted.current = appendDelta(element, painted.current, text);
      return;
    }
    if (phase === 'thinking') {
      element.textContent = THINKING_GLYPH;
      // The glyph is what the node holds, so the next stream paint does not
      // Extend it. `painted` tracks the text on screen, not the response.
      painted.current = THINKING_GLYPH;
      return;
    }
    if (phase === 'error') {
      element.textContent = text;
      painted.current = text;
      return;
    }
    element.textContent = '';
    painted.current = '';
    renderMarkdown(element, text);
    if (announced.current !== text) {
      announced.current = text;
      onRendered(truncateAnnounce(element.textContent, 200));
    }
    if (!markdownReady()) {
      // The libraries were still in flight, so rebuild once they land,
      // One message per idle slot, never a whole history in one task.
      onIdle(() => {
        if (element.isConnected) { renderMarkdown(element, text); }
      });
    }
  }, [text, phase, onRendered]);

  return (
    <div
      ref={ref}
      className={cn('agave-prose min-w-0', phase === 'thinking' && 'animate-pulse-soft text-faint')}
    />
  );
});

const StatsLine = ({ stats }: { stats: StreamStats }) => {
  const total = Math.trunc(Number(stats.time)) + (Math.trunc(Number(stats.pfMs)) || 0);
  const decode = `${fmtInt(stats.tokens)} tok @ ${fmtNum(Number(stats.tps), 2)}`;
  const prefill = `${fmtInt(stats.pfTok)} tok @ ${fmtNum(Number(stats.pfTps), 1)}`;
  return (
    <div className="mt-2.5 flex flex-wrap gap-x-4 border-t border-divider pt-2.5 font-mono text-2xs text-faint">
      <span>
        decode <span className="font-medium text-primary">{decode}</span> tok/s
      </span>
      {stats.pfTok && stats.pfTok !== '0' ? (
        <span>
          prefill <span className="font-medium text-primary">{prefill}</span> tok/s
        </span>
      ) : null}
      {stats.pfMs && stats.pfMs !== '0' ? (
        <span>
          TTFT <span className="font-medium text-primary">{fmtInt(stats.pfMs)}</span> ms
        </span>
      ) : null}
      <span>
        total <span className="font-medium text-primary">{fmtInt(total)}</span> ms
      </span>
    </div>
  );
};

type MessageProps = {
  bubble: Bubble;
  showStats: boolean;
  /** True for the newest assistant turn, the only one that can be retried. */
  canRegenerate: boolean;
  onRegenerate: () => void;
  onRendered: (id: number, rendered: string) => void;
};

const COPY_REVERT_MS = 2000;

/** The row under a response: copy and regenerate. Revealed with the message on
 *  hover or focus, always shown on touch (see .agave-reveal). */
const ACTION =
  'agave-reveal inline-flex min-h-11 items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-2xs text-faint ' +
  'transition-colors hover:bg-muted hover:text-primary';

/** Copy a finished response. */
const CopyResponse = ({ text }: { text: string }) => {
  const [label, setLabel] = useState('Copy');
  const copy = useCallback(() => {
    void copyText(text).then((result) => {
      setLabel(result === 'copied' ? 'Copied' : 'Failed');
      setTimeout(() => { setLabel('Copy'); }, COPY_REVERT_MS);
    });
  }, [text]);
  return (
    <button type="button" onClick={copy} aria-label="Copy response" className={ACTION}>
      <Copy className="size-3.5" aria-hidden="true" />
      {label}
    </button>
  );
};

/** Replay the newest assistant turn. */
const RegenerateButton = ({ retry, onRegenerate }: { retry: boolean; onRegenerate: () => void }) => (
  <button
    type="button"
    onClick={onRegenerate}
    aria-label={retry ? 'Retry generating response' : 'Regenerate response'}
    className={ACTION}
  >
    <RefreshCw className="size-3.5" aria-hidden="true" />
    {retry ? 'Retry' : 'Regenerate'}
  </button>
);

const Message = memo(({ bubble, showStats, canRegenerate, onRegenerate, onRendered }: MessageProps) => {
  const isUser = bubble.role === 'user';
  const roleId = `msg-role-${bubble.id}`;
  const failed = bubble.phase === 'error';
  const canCopy = !isUser && !failed && bubble.phase === 'done';
  const handleRendered = useCallback((rendered: string) => { onRendered(bubble.id, rendered); }, [bubble.id, onRendered]);
  return (
    <div
      role="group"
      aria-labelledby={roleId}
      className={cn(
        'group mx-auto flex w-full agave-measure flex-col gap-1',
        isUser ? 'items-end' : 'items-start',
      )}
    >
      <span
        id={roleId}
        className={cn('inline-flex items-center gap-1.5 px-1 font-mono text-xs font-medium', isUser ? 'text-faint' : 'text-primary')}
      >
        {isUser ? null : <span className="mark mark-sm" aria-hidden="true" />}
        {isUser ? 'You' : 'agave'}
      </span>
      <div
        dir="auto"
        role={failed ? 'alert' : undefined}
        className={cn(
          'relative min-w-0 max-w-full rounded-lg text-base',
          failed && 'w-full border border-destructive bg-destructive/10 px-4.5 py-3.5 text-sm text-destructive-foreground',
          !isUser && !failed && 'w-full px-1 py-1',
          isUser && 'max-w-4/5 rounded-ee-xs bg-primary/10 px-4.5 py-3',
        )}
      >
        {bubble.image === undefined ? null : (
          <img className="mb-2 block max-w-50 rounded-md border border-divider" src={bubble.image} alt="Attached image" />
        )}
        {isUser ? <div className="agave-prose min-w-0">{bubble.text}</div> : <MessageBody text={bubble.text} phase={bubble.phase} onRendered={handleRendered} />}
      </div>
      {bubble.stats && showStats ? <StatsLine stats={bubble.stats} /> : null}
      {canCopy || canRegenerate ? (
        <div className="flex gap-1">
          {canCopy ? <CopyResponse text={bubble.text} /> : null}
          {canRegenerate ? <RegenerateButton retry={failed} onRegenerate={onRegenerate} /> : null}
        </div>
      ) : null}
    </div>
  );
});

export { Message };
