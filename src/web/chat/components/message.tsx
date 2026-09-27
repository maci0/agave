import { Copy, RefreshCw } from 'lucide-react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { copyText, markdownReady, onIdle, renderMarkdown } from '../markdown';
import { fmtInt, fmtNum, truncateAnnounce } from '../format';
import type { Bubble, StreamStats } from '../types';
import { cn } from '../../ui/utils';

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
  onRendered?: (rendered: string) => void;
};

/**
 * The body of an assistant turn.
 *
 * React owns the element, not its children: the streaming path appends text
 * nodes and the final path fills the same node from sanitized markdown, so a
 * turn that goes from hundreds of partial tokens to a full markdown render
 * costs one reconcile rather than a re-render per token.
 */
export const MessageBody = memo(function MessageBody({ text, phase, onRendered }: MessageBodyProps) {
  const ref = useRef<HTMLDivElement>(null);
  const painted = useRef('');
  const announced = useRef<string | null>(null);

  useEffect(function () {
    const element = ref.current;
    if (!element) {return;}
    if (phase === 'streaming') {
      painted.current = appendDelta(element, painted.current, text);
      return;
    }
    if (phase === 'thinking') {
      element.textContent = THINKING_GLYPH;
      painted.current = '';
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
      onRendered?.(truncateAnnounce(element.textContent ?? '', 200));
    }
    if (!markdownReady()) {
      // The libraries were still in flight. Rebuild once they land, one message
      // per idle slot, so a restored history does not re-render in one task.
      onIdle(function () {
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
  const total = Number.parseInt(stats.time, 10) + (Number.parseInt(stats.pfMs, 10) || 0);
  const decode = `${fmtInt(stats.tokens)} tok @ ${fmtNum(Number.parseFloat(stats.tps), 2)}`;
  const prefill = `${fmtInt(stats.pfTok)} tok @ ${fmtNum(Number.parseFloat(stats.pfTps), 1)}`;
  return (
    <div className="mt-2.5 flex flex-wrap gap-x-4 border-t border-border pt-2.5 font-mono text-2xs text-faint">
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

const Message = memo(function Message({ bubble, showStats, canRegenerate, onRegenerate, onRendered }: MessageProps) {
  const [copyLabel, setCopyLabel] = useState('Copy');
  const isUser = bubble.role === 'user';
  const roleId = `msg-role-${bubble.id}`;
  const failed = bubble.phase === 'error';
  const regenLabel = failed ? 'Retry' : 'Regenerate';

  const handleRendered = useCallback((rendered: string) => { onRendered(bubble.id, rendered); }, [bubble.id, onRendered]);
  const handleCopy = useCallback(function () {
    void copyText(bubble.text).then(function (result) {
      setCopyLabel(result === 'copied' ? 'Copied' : 'Failed');
      setTimeout(function () { setCopyLabel('Copy'); }, COPY_REVERT_MS);
    });
  }, [bubble.text]);

  return (
    <div
      role="group"
      aria-labelledby={roleId}
      className={cn(
        'mx-auto flex w-full max-w-prose flex-col gap-1',
        isUser ? 'items-end' : 'items-start',
      )}
    >
      <span id={roleId} className={cn('px-1 font-mono text-xs font-medium', isUser ? 'text-faint' : 'text-primary')}>
        {isUser ? 'You' : 'agave'}
      </span>
      <div
        dir="auto"
        role={failed ? 'alert' : undefined}
        className={cn(
          'relative w-full min-w-0 max-w-full rounded-lg border px-[18px] py-3.5 text-base',
          failed ? 'border-destructive bg-destructive/10 text-sm text-destructive-foreground' : '',
          !isUser && !failed ? 'rounded-es-[2px] border-border bg-popover' : '',
          isUser ? 'rounded-ee-[2px] border-border bg-card' : '',
        )}
      >
        {bubble.image ? (
          <img className="mb-2 block max-w-[200px] rounded-lg border border-border" src={bubble.image} alt="Attached image" />
        ) : null}
        {isUser ? <div className="agave-prose min-w-0">{bubble.text}</div> : <MessageBody text={bubble.text} phase={bubble.phase} onRendered={handleRendered} />}
        {!isUser && !failed && bubble.phase === 'done' ? (
          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy response"
            className="agave-reveal absolute end-2 top-2 inline-flex min-h-11 min-w-11 items-center justify-center rounded-md p-1 font-mono text-2xs text-faint transition-colors hover:text-primary max-drawer:static max-drawer:mt-2 max-drawer:px-3"
          >
            {copyLabel === 'Copy' ? <Copy className="size-3.5" aria-hidden="true" /> : copyLabel}
          </button>
        ) : null}
      </div>
      {bubble.stats && showStats ? <StatsLine stats={bubble.stats} /> : null}
      {canRegenerate ? (
        <button
          type="button"
          onClick={onRegenerate}
          aria-label={failed ? 'Retry generating response' : 'Regenerate response'}
          className="agave-reveal inline-flex min-h-11 items-center gap-1.5 rounded-md border border-border px-2.5 py-1 font-mono text-2xs text-faint transition-colors hover:border-primary hover:bg-primary/10 hover:text-primary"
        >
          <RefreshCw className="size-3.5" aria-hidden="true" />
          {regenLabel}
        </button>
      ) : null}
    </div>
  );
});

export { Message };
