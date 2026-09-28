import { Download, Info, Menu, X } from 'lucide-react';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { fmtCtx, fmtInt } from '../format';
import type { ModelRecord } from '../types';

/** Share of the context window at which the counter turns into a warning. */
const CTX_WARN_RATIO = 0.85;

type Context = {
  used: number;
  max: number;
};

type AppHeaderProps = {
  model: ModelRecord | null;
  modelResolved: boolean;
  ctx: Context | null;
  sidebarOpen: boolean;
  /** A turn is in flight: the conversation actions wait for it, as /clear does. */
  streaming: boolean;
  onOpenSidebar: () => void;
  onRetryModel: () => void;
  onNew: () => void;
  onExport: () => void;
  onClear: () => void;
  onAbout: () => void;
};

const ContextBadge = ({ ctx }: { ctx: Context | null }) => {
  if (!ctx || ctx.max <= 0) {return null;}
  const nearFull = ctx.used / ctx.max >= CTX_WARN_RATIO;
  const label = `${nearFull ? '!\u00A0' : ''}${fmtCtx(ctx.used)}/${fmtCtx(ctx.max)}`;
  const described = `${nearFull ? 'Context nearly full' : 'Context'}: ${fmtInt(ctx.used)} of ${fmtInt(ctx.max)} tokens used`;
  return (
    <Badge variant={nearFull ? 'warning' : 'default'} className="ms-2 shrink-0" title={described} aria-label={described}>
      {label}
    </Badge>
  );
};

export const AppHeader = (props: AppHeaderProps) => {
  const { model, modelResolved } = props;
  return (
    <header className="z-10 flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border bg-card px-6 py-3 max-drawer:px-4">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2.5">
        <Button
          type="button"
          size="iconSm"
          className="hidden max-drawer:inline-flex"
          onClick={props.onOpenSidebar}
          title="Menu"
          aria-label={props.sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-expanded={props.sidebarOpen}
          aria-controls="sidebar"
        >
          <Menu className="size-5" aria-hidden="true" />
        </Button>
        <h1 className="inline-flex items-center gap-2 font-mono text-lg font-semibold tracking-tight text-primary">
          <span className="mark" aria-hidden="true" />
          agave
        </h1>
        {modelResolved ? (
          <Badge className="max-w-50" title={model?.id}>
            {model?.id ?? 'unknown'}
          </Badge>
        ) : (
          // Prefer a native button over role="button" on a live region (4.1.2).
          // The label stays honest: it is a control until the server answers.
          <Button
            type="button"
            size="sm"
            variant="primaryOutline"
            aria-label="Offline. Activate to retry connection"
            onClick={props.onRetryModel}
          >
            offline - click to retry
          </Button>
        )}
        <ContextBadge ctx={props.ctx} />
      </div>
      <div className="flex flex-wrap justify-end gap-1.5 ms-auto">
        <Button type="button" variant="primaryOutline" size="sm" onClick={props.onNew} disabled={props.streaming} aria-label="New conversation">
          + New
        </Button>
        <Button type="button" size="sm" onClick={props.onExport} disabled={props.streaming} title="Export conversation" aria-label="Export conversation">
          <span className="max-drawer:hidden">Export </span>
          <Download className="hidden size-4 max-drawer:inline-flex" aria-hidden="true" />
        </Button>
        <Button type="button" size="sm" onClick={props.onClear} disabled={props.streaming} title="Clear conversation" aria-label="Clear conversation">
          <span className="max-drawer:hidden">Clear </span>
          <X className="hidden size-4 max-drawer:inline-flex" aria-hidden="true" />
        </Button>
        <Button type="button" size="sm" onClick={props.onAbout} title="About" aria-label="About Agave">
          <span className="max-drawer:hidden">Info </span>
          <Info className="hidden size-4 max-drawer:inline-flex" aria-hidden="true" />
        </Button>
      </div>
    </header>
  );
};
