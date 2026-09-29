import { AboutIcon, ClearIcon, ExportIcon, Mark, MenuIcon, NewIcon } from '../../ui/icons';
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
    // A span has the generic role, which takes no accessible name, so the
    // Label this badge used to carry was dropped: the counter reached a
    // Screen reader as a bare ratio, with the near-full warning left to a "!"
    // Glyph and a color. The sentence is real text instead, and the numbers it
    // Repeats are hidden so the badge is not read twice.
    <Badge variant={nearFull ? 'warning' : 'default'} className="shrink-0" title={described}>
      <span aria-hidden="true">{label}</span>
      <span className="sr-only">{described}</span>
    </Badge>
  );
};

/** Conversation actions. Icons carry the meaning on narrow layouts, where the
 *  labels drop to keep the header on one row. */
const HeaderActions = (props: AppHeaderProps) => (
  <div className="flex flex-wrap justify-end gap-1 ms-auto">
    {/* The sidebar carries its own New key; this one covers the drawer
        layout, where the sidebar is closed. */}
    <Button
      type="button"
      variant="primaryOutline"
      size="sm"
      className="hidden max-drawer:inline-flex"
      onClick={props.onNew}
      disabled={props.streaming}
      aria-label="New conversation"
    >
      <NewIcon className="size-4" aria-hidden="true" />
      <span className="max-drawer:sr-only">New</span>
    </Button>
    <Button type="button" variant="plain" size="sm" onClick={props.onExport} disabled={props.streaming} title="Export conversation" aria-label="Export conversation">
      <ExportIcon className="size-4" aria-hidden="true" />
      <span className="max-drawer:hidden">Export</span>
    </Button>
    <Button type="button" variant="plain" size="sm" onClick={props.onClear} disabled={props.streaming} title="Clear conversation" aria-label="Clear conversation">
      <ClearIcon className="size-4" aria-hidden="true" />
      <span className="max-drawer:hidden">Clear</span>
    </Button>
    <Button type="button" variant="plain" size="sm" onClick={props.onAbout} title="About" aria-label="About Agave">
      <AboutIcon className="size-4" aria-hidden="true" />
      <span className="max-drawer:hidden">Info</span>
    </Button>
  </div>
);

export const AppHeader = (props: AppHeaderProps) => {
  const { model, modelResolved } = props;
  return (
    <header className="z-10 flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-divider bg-card px-6 py-3 max-drawer:px-4">
      <div className="flex items-center gap-2.5">
        <Button
          type="button"
          variant="plain"
          size="iconSm"
          className="hidden max-drawer:inline-flex"
          onClick={props.onOpenSidebar}
          title="Menu"
          aria-label={props.sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-expanded={props.sidebarOpen}
          aria-controls="sidebar"
        >
          <MenuIcon className="size-5" aria-hidden="true" />
        </Button>
        <h1 className="inline-flex items-center gap-2 font-mono text-lg font-semibold tracking-tight text-primary">
          <Mark />
          agave
        </h1>
      </div>
      {/* Model and context: beside the name on wide layouts, on their own row
          under it in the drawer layout so the actions keep the first row. */}
      <div className="flex min-w-0 flex-1 items-center gap-2 max-drawer:order-last max-drawer:basis-full">
        {modelResolved ? (
          <Badge className="max-w-72 min-w-0" title={model?.id}>
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
      <HeaderActions {...props} />
    </header>
  );
};
