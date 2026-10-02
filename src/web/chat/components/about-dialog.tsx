import { Dialog, DialogContent, DialogTitle } from '../../ui/dialog';
import { Mark } from '../../ui/icons';
import { fmtCtx } from '../format';
import { cn } from '../../ui/cn';

type AboutDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelName: string;
  backendName: string;
  /** Live context counters from `GET /v1/models`, absent while offline. */
  ctxSize: number;
  kvUsed: number;
};

/** Inline path or flag in the privacy note, matching `.agave-prose code`. */
const CODE = 'rounded-xs bg-card px-1.5 py-0.5 font-mono text-xs whitespace-nowrap text-primary';

const Row = ({ label, value, chip }: { label: string; value: string; chip?: boolean }) => (
  <div className="flex justify-between gap-4 border-b border-divider py-1.5 text-sm last:border-none">
    <span className="text-faint">{label}</span>
    <span className={cn('text-foreground', chip === true ? 'rounded-xs border border-border bg-card px-1.5 py-0.5 font-mono text-xs' : 'font-mono')}>
      {value}
    </span>
  </div>
);
/** A section heading inside the dialog. Sentence case, per the voice rules in
 *  docs/brand/README.md: the mono face at `xs` already sets these apart from
 *  the dialog title, so capitals and wide tracking would add a second, louder
 *  idea on top of it. */
const Section = ({ children }: { children: string }) => (
  <h3 className="mt-5 mb-1.5 font-mono text-xs font-medium text-primary">{children}</h3>
);

/** The About dialog. Radix owns the focus trap, Escape and the focus restore the
 *  hand-rolled modal used to reimplement.
 *
 *  It states the running configuration rather than a feature list. Three
 *  capability bullets ("runs locally", "opens models", "streams responses")
 *  read the same in every generated app and tell a reader standing at an
 *  OpenAI-compatible endpoint nothing they cannot see; the context size, the
 *  backend and the KV cache in use are the facts worth the space. */
export const AboutDialog = ({ open, onOpenChange, modelName, backendName, ctxSize, kvUsed }: AboutDialogProps) => (
  <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent>
      <DialogTitle className="mb-1">
        <Mark />
        About Agave
      </DialogTitle>
      <div className="text-sm leading-relaxed text-muted-foreground">
        <Section>System</Section>
        <Row label="Model" value={modelName || 'n/a'} />
        <Row label="Backend" value={backendName || 'n/a'} />
        <Row label="API" value="OpenAI-compatible" />
        <Section>Context</Section>
        <Row label="Window" value={ctxSize > 0 ? fmtCtx(ctxSize) : 'unknown'} />
        <Row label="In use" value={kvUsed > 0 ? fmtCtx(kvUsed) : '0'} />
        <Section>Privacy</Section>
        <p>
          Prompts and chats stay on this machine. Conversations are saved to{' '}
          <code className={CODE}>$XDG_CACHE_HOME/agave/conversations.json</code>{' '}
          (or <code className={CODE}>~/.cache/agave/conversations.json</code>) and restored on restart,
          unless the server runs with <code className={CODE}>--no-conv-store</code>, which keeps them in
          memory only. The system prompt lives in this tab&apos;s session storage. No analytics, no telemetry.
        </p>
        <Section>Shortcuts</Section>
        <Row label="Send" value="Enter" chip />
        <Row label="New line" value="Shift+Enter" chip />
        <Row label="Stop / close" value="Escape" chip />
      </div>
    </DialogContent>
  </Dialog>
);
