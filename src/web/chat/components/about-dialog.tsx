import { Dialog, DialogContent, DialogTitle } from '../../ui/dialog';
import { Mark } from '../../ui/icons';
import { cn } from '../../ui/cn';

type AboutDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelName: string;
  backendName: string;
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
 *  hand-rolled modal used to reimplement. */
export const AboutDialog = ({ open, onOpenChange, modelName, backendName }: AboutDialogProps) => (
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
        <Section>Features</Section>
        <ul className="list-disc ps-5 marker:text-success">
          <li className="my-1">Runs locally on CPU or GPU</li>
          <li className="my-1">Opens GGUF and SafeTensors models</li>
          <li className="my-1">Streams responses token by token</li>
        </ul>
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
