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
const CODE = 'rounded-xs bg-card px-1.5 py-0.5 font-mono text-xs text-primary';

const Row = ({ label, value, chip }: { label: string; value: string; chip?: boolean }) => (
  <div className="flex justify-between gap-4 border-b border-divider py-1.5 text-sm last:border-none">
    <span className="text-faint">{label}</span>
    <span className={cn('text-foreground', chip === true ? 'rounded-xs border border-border bg-card px-1.5 py-0.5 font-mono text-xs' : 'font-mono')}>
      {value}
    </span>
  </div>
);
/** The About dialog. Radix owns the focus trap, Escape and the focus restore the
 *  hand-rolled modal used to reimplement. */
export const AboutDialog = ({ open, onOpenChange, modelName, backendName }: AboutDialogProps) => (
  <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent>
      <DialogTitle className="mb-5">
        <Mark />
        About Agave
      </DialogTitle>
      <div className="leading-relaxed text-muted-foreground">
        <p>
          <strong className="text-foreground">Agave LLM Inference Engine</strong>
        </p>
        <h3 className="my-2 font-mono text-sm text-primary">System</h3>
        <Row label="Model" value={modelName || '-'} />
        <Row label="Backend" value={backendName || '-'} />
        <Row label="API" value="OpenAI-compatible" />
        <h3 className="my-2 font-mono text-sm text-primary">Features</h3>
        <ul className="my-1 ps-5">
          <li className="my-1 text-sm">Runs locally on CPU or GPU</li>
          <li className="my-1 text-sm">Open GGUF and SafeTensors models</li>
          <li className="my-1 text-sm">Responses stream token by token</li>
          <li className="my-1 text-sm">Chats stay on this machine</li>
        </ul>
        <h3 className="my-2 font-mono text-sm text-primary">Privacy</h3>
        <p className="text-sm">
          Prompts and chats stay on this machine. Conversations are saved to a local file (
          <code className={CODE}>$XDG_CACHE_HOME/agave/conversations.json</code>
          , or <code className={CODE}>~/.cache/agave/conversations.json</code>
          ) and are restored on restart; they stay in memory only when the server is started with{' '}
          <code className={CODE}>--no-conv-store</code>. The system prompt is kept
          in session storage for this browser tab and is not sent to third parties. No analytics or telemetry.
        </p>
        <h3 className="my-2 font-mono text-sm text-primary">Shortcuts</h3>
        <Row label="Send" value="Enter" chip />
        <Row label="New line" value="Shift+Enter" chip />
        <Row label="Stop / close" value="Escape" chip />
      </div>
    </DialogContent>
  </Dialog>
);