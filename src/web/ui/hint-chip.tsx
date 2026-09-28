import type { ReactNode } from 'react';
import { cn } from './cn';

/**
 * The hint chips under an empty state: mono, 2xs, faint, one short line each.
 * They state what the surface will and will not do before a turn runs, so the
 * two chat surfaces state it in the same shape rather than each inventing one.
 */
const HINT_CHIP =
  'inline-flex min-h-11 items-center rounded-lg border border-border bg-card px-2.5 py-1 font-mono text-2xs text-faint';

/** A static constraint or affordance: "Enter to send". */
export const HintChip = ({ children }: { children: ReactNode }) => (
  <span className={HINT_CHIP}>{children}</span>
);

/** The one hint that is also a control, so it takes the chip's hover and
 *  underline treatment to say so. */
export const HintAction = ({ children, onClick, label }: { children: ReactNode; onClick: () => void; label: string }) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    className={cn(
      HINT_CHIP,
      'justify-center underline decoration-primary decoration-2 underline-offset-2 transition-colors hover:border-primary hover:text-primary',
    )}
  >
    {children}
  </button>
);
