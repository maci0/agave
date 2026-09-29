import type { ReactNode } from 'react';

/**
 * The hints under an empty state: mono, 2xs, one short line each. They state
 * what the surface will and will not do before a turn runs, so the two chat
 * surfaces state it in the same shape rather than each inventing one.
 *
 * A static hint is plain text with a leading dot, so it cannot be mistaken for
 * a control; only `HintAction` wears a frame, because only it can be pressed.
 */
export const HintChip = ({ children }: { children: ReactNode }) => (
  <span className="inline-flex min-h-11 items-center gap-2 px-2 font-mono text-2xs text-faint before:size-1 before:rounded-pill before:bg-success before:content-['']">
    {children}
  </span>
);

/** The one hint that is also a control. */
export const HintAction = ({ children, onClick, label }: { children: ReactNode; onClick: () => void; label: string }) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    className="inline-flex min-h-11 items-center justify-center rounded-lg border border-border bg-card px-3 py-1 font-mono text-2xs text-muted-foreground transition-colors hover:border-primary hover:text-primary"
  >
    {children}
  </button>
);
