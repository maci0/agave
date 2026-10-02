import type { ReactNode } from 'react';
import { Mark } from './icons';

/**
 * The state before the first prompt: the rosette, a mono title, one line of
 * plain copy, and the hint chips. It is the product's front door on both chat
 * surfaces, so it lives here rather than being drawn once per shell.
 *
 * Inline-start aligned, not centered. A centered icon over a centered headline
 * over a centered subhead over a centered chip row is the shape every
 * generated app opens on, and it is the one shape that carries nothing of the
 * product. What identifies agave is that both surfaces are instrument panels
 * held to one reading measure (`.agave-measure`), so the resting state sits on
 * the same left edge and the same measure as the transcript it will be
 * replaced by. The prompt will arrive at that edge too, and it moves less.
 */
export const EmptyState = ({ title, line, hints }: { title: string; line: string; hints: ReactNode }) => (
  <div className="m-auto w-full agave-measure px-5 py-10 text-start">
    <Mark size="lg" />
    <h2 className="mb-2 font-mono text-lg font-semibold text-foreground">{title}</h2>
    <p className="mb-6 text-base text-muted-foreground">{line}</p>
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">{hints}</div>
  </div>
);
