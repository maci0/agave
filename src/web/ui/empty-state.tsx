import type { ReactNode } from 'react';
import { Mark } from './icons';

/**
 * The state before the first prompt: the rosette, a mono title, one line of
 * plain copy, and the hint chips. It is the product's front door on both chat
 * surfaces, so it lives here rather than being drawn once per shell.
 */
export const EmptyState = ({ title, line, hints }: { title: string; line: string; hints: ReactNode }) => (
  <div className="m-auto px-5 py-10 text-center">
    <Mark size="lg" />
    <h2 className="mb-2 font-mono text-lg font-semibold text-foreground">{title}</h2>
    <p className="mb-6 text-base text-muted-foreground">{line}</p>
    <div className="flex flex-wrap justify-center gap-2">{hints}</div>
  </div>
);
