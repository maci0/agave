/**
 * "Skip to message input": off-screen until focused, then pinned to the top
 * inline-start corner above everything else. Both chat surfaces render it first
 * in the tab order.
 */
export const SkipLink = ({ href, children }: { href: string; children: string }) => (
  <a
    href={href}
    className={
      'sr-only z-100 rounded-md bg-primary px-4 py-2 font-mono text-sm font-medium text-primary-foreground ' +
      'focus:not-sr-only focus:fixed focus:start-4 focus:top-4'
    }
  >
    {children}
  </a>
);
