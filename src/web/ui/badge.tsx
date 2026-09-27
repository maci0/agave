import { cva, type VariantProps } from 'class-variance-authority';
import type { HTMLAttributes } from 'react';
import { cn } from './utils';

/** Badge: the header model name, the context-window counter, status pills. */
const badgeVariants = cva(
  'inline-flex max-w-50 items-center truncate rounded-lg border px-2.5 py-1 font-mono text-xs',
  {
    variants: {
      variant: {
        default: 'border-border bg-popover text-muted-foreground',
        warning: 'border-destructive bg-destructive/10 text-destructive-foreground',
        primary: 'border-primary bg-primary/10 text-primary',
        invisible: 'hidden',
      },
    },
    defaultVariants: { variant: 'default' },
  },
);

type BadgeProps = HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>;

/** Span badge. Pass a `title` and `aria-label` when the text is truncated. */
export const Badge = ({ className, variant, ...props }: BadgeProps) =>
  <span className={cn(badgeVariants({ variant }), className)} {...props} />;

export { badgeVariants };
