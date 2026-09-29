import { cva, type VariantProps } from 'class-variance-authority';
import type { ButtonHTMLAttributes } from 'react';
import { cn } from './cn';

/**
 * Button.
 *
 * Every size keeps a 44px minimum target (WCAG 2.5.8) and the mono face, so the
 * controls read as instrument-panel chrome. `primaryOutline` is the agave
 * action style: an amber outline over a 10% amber wash that fills solid on
 * hover; `ghost` is bordered chrome and `plain` the borderless toolbar key;
 * `solid` is the send key; `floating` sits over the transcript.
 */
const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-lg border font-mono font-medium whitespace-nowrap ' +
    'transition-colors duration-200 disabled:pointer-events-none disabled:opacity-40 ' +
    'aria-invalid:border-destructive',
  {
    variants: {
      variant: {
        solid: 'border-transparent bg-primary text-primary-foreground hover:bg-primary/85',
        primaryOutline:
          'border-primary bg-primary/10 text-primary hover:bg-primary hover:text-primary-foreground',
        ghost: 'border-border bg-transparent text-faint hover:border-border-strong hover:bg-muted hover:text-foreground',
        destructive:
          'border-destructive bg-destructive/10 text-destructive hover:bg-destructive hover:text-background',
        plain: 'border-transparent bg-transparent text-faint hover:bg-muted hover:text-foreground',
        plainDestructive:
          'border-transparent bg-transparent text-faint hover:bg-destructive/10 hover:text-destructive-foreground',
        /* The glyph is --background on --destructive (5.98:1) and on the lighter
           hover red (8.47:1); white measured 2.99:1, below 1.4.3. */
        destructiveSolid: 'border-transparent bg-destructive text-background hover:bg-destructive-foreground',
        floating: 'border-border-strong bg-popover text-foreground shadow-overlay hover:bg-muted',
      },
      size: {
        xs: 'min-h-11 px-2 py-1 text-2xs',
        sm: 'min-h-11 px-3 py-1.5 text-xs',
        default: 'min-h-11 px-3 py-1.5 text-xs',
        lg: 'min-h-11 px-5 py-3 text-sm',
        icon: 'size-12 p-3 text-lg',
        iconSm: 'size-11 p-2.5 text-base',
        iconRound: 'size-11 rounded-pill p-2.5 text-base',
      },
      active: {
        true: 'border-primary bg-primary/10 text-primary',
        false: '',
      },
    },
    defaultVariants: {
      variant: 'ghost',
      size: 'default',
      active: false,
    },
  },
);

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants>;

/** A `<button>`, always. shadcn's `asChild` escape hatch came with the Radix
 *  Slot primitive; nothing in either surface rendered anything but a button, so
 *  the primitive and the prop went with it. */
export const Button = ({ className, variant, size, active, ...props }: ButtonProps) => (
  <button className={cn(buttonVariants({ variant, size, active }), className)} {...props} />
);

export { buttonVariants };
