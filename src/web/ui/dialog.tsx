import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import type { ComponentProps } from 'react';
import { cn } from './utils';

/**
 * Modal dialog.
 *
 * Radix supplies the focus trap, the `aria-modal` wiring, Escape handling and
 * the focus restore that the hand-rolled modal reimplemented. `side="left"`
 * turns the same primitive into the mobile conversation drawer, which is the
 * one place this surface uses a sheet instead of a centered card.
 */
export function Dialog(props: ComponentProps<typeof DialogPrimitive.Root>) {
  return <DialogPrimitive.Root data-slot="dialog" {...props} />;
}

export function DialogTrigger(props: ComponentProps<typeof DialogPrimitive.Trigger>) {
  return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />;
}

export function DialogClose(props: ComponentProps<typeof DialogPrimitive.Close>) {
  return <DialogPrimitive.Close data-slot="dialog-close" {...props} />;
}

type DialogContentProps = ComponentProps<typeof DialogPrimitive.Content> & {
  /** `left` renders a drawer pinned to the inline start edge. */
  side?: 'center' | 'left';
  /** Hide the built-in close button (the About dialog keeps its own). */
  hideClose?: boolean;
};

const overlayClass =
  'fixed inset-0 z-40 bg-black/60 transition-opacity duration-200 data-[state=closed]:opacity-0 data-[state=open]:opacity-100';

export function DialogContent({ className, children, side = 'center', hideClose, ...props }: DialogContentProps) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className={overlayClass} />
      <DialogPrimitive.Content
        className={cn(
          'fixed z-50 border border-border bg-popover text-popover-foreground shadow-[0_8px_20px_rgb(0_0_0/0.45)]',
          side === 'center'
            ? 'inset-1/2 w-[min(480px,90vw)] max-h-[85dvh] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg p-7'
            : 'inset-y-0 start-0 flex w-(--spacing-sidebar) max-w-[85vw] flex-col border-e',
          className,
        )}
        {...props}
      >
        {children}
        {hideClose ? null : (
          <DialogPrimitive.Close
            className="absolute end-4 top-4 inline-flex size-11 items-center justify-center rounded-sm text-faint transition-colors hover:bg-card hover:text-foreground"
            aria-label="Close dialog"
          >
            <X className="size-5" aria-hidden="true" />
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export function DialogTitle({ className, ...props }: ComponentProps<typeof DialogPrimitive.Title>) {
  return (
    <DialogPrimitive.Title
      className={cn('font-mono text-lg font-semibold text-primary', className)}
      {...props}
    />
  );
}

export function DialogDescription({
  className,
  ...props
}: ComponentProps<typeof DialogPrimitive.Description>) {
  return <DialogPrimitive.Description className={cn('text-sm text-muted-foreground', className)} {...props} />;
}
