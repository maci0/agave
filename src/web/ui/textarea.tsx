import type { TextareaHTMLAttributes } from 'react';
import { cn } from './utils';

/** Multiline input: system prompt, message composer. */
export function Textarea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground',
        'transition-[border-color,box-shadow] outline-none placeholder:text-faint',
        'focus:border-primary focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-primary)_10%,transparent)]',
        'disabled:pointer-events-none disabled:opacity-50',
        className,
      )}
      {...props}
    />
  );
}
