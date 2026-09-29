import type { TextareaHTMLAttributes } from 'react';
import { cn } from './cn';

/** Multiline input: system prompt, message composer. */
export const Textarea = ({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) => (
  <textarea
    className={cn(
      'w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground',
      'transition outline-none',
      'focus:border-primary focus:shadow-focus',
      'disabled:pointer-events-none disabled:opacity-50',
      className,
    )}
    {...props}
  />
);