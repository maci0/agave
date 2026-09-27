import type { InputHTMLAttributes, Ref } from 'react';
import { cn } from './utils';

type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  /** React 19 passes `ref` as an ordinary prop, so a plain function component
   *  can forward it without `forwardRef`. */
  ref?: Ref<HTMLInputElement>;
};

/**
 * Text input. `mono` is for fields that carry numbers or identifiers (token
 * budgets, model URLs) so digits keep a fixed width while the value changes.
 */
export const Input = ({ className, type, ...props }: InputProps) => {
  return (
    <input
      type={type}
      className={cn(
        'min-h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground',
        'transition-[border-color,box-shadow] outline-none placeholder:text-faint',
        'focus:border-primary focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-primary)_10%,transparent)]',
        'disabled:pointer-events-none disabled:opacity-60',
        className,
      )}
      {...props}
    />
  );
};
