import { cva, type VariantProps } from 'class-variance-authority';
import type { InputHTMLAttributes, Ref } from 'react';
import { cn } from './cn';

/**
 * Text input. `mono` is for fields that carry numbers or identifiers (token
 * budgets, model URLs) so digits keep a fixed width while the value changes.
 * `lg` is the composer field: base text, lifted to 16px on touch layouts so
 * iOS does not zoom the page into it.
 */
const inputVariants = cva(
  'min-h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-foreground ' +
    'transition outline-none focus:border-primary focus:shadow-focus ' +
    'disabled:pointer-events-none disabled:opacity-60',
  {
    variants: {
      font: {
        sans: '',
        mono: 'font-mono',
      },
      size: {
        default: 'text-sm',
        lg: 'text-base max-drawer:text-touch',
      },
    },
    defaultVariants: { font: 'sans', size: 'default' },
  },
);

type InputProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> & VariantProps<typeof inputVariants> & {
  /** The framework passes `ref` as an ordinary prop, so a plain function
   *  component forwards it without `forwardRef`. */
  ref?: Ref<HTMLInputElement>;
};

export const Input = ({ className, type, font, size, ...props }: InputProps) => (
  <input type={type} className={cn(inputVariants({ font, size }), className)} {...props} />
);
