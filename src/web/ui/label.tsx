import { Root as LabelRoot } from '@radix-ui/react-label';
import type { ComponentProps } from 'react';
import { cn } from './cn';

/** Form label. Chrome labels are mono at the 2xs/xs steps, never the prose size. */
export const Label = ({ className, ...props }: ComponentProps<typeof LabelRoot>) => (
  <LabelRoot
    className={cn('font-mono text-xs leading-none text-faint select-none', className)}
    {...props}
  />
);
