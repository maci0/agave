import * as LabelPrimitive from '@radix-ui/react-label';
import type { ComponentProps } from 'react';
import { cn } from './utils';

/** Form label. Chrome labels are mono at the 2xs/xs steps, never the prose size. */
export const Label = ({ className, ...props }: ComponentProps<typeof LabelPrimitive.Root>) => (
  <LabelPrimitive.Root
    className={cn('font-mono text-xs leading-none text-faint select-none', className)}
    {...props}
  />
);