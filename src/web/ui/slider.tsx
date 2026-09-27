import * as SliderPrimitive from '@radix-ui/react-slider';
import type { ComponentProps } from 'react';
import { cn } from './utils';

/**
 * Range slider (temperature, top-p).
 *
 * Radix owns the thumb role and keyboard stepping, so each thumb carries its
 * own `aria-label` and `aria-valuetext`; the value text is what a screen reader
 * announces, so the consumer formats a fixed number of digits there.
 */
export const Slider = ({ className, thumbProps, ...props }: ComponentProps<typeof SliderPrimitive.Root> & {
  /** Applied to the thumb, where Radix puts the slider role and value text. */
  thumbProps?: ComponentProps<typeof SliderPrimitive.Thumb>;
}) => {
  return (
    <SliderPrimitive.Root
      className={cn(
        'relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-border">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        className="block size-5 rounded-full border-2 border-primary bg-background transition-[box-shadow] hover:shadow-[0_0_0_4px_color-mix(in_oklab,var(--color-primary)_15%,transparent)] focus-visible:shadow-[0_0_0_4px_color-mix(in_oklab,var(--color-primary)_25%,transparent)]"
        {...thumbProps}
      />
    </SliderPrimitive.Root>
  );
};
