import { Range, Root, Thumb, Track } from '@radix-ui/react-slider';
import type { ComponentProps } from 'react';
import { cn } from './cn';

type SliderProps = ComponentProps<typeof Root> & {
  /** Applied to the thumb, where Radix puts the slider role and value text. */
  thumbProps?: ComponentProps<typeof Thumb>;
};

/**
 * Range slider (temperature, top-p).
 *
 * Radix owns the thumb role and keyboard stepping, so each thumb carries its
 * own `aria-label` and `aria-valuetext`; the value text is what a screen reader
 * announces, so the consumer formats a fixed number of digits there.
 */
export const Slider = ({ className, thumbProps, ...props }: SliderProps) => (
  <Root
    className={cn('relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50', className)}
    {...props}
  >
    <Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-border">
      <Range className="absolute h-full bg-primary" />
    </Track>
    <Thumb
      className="block size-5 rounded-full border-2 border-primary bg-background transition-[box-shadow] hover:shadow-halo focus-visible:shadow-halo-focus"
      {...thumbProps}
    />
  </Root>
);
