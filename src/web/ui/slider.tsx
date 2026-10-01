import type { InputHTMLAttributes } from 'react';
import { cn } from './cn';

type SliderProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'value' | 'onChange'> & {
  value: number;
  /** Called with the new value on every drag step and arrow key. */
  onValueChange: (next: number) => void;
};

/**
 * Range slider (temperature, top-p).
 *
 * This is a native `<input type="range">`, not the Radix primitive shadcn wraps.
 * Radix's Slider positions its thumb from `context.values[index]`, and `index`
 * comes from its Collection; that registration does not reach Radix through
 * `preact/compat`, so the thumb stayed at `left: calc(0% + 0px)` while the value
 * changed underneath it, and `aria-valuenow` never rendered. The native control
 * carries the same semantics (`role=slider`, value, min, max, keyboard stepping)
 * with no shim, and `accent-color` plus the thumb variants keep the agave look.
 *
 * `aria-valuetext` is what a screen reader announces, so callers format the
 * value to the precision the setting actually has ("0.0", "1.00").
 *
 * The track and thumb take `rounded-pill`, the token on the radius ramp, not
 * Tailwind's `rounded-full`. The two are the same 999px today, but `--radius-pill`
 * is the one the brand guide documents, so shrinking or squaring the ramp moves
 * the slider with it.
 */
export const Slider = ({ className, value, onValueChange, ...props }: SliderProps) => (
  <input
    type="range"
    value={value}
    onChange={function (event) { onValueChange(Number(event.target.value)); }}
    className={cn(
      'h-1.5 w-full cursor-pointer appearance-none rounded-pill bg-border accent-primary',
      '[&::-webkit-slider-thumb]:size-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-pill [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary [&::-webkit-slider-thumb]:bg-background',
      '[&::-moz-range-thumb]:size-5 [&::-moz-range-thumb]:rounded-pill [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-primary [&::-moz-range-thumb]:bg-background',
      className,
    )}
    {...props}
  />
);
