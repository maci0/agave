import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

/** The tailwind-merge package knows only Tailwind's stock scales. The custom steps from
 *  theme.css are registered here, or `text-touch` reads as a color and drops
 *  `text-foreground`, and `rounded-pill` never overrides `rounded-lg`. Keep in
 *  step with the @theme block. */
const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      text: ['2xs', 'touch'],
      radius: ['pill'],
      shadow: ['focus', 'halo', 'halo-focus', 'overlay'],
    },
  },
});

/** Shadcn class merge: conditional classes, then Tailwind conflict resolution. */
export const cn = (...inputs: Array<ClassValue>): string =>
  twMerge(clsx(inputs));
