import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Shadcn class merge: conditional classes, then Tailwind conflict resolution. */
export const cn = (...inputs: Array<ClassValue>): string =>
  twMerge(clsx(inputs));
