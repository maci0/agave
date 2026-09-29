/**
 * The icon set. Every glyph either chat surface draws comes from here, named
 * for the action it stands for rather than its picture, so one action keeps one
 * glyph everywhere (oxlint `no-restricted-imports` keeps `lucide-react` out of
 * every other file). docs/brand/README.md lists the set with its rules: 24px
 * grid, 2px stroke, drawn at size-4 in text and size-5 on icon keys, always
 * `aria-hidden` beside a text or `aria-label`.
 */
import { cva, type VariantProps } from 'class-variance-authority';

export {
  ArrowDown as JumpToLatestIcon,
  Copy as CopyIcon,
  Download as ExportIcon,
  Eraser as ClearIcon,
  Image as AttachImageIcon,
  Info as AboutIcon,
  Menu as MenuIcon,
  Plus as NewIcon,
  RefreshCw as RegenerateIcon,
  SlidersHorizontal as SettingsIcon,
  Trash as DeleteIcon,
  X as CloseIcon,
} from 'lucide-react';

const markVariants = cva('mark', {
  variants: {
    size: {
      sm: 'mark-sm',
      md: '',
      lg: 'mark-lg',
    },
  },
  defaultVariants: { size: 'md' },
});

/** The agave rosette in rosette green, decorative: the text beside it names
 *  the product. `sm` labels an assistant turn, `md` sits in headers, `lg`
 *  anchors an empty state. */
export const Mark = ({ size }: VariantProps<typeof markVariants>) => (
  <span className={markVariants({ size })} aria-hidden="true" />
);
