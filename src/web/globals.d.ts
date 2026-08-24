/** CDN globals injected by `head.html` before the embedded chat script. */

interface MarkedOptions {
  breaks?: boolean;
  gfm?: boolean;
}

interface MarkedStatic {
  setOptions(options: MarkedOptions): void;
  parse(src: string): string;
}

declare const marked: MarkedStatic;

interface DOMPurifyConfig {
  ADD_TAGS?: string[];
}

interface DOMPurifyStatic {
  sanitize(dirty: string, cfg?: DOMPurifyConfig): string;
}

declare const DOMPurify: DOMPurifyStatic | undefined;

interface HljsStatic {
  highlightElement(block: HTMLElement): void;
}

declare const hljs: HljsStatic | undefined;
