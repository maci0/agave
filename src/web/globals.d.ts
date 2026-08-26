/** CDN globals injected by `head.html` before the embedded chat script. */

type MarkedOptions = {
  breaks?: boolean;
  gfm?: boolean;
};

type MarkedStatic = {
  setOptions(options: MarkedOptions): void;
  parse(src: string): string;
};

declare const marked: MarkedStatic;

type DOMPurifyConfig = {
  ADD_TAGS?: Array<string>;
};

type DOMPurifyStatic = {
  sanitize(dirty: string, cfg?: DOMPurifyConfig): string;
};

declare const DOMPurify: DOMPurifyStatic | undefined;

type HljsStatic = {
  highlightElement(block: HTMLElement): void;
};

declare const hljs: HljsStatic | undefined;
