/** CDN globals. marked and DOMPurify load with `defer` in `head.html`.
 * highlight.js is fetched on the first fenced code block. */

type MarkedOptions = {
  breaks?: boolean;
  gfm?: boolean;
};

type MarkedStatic = {
  setOptions(options: MarkedOptions): void;
  parse(src: string): string;
};

declare const marked: MarkedStatic | undefined;

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
