/** The committed stylesheet, read as text by a test that asserts on a rule the
 *  server serves. A Bun text import (`with { type: 'text' }`) resolves at
 *  runtime; tsc needs the module shape declared. `src/web/style.css` is the
 *  artifact scripts/build-web.sh writes, not a source stylesheet. */
declare module '*.css' {
  const stylesheet: string;
  export default stylesheet;
}

/** CDN globals. marked and DOMPurify are fetched on the first rendered response
 * (`loadMarkdown`), highlight.js on the first fenced code block. */

type MarkedOptions = {
  breaks?: boolean;
  gfm?: boolean;
};

type MarkedStatic = {
  setOptions: (options: MarkedOptions) => void;
  parse: (src: string) => string;
};

declare const marked: MarkedStatic | undefined;

type DOMPurifyConfig = {
  ADD_TAGS?: Array<string>;
};

type DOMPurifyStatic = {
  sanitize: (dirty: string, cfg?: DOMPurifyConfig) => string;
};

declare const DOMPurify: DOMPurifyStatic | undefined;

type HljsLanguage = {
  name?: string;
};

type HljsStatic = {
  highlightElement: (block: HTMLElement) => void;
  getLanguage: (name: string) => HljsLanguage | undefined;
};

declare const hljs: HljsStatic | undefined;
