/** Engine and network causes mapped to short copy that says what to do next.
 *
 *  The browser SDK (`web/agave.ts`) reports a stable `code` plus diagnostic
 *  text; a user needs the first and never the second. The plain-`Error` chains
 *  cover the WASM boundary surfacing a message instead of an `AgaveError`.
 */


/** What the SDK throws: an Error carrying a stable `code` (docs in agave.ts)
 *  and, for downloads, the HTTP status. Matched structurally so this module
 *  does not depend on the class identity across the script boundary. */
type EngineFailure = Error & { code: string; httpStatus?: number };

const isEngineFailure = (cause: unknown): cause is EngineFailure => cause instanceof Error && 'code' in cause;

/** Copy for one SDK error code. Lookup is a parameter type, so an unknown code
 *  falls through to the caller's default copy instead of widening a binding. */
type Copy = Readonly<Record<string, string>>;

const copyFor = (copy: Copy, code: string): string | undefined => copy[code];

const LOAD_COPY = {
  wasm_fetch_failed: 'Could not load the inference engine. Reload the page, or check that agave.wasm is being served.',
  wasm_invalid: 'The inference engine failed to start. Reload the page.',
  gguf_parse: 'This file is not a valid GGUF model.',
  unsupported_arch: 'This model architecture is not supported in the browser.',
  no_vocab: 'This GGUF file has no vocabulary and cannot be used.',
  // oxlint-disable-next-line @rikalabs/no-hardcoded-secrets -- an SDK error code, not a credential
  tokenizer: 'Could not read the tokenizer from this model file.',
  init_failed: 'Could not initialize this model in the browser.',
  alloc_failed: 'The model is too large to fit in this browser.',
  not_initialized: 'The engine is not ready. Reload the page and try again.',
};

const GENERATE_COPY = {
  not_initialized: 'Load a GGUF model first.',
  no_model: 'Load a GGUF model first.',
  alloc_failed: 'Not enough memory to generate a reply. Try a smaller model.',
  invalid_argument: 'Generation settings are out of range. Reload the page and try again.',
  // oxlint-disable-next-line @rikalabs/no-hardcoded-secrets -- an SDK error code, not a credential
  tokenize: 'That message could not be encoded for this model. Try shorter or plain text.',
  wasm_invalid: 'The inference engine failed. Reload the page.',
};

const DOWNLOAD_REFUSED = 'Could not download the model. Check the URL, or drop a GGUF file instead. Some hosts block browser downloads.';

const downloadCopy = (httpStatus: number | undefined): string => {
  if (httpStatus === 404) { return 'The model URL was not found. Check the link.'; }
  if (httpStatus === 401 || httpStatus === 403) { return 'The model URL refused the download. Try dropping a GGUF file instead.'; }
  return DOWNLOAD_REFUSED;
};

/** Match the SDK message text, for the paths that reach the page as a bare Error. */
const fromMessage = (message: string, prefix: string): string => {
  const lower = message.toLowerCase();
  const known: Array<[RegExp, string]> = [
    [/^(?:failed to fetch|load failed)$|networkerror/u, DOWNLOAD_REFUSED],
    [/http 404/u, 'The model URL was not found. Check the link.'],
    [/http (?:401|403)/u, 'The model URL refused the download. Try dropping a GGUF file instead.'],
    [/^(?:gguf parse error|.*not a valid gguf)/u, 'This file is not a valid GGUF model.'],
    [/^unsupported arch/u, 'This model architecture is not supported in the browser.'],
    [/^no vocab/u, 'This GGUF file has no vocabulary and cannot be used.'],
    [/^tok error/u, 'Could not read the tokenizer from this model file.'],
    [/^(?:model init error|failed to initialize model)/u, 'Could not initialize this model in the browser.'],
    [/failed to allocate|out of memory/u, 'The model is too large to fit in this browser.'],
    [/engine not initialized/u, 'The engine is not ready. Reload the page and try again.'],
  ];
  for (const [pattern, text] of known) {
    if (pattern.test(lower)) { return text; }
  }
  return message.startsWith('Could not') ? message : `${prefix}: ${message}`;
};

/** Map a model-load cause to copy a reader can act on. */
export const friendlyLoadError = (cause: unknown): string => {
  if (isEngineFailure(cause)) {
    if (cause.code === 'download_failed') { return downloadCopy(cause.httpStatus); }
    return copyFor(LOAD_COPY, cause.code) ?? `Could not load model: ${cause.message}`;
  }
  const message = cause instanceof Error ? cause.message : String(cause);
  return fromMessage(message, 'Could not load model');
};

/** Map a generation cause to copy, mirroring friendlyLoadError. */
export const friendlyGenerateError = (cause: unknown): string => {
  if (isEngineFailure(cause)) {
    return copyFor(GENERATE_COPY, cause.code) ?? `Could not generate a reply: ${cause.message}`;
  }
  const message = cause instanceof Error ? cause.message : String(cause);
  const lower = message.toLowerCase();
  if (lower === 'no model loaded' || lower === 'model not initialized') { return 'Load a GGUF model first.'; }
  if (lower.includes('out of memory') || lower.includes('failed to allocate')) {
    return 'Not enough memory to generate a reply. Try a smaller model.';
  }
  if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
    return 'The connection dropped while generating. Try again.';
  }
  return message.startsWith('Could not') ? message : `Could not generate a reply: ${message}`;
};
