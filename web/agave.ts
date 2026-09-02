/* oxlint-disable @rikalabs/no-standalone-classes -- stateful engine class; public API is constructor-based per web/index.html */
/**
 * Agave WASM browser inference glue (`web/`), not the HTTP chat UI.
 *
 * The server chat UI lives in `src/web/` and is embedded by `server.zig`.
 * This file pairs with `web/index.html` and `agave.wasm` for in-browser runs.
 *
 * Usage:
 *   const agave = new AgaveEngine();
 *   await agave.init();
 *   await agave.loadModel('https://example.com/model.gguf');
 *   try {
 *     const output = await agave.generate('What is 2+2?', { maxTokens: 100 });
 *   } catch (e) {
 *     if (e instanceof AgaveError && e.code === 'no_model') {
 *       // load a model, then retry
 *     }
 *     throw e;
 *   }
 */

type AgaveWasmExports = {
  memory: WebAssembly.Memory;
  agave_alloc(len: number): number;
  agave_dealloc(ptr: number, len: number): void;
  agave_init(ptr: number, len: number): number;
  agave_get_output(ctx: number, buf: number, len: number): number;
  agave_generate(ctx: number, promptPtr: number, promptLen: number, maxTokens: number): number;
  agave_last_error(ctx: number): number;
  agave_free(ctx: number): void;
};

type GenerateOptions = {
  maxTokens?: number;
};

type ModelSource = string | ArrayBuffer | ArrayBufferView;

/** Codes a caller can switch on without matching `Error.message`. */
type AgaveErrorCode =
  | 'not_initialized'
  | 'no_model'
  | 'alloc_failed'
  | 'wasm_invalid'
  | 'wasm_fetch_failed'
  | 'download_failed'
  | 'gguf_parse'
  | 'unsupported_arch'
  | 'no_vocab'
  | 'tokenizer'
  | 'init_failed'
  | 'generate_failed';

/**
 * Recoverable engine failure. `code` is stable; `message` is diagnostic text.
 * `httpStatus` is set when a fetch failed with an HTTP status.
 */
class AgaveError extends Error {
  readonly code: AgaveErrorCode;
  readonly httpStatus: number | undefined;

  constructor(code: AgaveErrorCode, message: string, httpStatus?: number) {
    super(message);
    this.name = 'AgaveError';
    this.code = code;
    this.httpStatus = httpStatus;
  }
}

/** Token budget used when the caller omits one or passes zero. */
const default_max_tokens = 100;

/** Scratch buffer for `agave_get_output` (matches `max_output_bytes` in wasm_entry). */
const output_buf_size = 16_384;

/** Default `init()` fetch when the caller does not pass a module URL or bytes. */
const default_wasm_url = 'agave.wasm';

/** Numeric values of `WasmError` in `src/wasm_entry.zig`. */
const wasm_err = {
  ok: 0,
  not_ready: 1,
  tokenize: 2,
  gguf_parse: 3,
  unsupported_arch: 4,
  no_vocab: 5,
  tokenizer: 6,
  model_init: 7,
  invalid_handle: 8,
} as const;

/** Every function export this glue calls, checked once at instantiation. */
const required_exports = [
  'agave_alloc',
  'agave_dealloc',
  'agave_init',
  'agave_get_output',
  'agave_generate',
  'agave_last_error',
  'agave_free',
] as const;

/**
 * Narrow a freshly instantiated module's exports to the shape this file calls.
 *
 * `WebAssembly.Instance.exports` is an untyped record, so this is a trust
 * boundary: a mismatched agave.wasm otherwise fails much later as "not a
 * function" inside a generate call. Throws naming the missing export instead.
 */
const wasmExports = (instance: WebAssembly.Instance): AgaveWasmExports => {
  const { exports } = instance;
  if (!(exports.memory instanceof WebAssembly.Memory)) {
    throw new AgaveError('wasm_invalid', 'agave.wasm: missing or invalid memory export');
  }
  for (const name of required_exports) {
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- boundary check on an untyped wasm export record
    if (typeof exports[name] !== 'function') {
      throw new AgaveError('wasm_invalid', `agave.wasm: missing export ${name}`);
    }
  }
  // SAFETY: Memory and every entry of required_exports were verified just above.
  // Together they are the whole of AgaveWasmExports.
  // oxlint-disable-next-line typescript-eslint/no-unsafe-type-assertion -- narrowing after the checks above
  return exports as AgaveWasmExports;
};

const bytesFromBuffer = (source: ArrayBuffer | ArrayBufferView): Uint8Array => {
  if (source instanceof ArrayBuffer) {
    return new Uint8Array(source);
  }
  return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
};

const readCtxOutput = (exp: AgaveWasmExports, ctx: number): string => {
  const outPtr = exp.agave_alloc(output_buf_size);
  if (outPtr === 0) {
    throw new AgaveError('alloc_failed', 'Failed to allocate WASM memory for output');
  }
  try {
    const outLen = exp.agave_get_output(ctx, outPtr, output_buf_size);
    return new TextDecoder().decode(new Uint8Array(exp.memory.buffer, outPtr, outLen));
  } finally {
    exp.agave_dealloc(outPtr, output_buf_size);
  }
};

const initErrorCode = (wasm_code: number): AgaveErrorCode => {
  switch (wasm_code) {
    case wasm_err.gguf_parse:
      return 'gguf_parse';
    case wasm_err.unsupported_arch:
      return 'unsupported_arch';
    case wasm_err.no_vocab:
      return 'no_vocab';
    case wasm_err.tokenizer:
      return 'tokenizer';
    default:
      return 'init_failed';
  }
};

const generateErrorCode = (wasm_code: number): AgaveErrorCode => {
  if (wasm_code === wasm_err.not_ready || wasm_code === wasm_err.invalid_handle) {
    return 'no_model';
  }
  return 'generate_failed';
};

class AgaveEngine {
  wasm: WebAssembly.Instance | null = null;
  ctx = 0;
  ready = false;
  initMessage = '';

  /**
   * Instantiate `agave.wasm`. Pass a URL or an already-fetched module buffer to
   * skip the default same-origin fetch (tests, custom hosting).
   */
  async init(source: string | ArrayBuffer = default_wasm_url): Promise<void> {
    let bytes: ArrayBuffer;
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- boundary type test for the string|ArrayBuffer union
    if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) {
        throw new AgaveError(
          'wasm_fetch_failed',
          `Failed to download agave.wasm (HTTP ${String(response.status)})`,
          response.status,
        );
      }
      bytes = await response.arrayBuffer();
    } else {
      bytes = source;
    }
    let wasmMemory: WebAssembly.Memory | null = null;
    const importObject: WebAssembly.Imports = {
      env: {
        // WebGPU API imports would go here for GPU backend
        // For now, CPU-only via WASM
      },
      wasi_snapshot_preview1: {
        // Minimal WASI stubs for Zig's std library
        fd_write: () => 0,
        fd_read: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0,
        proc_exit: (code: number) => { throw new AgaveError('wasm_invalid', `Process exit: ${String(code)}`); },
        environ_get: () => 0,
        environ_sizes_get: () => 0,
        clock_time_get: () => 0,
        random_get: (ptr: number, len: number) => {
          if (!wasmMemory) {return -1;}
          crypto.getRandomValues(new Uint8Array(wasmMemory.buffer, ptr, len));
          return 0;
        },
      },
    };

    let instance: WebAssembly.Instance;
    try {
      const result = await WebAssembly.instantiate(bytes, importObject);
      instance = result.instance;
    } catch (e) {
      if (e instanceof AgaveError) {throw e;}
      const msg = e instanceof Error ? e.message : String(e);
      throw new AgaveError('wasm_invalid', `Failed to instantiate agave.wasm: ${msg}`);
    }
    wasmMemory = wasmExports(instance).memory;
    this.wasm = instance;
    this.ready = true;
    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log('Agave WASM engine initialized');
  }

  /**
   * Load a model from a URL, ArrayBuffer, or typed-array view.
   */
  async loadModel(source: ModelSource): Promise<void> {
    if (!this.wasm) { throw new AgaveError('not_initialized', 'Engine not initialized'); }
    const exp = wasmExports(this.wasm);
    let data: Uint8Array;
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- boundary type test for the string|buffer union; no schema parser to delegate to
    if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) {
        throw new AgaveError(
          'download_failed',
          `Failed to download model (HTTP ${String(response.status)})`,
          response.status,
        );
      }
      data = new Uint8Array(await response.arrayBuffer());
    } else {
      data = bytesFromBuffer(source);
    }

    // Allocate WASM memory and copy model data
    const ptr = exp.agave_alloc(data.byteLength);
    if (ptr === 0) { throw new AgaveError('alloc_failed', 'Failed to allocate WASM memory for model'); }

    const wasmMem = new Uint8Array(exp.memory.buffer, ptr, data.byteLength);
    wasmMem.set(data);

    // Initialize a new context first; keep the previous one until this succeeds
    // so a failed reload does not leave the chat with no model.
    const newCtx = exp.agave_init(ptr, data.byteLength);
    // Model buffer is borrowed by GGUF and freed by agave_free. If init could
    // not allocate a context, the host still owns `ptr` and must dealloc it.
    if (newCtx === 0) {
      exp.agave_dealloc(ptr, data.byteLength);
      throw new AgaveError('init_failed', 'Failed to initialize model');
    }

    let initMessage: string;
    try {
      initMessage = readCtxOutput(exp, newCtx);
    } catch (e) {
      exp.agave_free(newCtx);
      throw e;
    }
    const wasm_code = exp.agave_last_error(newCtx);

    // agave_init returns a context on parse/init errors too, with a diagnostic
    // instead of the "Loaded:" banner. Do not treat that as a successful load.
    if (wasm_code !== wasm_err.ok || !initMessage.startsWith('Loaded:')) {
      exp.agave_free(newCtx);
      throw new AgaveError(
        initErrorCode(wasm_code),
        initMessage || 'Failed to initialize model',
      );
    }

    if (this.ctx) {exp.agave_free(this.ctx);}
    this.ctx = newCtx;
    this.initMessage = initMessage;

    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log(`Model loaded: ${(data.byteLength / 1024 / 1024).toFixed(1)} MB, ${this.initMessage}`);
  }

  /**
   * Generate text from a prompt.
   */
  // oxlint-disable-next-line eslint/require-await, typescript-eslint/require-await -- async signature is the documented Promise API contract
  async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
    if (!this.wasm) { throw new AgaveError('not_initialized', 'Engine not initialized'); }
    if (!this.ctx) { throw new AgaveError('no_model', 'No model loaded'); }
    const exp = wasmExports(this.wasm);

    // Both an omitted and an explicitly zero budget mean "use the default".
    const maxTokens = options.maxTokens === undefined || options.maxTokens === 0
      ? default_max_tokens
      : options.maxTokens;
    const encoder = new TextEncoder();
    const promptBytes = encoder.encode(prompt);

    // Copy prompt to WASM memory. alloc(0) returns 0, which is not an OOM.
    let promptPtr = 0;
    if (promptBytes.length > 0) {
      promptPtr = exp.agave_alloc(promptBytes.length);
      if (promptPtr === 0) {
        throw new AgaveError('alloc_failed', 'Failed to allocate WASM memory for prompt');
      }
      const promptMem = new Uint8Array(exp.memory.buffer, promptPtr, promptBytes.length);
      promptMem.set(promptBytes);
    }

    exp.agave_generate(this.ctx, promptPtr, promptBytes.length, maxTokens);
    if (promptPtr !== 0) {exp.agave_dealloc(promptPtr, promptBytes.length);}

    const output = readCtxOutput(exp, this.ctx);
    const wasm_code = exp.agave_last_error(this.ctx);
    if (wasm_code !== wasm_err.ok) {
      throw new AgaveError(
        generateErrorCode(wasm_code),
        output || 'Generation failed',
      );
    }

    return output;
  }

  /**
   * Free the loaded model. The WASM instance stays so `loadModel` can run again
   * without another `init()`.
   */
  destroy(): void {
    if (this.wasm && this.ctx) {
      wasmExports(this.wasm).agave_free(this.ctx);
      this.ctx = 0;
    }
  }
}

// Explicit binding: classic scripts should not rely on declaration-position magic.
// Object.assign widens globalThis structurally, so no cast is needed.
Object.assign(globalThis, { AgaveEngine, AgaveError });
