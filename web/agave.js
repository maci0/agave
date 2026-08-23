// oxlint-disable @rikalabs/no-standalone-classes import/unambiguous -- classic browser script loaded via <script src> in web/index.html; AgaveEngine is a stateful engine with a documented constructor API
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
 *   const output = await agave.generate('What is 2+2?', { maxTokens: 100 });
 */

class AgaveEngine {
  wasm = null;
  ctx = 0;
  ready = false;

  async init() {
    const response = await fetch('agave.wasm');
    const bytes = await response.arrayBuffer();
    let wasmMemory = null;
    const importObject = {
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
        proc_exit: (code) => { throw new Error(`Process exit: ${code}`); },
        environ_get: () => 0,
        environ_sizes_get: () => 0,
        clock_time_get: () => 0,
        random_get: (ptr, len) => {
          crypto.getRandomValues(new Uint8Array(wasmMemory.buffer, ptr, len));
          return 0;
        },
      },
    };

    const result = await WebAssembly.instantiate(bytes, importObject);
    wasmMemory = result.instance.exports.memory;
    this.wasm = result.instance;
    this.ready = true;
    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log('Agave WASM engine initialized');
  }

  /**
   * Load a model from a URL or ArrayBuffer.
   * @param {string|ArrayBuffer} source - URL to fetch or raw model data
   */
  async loadModel(source) {
    let data = source;
    // Runtime union discrimination at the API boundary. The JSDoc contract admits
    // both shapes; there is no schema parser to delegate to.
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- boundary type test for the string|ArrayBuffer union
    if (typeof source === 'string') {
      const response = await fetch(source);
      data = await response.arrayBuffer();
    }

    // Allocate WASM memory and copy model data
    const ptr = this.wasm.exports.agave_alloc(data.byteLength);
    if (ptr === 0) { throw new Error('Failed to allocate WASM memory for model'); }

    const wasmMem = new Uint8Array(this.wasm.exports.memory.buffer, ptr, data.byteLength);
    wasmMem.set(new Uint8Array(data));

    // Initialize inference context
    this.ctx = this.wasm.exports.agave_init(ptr, data.byteLength);
    // Model buffer is borrowed by GGUF — do NOT agave_dealloc until agave_free.
    if (this.ctx === 0) { throw new Error('Failed to initialize model'); }

    // Read init status message
    const statusBufSize = 4096;
    const statusPtr = this.wasm.exports.agave_alloc(statusBufSize);
    const statusLen = this.wasm.exports.agave_get_output(this.ctx, statusPtr, statusBufSize);
    const statusMem = new Uint8Array(this.wasm.exports.memory.buffer, statusPtr, statusLen);
    this.initMessage = new TextDecoder().decode(statusMem);
    this.wasm.exports.agave_dealloc(statusPtr, statusBufSize);

    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log(`Model loaded: ${(data.byteLength / 1024 / 1024).toFixed(1)} MB — ${this.initMessage}`);
  }

  /**
   * Generate text from a prompt.
   * @param {string} prompt - Input text
   * @param {Object} options - { maxTokens: 100 }
   * @returns {Promise<string>} Generated text
   */
  // Async signature is the documented API contract even though the body is sync.
  // oxlint-disable-next-line eslint/require-await typescript-eslint/require-await -- Promise API contract
  async generate(prompt, options = {}) {
    if (!this.ctx) { throw new Error('No model loaded'); }

    // Explicit zero must not fall back to the default token budget.
    // oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- falsy-or semantics intended for numeric options
    const maxTokens = options.maxTokens || 100;
    const encoder = new TextEncoder();
    const promptBytes = encoder.encode(prompt);

    // Copy prompt to WASM memory
    const promptPtr = this.wasm.exports.agave_alloc(promptBytes.length);
    const promptMem = new Uint8Array(this.wasm.exports.memory.buffer, promptPtr, promptBytes.length);
    promptMem.set(promptBytes);

    // Generate
    this.wasm.exports.agave_generate(this.ctx, promptPtr, promptBytes.length, maxTokens);
    this.wasm.exports.agave_dealloc(promptPtr, promptBytes.length);

    // Read output
    const outBufSize = 16_384;
    const outPtr = this.wasm.exports.agave_alloc(outBufSize);
    const outLen = this.wasm.exports.agave_get_output(this.ctx, outPtr, outBufSize);
    const outMem = new Uint8Array(this.wasm.exports.memory.buffer, outPtr, outLen);
    const decoder = new TextDecoder();
    const output = decoder.decode(outMem);
    this.wasm.exports.agave_dealloc(outPtr, outBufSize);

    return output;
  }

  /**
   * Free resources.
   */
  destroy() {
    if (this.ctx) {
      this.wasm.exports.agave_free(this.ctx);
      this.ctx = 0;
    }
  }
}

// Classic scripts expose the class implicitly; pin it explicitly so the
// contract in web/index.html does not depend on declaration-position magic.
globalThis.AgaveEngine = AgaveEngine;
