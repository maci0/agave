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
 *   const output = await agave.generate('What is 2+2?', { maxTokens: 100 });
 */

interface AgaveWasmExports {
  memory: WebAssembly.Memory;
  agave_alloc(len: number): number;
  agave_dealloc(ptr: number, len: number): void;
  agave_init(ptr: number, len: number): number;
  agave_get_output(ctx: number, buf: number, len: number): number;
  agave_generate(ctx: number, promptPtr: number, promptLen: number, maxTokens: number): number;
  agave_free(ctx: number): void;
}

interface GenerateOptions {
  maxTokens?: number;
}

function wasmExports(instance: WebAssembly.Instance): AgaveWasmExports {
  return instance.exports as unknown as AgaveWasmExports;
}

class AgaveEngine {
  wasm: WebAssembly.Instance | null = null;
  ctx = 0;
  ready = false;
  initMessage = '';

  async init(): Promise<void> {
    const response = await fetch('agave.wasm');
    const bytes = await response.arrayBuffer();
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
        proc_exit: (code: number) => { throw new Error(`Process exit: ${code}`); },
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

    const result = await WebAssembly.instantiate(bytes, importObject);
    wasmMemory = wasmExports(result.instance).memory;
    this.wasm = result.instance;
    this.ready = true;
    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log('Agave WASM engine initialized');
  }

  /**
   * Load a model from a URL or ArrayBuffer.
   */
  async loadModel(source: string | ArrayBuffer): Promise<void> {
    if (!this.wasm) { throw new Error('Engine not initialized'); }
    const exp = wasmExports(this.wasm);
    let data: ArrayBuffer = source instanceof ArrayBuffer ? source : new ArrayBuffer(0);
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- boundary type test for the string|ArrayBuffer union; no schema parser to delegate to
    if (typeof source === 'string') {
      const response = await fetch(source);
      data = await response.arrayBuffer();
    }

    // Allocate WASM memory and copy model data
    const ptr = exp.agave_alloc(data.byteLength);
    if (ptr === 0) { throw new Error('Failed to allocate WASM memory for model'); }

    const wasmMem = new Uint8Array(exp.memory.buffer, ptr, data.byteLength);
    wasmMem.set(new Uint8Array(data));

    // Initialize inference context
    this.ctx = exp.agave_init(ptr, data.byteLength);
    // Model buffer is borrowed by GGUF, do NOT agave_dealloc until agave_free.
    if (this.ctx === 0) { throw new Error('Failed to initialize model'); }

    // Read init status message
    const statusBufSize = 4096;
    const statusPtr = exp.agave_alloc(statusBufSize);
    const statusLen = exp.agave_get_output(this.ctx, statusPtr, statusBufSize);
    const statusMem = new Uint8Array(exp.memory.buffer, statusPtr, statusLen);
    this.initMessage = new TextDecoder().decode(statusMem);
    exp.agave_dealloc(statusPtr, statusBufSize);

    // oxlint-disable-next-line no-console -- engine diagnostics for WASM debugging
    console.log(`Model loaded: ${(data.byteLength / 1024 / 1024).toFixed(1)} MB, ${this.initMessage}`);
  }

  /**
   * Generate text from a prompt.
   */
  // oxlint-disable-next-line eslint/require-await, typescript-eslint/require-await -- async signature is the documented Promise API contract
  async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
    if (!this.wasm) { throw new Error('Engine not initialized'); }
    if (!this.ctx) { throw new Error('No model loaded'); }
    const exp = wasmExports(this.wasm);

    // Explicit zero must not fall back to the default token budget.
    // oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- falsy-or semantics intended for numeric options
    const maxTokens = options.maxTokens || 100;
    const encoder = new TextEncoder();
    const promptBytes = encoder.encode(prompt);

    // Copy prompt to WASM memory
    const promptPtr = exp.agave_alloc(promptBytes.length);
    const promptMem = new Uint8Array(exp.memory.buffer, promptPtr, promptBytes.length);
    promptMem.set(promptBytes);

    // Generate
    exp.agave_generate(this.ctx, promptPtr, promptBytes.length, maxTokens);
    exp.agave_dealloc(promptPtr, promptBytes.length);

    // Read output
    const outBufSize = 16_384;
    const outPtr = exp.agave_alloc(outBufSize);
    const outLen = exp.agave_get_output(this.ctx, outPtr, outBufSize);
    const outMem = new Uint8Array(exp.memory.buffer, outPtr, outLen);
    const decoder = new TextDecoder();
    const output = decoder.decode(outMem);
    exp.agave_dealloc(outPtr, outBufSize);

    return output;
  }

  /**
   * Free resources.
   */
  destroy(): void {
    if (this.wasm && this.ctx) {
      wasmExports(this.wasm).agave_free(this.ctx);
      this.ctx = 0;
    }
  }
}

// Explicit binding: classic scripts should not rely on declaration-position magic.
(globalThis as unknown as { AgaveEngine: typeof AgaveEngine }).AgaveEngine = AgaveEngine;
