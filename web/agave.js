(() => {
  // web/agave.ts
  class AgaveError extends Error {
    code;
    httpStatus;
    constructor(code, message, httpStatus) {
      super(message);
      this.name = "AgaveError";
      this.code = code;
      this.httpStatus = httpStatus;
    }
  }
  var default_max_tokens = 100;
  var max_u32 = 4294967295;
  var output_buf_size = 16384;
  var default_wasm_url = "agave.wasm";
  var wasm_err = {
    ok: 0,
    not_ready: 1,
    tokenize: 2,
    gguf_parse: 3,
    unsupported_arch: 4,
    no_vocab: 5,
    tokenizer: 6,
    model_init: 7,
    invalid_handle: 8
  };
  var required_exports = [
    "agave_alloc",
    "agave_dealloc",
    "agave_init",
    "agave_get_output",
    "agave_generate",
    "agave_last_error",
    "agave_free"
  ];
  var wrapFetchError = (cause, code, label) => {
    if (cause instanceof AgaveError) {
      return cause;
    }
    const msg = cause instanceof Error ? cause.message : String(cause);
    return new AgaveError(code, `Failed to download ${label}: ${msg}`);
  };
  var fetchOrThrow = async (url, init, code, label) => {
    try {
      return await fetch(url, init);
    } catch (error) {
      throw wrapFetchError(error, code, label);
    }
  };
  var fetchBuffer = async (url, code, label, signal) => {
    const init = signal === undefined ? undefined : { signal };
    const response = await fetchOrThrow(url, init, code, label);
    if (!response.ok) {
      throw new AgaveError(code, `Failed to download ${label} (HTTP ${String(response.status)})`, response.status);
    }
    try {
      return await response.arrayBuffer();
    } catch (error) {
      throw wrapFetchError(error, code, label);
    }
  };
  var instantiateModule = async (bytes, importObject) => {
    try {
      const { instance } = await WebAssembly.instantiate(bytes, importObject);
      return instance;
    } catch (error) {
      if (error instanceof AgaveError) {
        throw error;
      }
      const msg = error instanceof Error ? error.message : String(error);
      throw new AgaveError("wasm_invalid", `Failed to instantiate agave.wasm: ${msg}`);
    }
  };
  var wasmExports = (instance) => {
    const { exports } = instance;
    if (!(exports.memory instanceof WebAssembly.Memory)) {
      throw new AgaveError("wasm_invalid", "agave.wasm: missing or invalid memory export");
    }
    for (const name of required_exports) {
      if (typeof exports[name] !== "function") {
        throw new AgaveError("wasm_invalid", `agave.wasm: missing export ${name}`);
      }
    }
    return exports;
  };
  var bytesFromBuffer = (source) => {
    if (source instanceof ArrayBuffer) {
      return new Uint8Array(source);
    }
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  };
  var arrayBufferFromSource = (source) => {
    if (source instanceof ArrayBuffer) {
      return source;
    }
    const copy = new Uint8Array(source.byteLength);
    copy.set(bytesFromBuffer(source));
    return copy.buffer;
  };
  var readCtxOutput = (exp, ctx) => {
    const outPtr = exp.agave_alloc(output_buf_size);
    if (outPtr === 0) {
      throw new AgaveError("alloc_failed", "Failed to allocate WASM memory for output");
    }
    try {
      const outLen = exp.agave_get_output(ctx, outPtr, output_buf_size);
      return new TextDecoder().decode(new Uint8Array(exp.memory.buffer, outPtr, outLen));
    } finally {
      exp.agave_dealloc(outPtr, output_buf_size);
    }
  };
  var initErrorCode = (wasm_code) => {
    switch (wasm_code) {
      case wasm_err.gguf_parse: {
        return "gguf_parse";
      }
      case wasm_err.unsupported_arch: {
        return "unsupported_arch";
      }
      case wasm_err.no_vocab: {
        return "no_vocab";
      }
      case wasm_err.tokenizer: {
        return "tokenizer";
      }
      default: {
        return "init_failed";
      }
    }
  };
  var generateErrorCode = (wasm_code) => {
    if (wasm_code === wasm_err.not_ready || wasm_code === wasm_err.invalid_handle) {
      return "no_model";
    }
    if (wasm_code === wasm_err.tokenize) {
      return "tokenize";
    }
    return "generate_failed";
  };

  class AgaveEngine {
    #wasm = null;
    #ctx = 0;
    #init_message = "";
    get ready() {
      return this.#wasm !== null;
    }
    get hasModel() {
      return this.#ctx !== 0;
    }
    get initMessage() {
      return this.#init_message;
    }
    async init(source = default_wasm_url, signal) {
      const bytes = typeof source === "string" ? await fetchBuffer(source, "wasm_fetch_failed", "agave.wasm", signal) : arrayBufferFromSource(source);
      let wasmMemory = null;
      const importObject = {
        env: {},
        wasi_snapshot_preview1: {
          fd_write: () => 0,
          fd_read: () => 0,
          fd_close: () => 0,
          fd_seek: () => 0,
          proc_exit: (code) => {
            throw new AgaveError("wasm_invalid", `Process exit: ${String(code)}`);
          },
          environ_get: () => 0,
          environ_sizes_get: () => 0,
          clock_time_get: () => 0,
          random_get: (ptr, len) => {
            if (!wasmMemory) {
              return -1;
            }
            crypto.getRandomValues(new Uint8Array(wasmMemory.buffer, ptr, len));
            return 0;
          }
        }
      };
      const instance = await instantiateModule(bytes, importObject);
      wasmMemory = wasmExports(instance).memory;
      const prev = this.#wasm;
      const prev_ctx = this.#ctx;
      this.#wasm = instance;
      this.#ctx = 0;
      this.#init_message = "";
      if (prev && prev_ctx) {
        wasmExports(prev).agave_free(prev_ctx);
      }
      console.log("Agave WASM engine initialized");
    }
    async fetchModel(url, options = {}) {
      const init = options.signal === undefined ? undefined : { signal: options.signal };
      const response = await fetchOrThrow(url, init, "download_failed", "model");
      if (!response.ok) {
        throw new AgaveError("download_failed", `Failed to download model (HTTP ${String(response.status)})`, response.status);
      }
      if (!options.onProgress) {
        try {
          return await response.arrayBuffer();
        } catch (error) {
          throw wrapFetchError(error, "download_failed", "model");
        }
      }
      const total = Number(response.headers.get("content-length")) || 0;
      const reader = response.body?.getReader();
      if (!reader) {
        const buffer = await response.arrayBuffer();
        options.onProgress({ received: buffer.byteLength, total });
        return buffer;
      }
      const chunks = [];
      let received = 0;
      for (;; ) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        chunks.push(value);
        received += value.byteLength;
        options.onProgress({ received, total });
      }
      const out = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return out.buffer;
    }
    async loadModel(source, signal) {
      if (!this.#wasm) {
        throw new AgaveError("not_initialized", "Engine not initialized");
      }
      const exp = wasmExports(this.#wasm);
      const data = typeof source === "string" ? new Uint8Array(await fetchBuffer(source, "download_failed", "model", signal)) : bytesFromBuffer(source);
      if (data.byteLength === 0) {
        throw new AgaveError("invalid_argument", "Model source is empty");
      }
      const ptr = exp.agave_alloc(data.byteLength);
      if (ptr === 0) {
        throw new AgaveError("alloc_failed", "Failed to allocate WASM memory for model");
      }
      const wasmMem = new Uint8Array(exp.memory.buffer, ptr, data.byteLength);
      wasmMem.set(data);
      const newCtx = exp.agave_init(ptr, data.byteLength);
      if (newCtx === 0) {
        exp.agave_dealloc(ptr, data.byteLength);
        throw new AgaveError("init_failed", "Failed to initialize model");
      }
      const initMessage = (() => {
        try {
          return readCtxOutput(exp, newCtx);
        } catch (error) {
          exp.agave_free(newCtx);
          throw error;
        }
      })();
      const wasm_code = exp.agave_last_error(newCtx);
      if (wasm_code !== wasm_err.ok || !initMessage.startsWith("Loaded:")) {
        exp.agave_free(newCtx);
        throw new AgaveError(initErrorCode(wasm_code), initMessage || "Failed to initialize model");
      }
      if (this.#ctx) {
        exp.agave_free(this.#ctx);
      }
      this.#ctx = newCtx;
      this.#init_message = initMessage;
      console.log(`Model loaded: ${(data.byteLength / 1024 / 1024).toFixed(1)} MB, ${this.#init_message}`);
    }
    async generate(prompt, options = {}) {
      if (!this.#wasm) {
        throw new AgaveError("not_initialized", "Engine not initialized");
      }
      if (!this.#ctx) {
        throw new AgaveError("no_model", "No model loaded");
      }
      const exp = wasmExports(this.#wasm);
      const requested = options.maxTokens;
      const out_of_range = requested !== undefined && requested !== 0 && (!Number.isInteger(requested) || requested < 0 || requested > max_u32);
      if (out_of_range) {
        throw new AgaveError("invalid_argument", "maxTokens must be a non-negative integer that fits in 32 bits (0 uses the default)");
      }
      const maxTokens = requested === undefined || requested === 0 ? default_max_tokens : requested;
      const encoder = new TextEncoder;
      const promptBytes = encoder.encode(prompt);
      let promptPtr = 0;
      try {
        if (promptBytes.length > 0) {
          promptPtr = exp.agave_alloc(promptBytes.length);
          if (promptPtr === 0) {
            throw new AgaveError("alloc_failed", "Failed to allocate WASM memory for prompt");
          }
          const promptMem = new Uint8Array(exp.memory.buffer, promptPtr, promptBytes.length);
          promptMem.set(promptBytes);
        }
        exp.agave_generate(this.#ctx, promptPtr, promptBytes.length, maxTokens);
        const output = readCtxOutput(exp, this.#ctx);
        const wasm_code = exp.agave_last_error(this.#ctx);
        if (wasm_code !== wasm_err.ok) {
          throw new AgaveError(generateErrorCode(wasm_code), output || "Generation failed");
        }
        return output;
      } finally {
        if (promptPtr !== 0) {
          exp.agave_dealloc(promptPtr, promptBytes.length);
        }
      }
    }
    destroy() {
      if (this.#wasm && this.#ctx) {
        wasmExports(this.#wasm).agave_free(this.#ctx);
      }
      this.#ctx = 0;
      this.#init_message = "";
    }
  }
  Object.assign(globalThis, { AgaveEngine, AgaveError });
})();
