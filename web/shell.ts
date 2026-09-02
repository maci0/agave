/**
 * Standalone WASM chat shell. Loaded after `agave.js` by `web/index.html`.
 * Distinct from `src/web/` (HTTP --serve chat UI).
 */

const engine = new AgaveEngine();
const chat = document.getElementById('chat') as HTMLElement;
const statusEl = document.getElementById('status') as HTMLElement;
const promptInput = document.getElementById('prompt') as HTMLInputElement;
const dropZone = document.getElementById('drop-zone') as HTMLElement;

/** GGUF files start with this 4-byte magic (`GGUF`). */
const gguf_magic = [0x47, 0x47, 0x55, 0x46] as const;

function announceToSR(text: string): void {
  const el = document.getElementById('sr-announce');
  if (el) { el.textContent = ''; setTimeout(() => { el.textContent = text; }, 100); }
}

function fmtMb(bytes: number): string {
  return (bytes / 1024 / 1024).toLocaleString(undefined, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  });
}

function isGgufName(name: string): boolean {
  return name.toLowerCase().endsWith('.gguf');
}

function isGgufBuffer(data: ArrayBuffer): boolean {
  if (data.byteLength < gguf_magic.length) {return false;}
  const head = new Uint8Array(data, 0, gguf_magic.length);
  return head[0] === gguf_magic[0] && head[1] === gguf_magic[1]
    && head[2] === gguf_magic[2] && head[3] === gguf_magic[3];
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

function setUrlError(message: string): void {
  const urlInput = document.getElementById('model-url') as HTMLInputElement;
  const urlError = document.getElementById('url-error') as HTMLElement;
  statusEl.textContent = message;
  urlError.textContent = message;
  announceToSR(message);
  urlInput.setAttribute('aria-invalid', 'true');
  urlInput.setAttribute('aria-describedby', 'url-error');
  urlInput.focus();
}

function clearUrlError(): void {
  const urlInput = document.getElementById('model-url') as HTMLInputElement;
  const urlError = document.getElementById('url-error') as HTMLElement;
  urlInput.removeAttribute('aria-invalid');
  urlInput.removeAttribute('aria-describedby');
  urlError.textContent = '';
}

/** Map engine/network failures to short, actionable copy. */
function friendlyLoadError(error: unknown): string {
  if (error instanceof AgaveError) {
    switch (error.code) {
      case 'wasm_fetch_failed':
        return 'Could not load the inference engine. Reload the page, or check that agave.wasm is being served.';
      case 'wasm_invalid':
        return 'The inference engine failed to start. Reload the page.';
      case 'download_failed':
        if (error.httpStatus === 404) {return 'The model URL was not found. Check the link.';}
        if (error.httpStatus === 403 || error.httpStatus === 401) {
          return 'The model URL refused the download. Try dropping a GGUF file instead.';
        }
        return 'Could not download the model. Check the URL, or drop a GGUF file instead. Some hosts block browser downloads.';
      case 'gguf_parse':
        return 'This file is not a valid GGUF model.';
      case 'unsupported_arch':
        return 'This model architecture is not supported in the browser.';
      case 'no_vocab':
        return 'This GGUF file has no vocabulary and cannot be used.';
      case 'tokenizer':
        return 'Could not read the tokenizer from this model file.';
      case 'init_failed':
        return 'Could not initialize this model in the browser.';
      case 'alloc_failed':
        return 'The model is too large to fit in this browser.';
      case 'not_initialized':
        return 'The engine is not ready. Reload the page and try again.';
      default:
        return error.message.startsWith('Could not') ? error.message : `Could not load model: ${error.message}`;
    }
  }
  const msg = error instanceof Error ? error.message : String(error);
  const lower = msg.toLowerCase();
  if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
    return 'Could not download the model. Check the URL, or drop a GGUF file instead. Some hosts block browser downloads.';
  }
  if (lower.includes('http 404')) {return 'The model URL was not found. Check the link.';}
  if (lower.includes('http 403') || lower.includes('http 401')) {
    return 'The model URL refused the download. Try dropping a GGUF file instead.';
  }
  if (lower.startsWith('gguf parse error') || lower.includes('not a valid gguf')) {
    return 'This file is not a valid GGUF model.';
  }
  if (lower.startsWith('unsupported arch')) {
    return 'This model architecture is not supported in the browser.';
  }
  if (lower.startsWith('no vocab')) {return 'This GGUF file has no vocabulary and cannot be used.';}
  if (lower.startsWith('tok error')) {return 'Could not read the tokenizer from this model file.';}
  if (lower.startsWith('model init error') || lower === 'failed to initialize model') {
    return 'Could not initialize this model in the browser.';
  }
  if (lower.includes('failed to allocate')) {
    return 'The model is too large to fit in this browser.';
  }
  if (lower === 'engine not initialized') {
    return 'The engine is not ready. Reload the page and try again.';
  }
  return msg.startsWith('Could not') ? msg : `Could not load model: ${msg}`;
}

async function downloadModel(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download model (HTTP ${String(response.status)})`);
  }
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body?.getReader();
  if (!reader) {return response.arrayBuffer();}
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {break;}
    chunks.push(value);
    received += value.byteLength;
    if (total > 0) {
      const pct = Math.round((received / total) * 100);
      statusEl.textContent = `Downloading model… ${fmtMb(received)} / ${fmtMb(total)} MB (${String(pct)}%)`;
    } else {
      statusEl.textContent = `Downloading model… ${fmtMb(received)} MB`;
    }
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

// Drag & drop
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', (e) => {
  const to = e.relatedTarget;
  if (to instanceof Node && dropZone.contains(to)) {return;}
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files[0];
  if (file) {void loadModelFromBuffer(file);}
});

async function loadModelFromUrl(): Promise<void> {
  const urlInput = document.getElementById('model-url') as HTMLInputElement;
  const url = urlInput.value.trim();
  if (!url) {
    setUrlError('Enter a model URL first');
    return;
  }
  if (!isHttpUrl(url)) {
    setUrlError('Enter a valid http(s) URL to a GGUF file');
    return;
  }
  clearUrlError();
  await initAndLoad(async () => {
    statusEl.textContent = 'Downloading model…';
    const data = await downloadModel(url);
    if (!isGgufBuffer(data)) {
      throw new Error('This file is not a valid GGUF model.');
    }
    await engine.loadModel(data);
  }, true);
}

function loadModelFromFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {return;}
  void loadModelFromBuffer(file);
}

async function loadModelFromBuffer(file: File): Promise<void> {
  if (file.name && !isGgufName(file.name)) {
    const message = 'This is not a GGUF model file. Choose a file ending in .gguf.';
    statusEl.textContent = message;
    addMessage('error', message);
    announceToSR(message);
    return;
  }
  await initAndLoad(async () => {
    statusEl.textContent = `Reading ${file.name} (${fmtMb(file.size)} MB)…`;
    const data = await file.arrayBuffer();
    if (!isGgufBuffer(data)) {
      throw new Error('This file is not a valid GGUF model.');
    }
    await engine.loadModel(data);
  }, false);
}

async function initAndLoad(loadFn: () => Promise<void>, fromUrl: boolean): Promise<void> {
  const loadBtn = document.getElementById('load-btn') as HTMLButtonElement;
  const dropZoneEl = document.getElementById('drop-zone') as HTMLElement;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  const urlInput = document.getElementById('model-url') as HTMLInputElement;
  const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
  const hadModel = Boolean(engine.ctx);
  loadBtn.disabled = true;
  loadBtn.setAttribute('aria-busy', 'true');
  loadBtn.textContent = 'Loading…';
  dropZoneEl.setAttribute('aria-disabled', 'true');
  fileInput.disabled = true;
  urlInput.disabled = true;
  promptInput.disabled = true;
  sendBtn.disabled = true;
  statusEl.textContent = 'Initializing engine…';
  try {
    if (!engine.ready) {await engine.init();}
    await loadFn();
    addMessage('system', engine.initMessage || 'Model loaded');
    statusEl.textContent = 'Ready';
    announceToSR('Model loaded. Ready to chat.');
    promptInput.disabled = false;
    promptInput.placeholder = 'Type a message...';
    sendBtn.disabled = false;
    const hint = document.getElementById('input-hint');
    if (hint) {hint.hidden = false;}
    promptInput.setAttribute('aria-describedby', 'input-hint');
    const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement | null;
    if (clearBtn) {clearBtn.hidden = false;}
    promptInput.focus();
  } catch (e) {
    const message = friendlyLoadError(e);
    statusEl.textContent = message;
    addMessage('error', message);
    announceToSR(`Error loading model: ${message}`);
    if (fromUrl) {
      urlInput.setAttribute('aria-invalid', 'true');
      urlInput.setAttribute('aria-describedby', 'url-error');
      const urlError = document.getElementById('url-error') as HTMLElement;
      urlError.textContent = message;
    }
    // A failed reload must not disable chat if the previous model is still loaded.
    if (hadModel && engine.ctx) {
      promptInput.disabled = false;
      sendBtn.disabled = false;
    }
  }
  loadBtn.disabled = false;
  loadBtn.removeAttribute('aria-busy');
  loadBtn.textContent = 'Load model';
  dropZoneEl.removeAttribute('aria-disabled');
  fileInput.disabled = false;
  urlInput.disabled = false;
}

function restoreEmptyChat(): void {
  chat.replaceChildren();
  const empty = document.createElement('div');
  empty.id = 'chat-empty';
  empty.className = 'msg empty-hint';
  empty.textContent = engine.ctx
    ? 'Send a prompt.'
    : 'Load a GGUF model above, then send a prompt.';
  chat.append(empty);
}

function clearChat(): void {
  if (isSending) {return;}
  if (document.getElementById('chat-empty')) {
    statusEl.textContent = 'Nothing to clear';
    return;
  }
  if (!confirm('Clear this conversation?')) {return;} // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
  restoreEmptyChat();
  statusEl.textContent = engine.ctx ? 'Ready' : 'Load a GGUF model to begin';
  announceToSR('Conversation cleared');
  promptInput.focus();
}

let isSending = false;
async function send(): Promise<void> {
  const text = promptInput.value.trim();
  if (!text || isSending) {return;}
  isSending = true;

  const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
  addMessage('user', text);
  promptInput.value = '';
  promptInput.disabled = true;
  sendBtn.disabled = true;
  sendBtn.setAttribute('aria-busy', 'true');
  sendBtn.textContent = 'Generating…';
  chat.setAttribute('aria-busy', 'true');
  statusEl.textContent = 'Generating…';
  announceToSR('Generating response…');

  // Visible in-chat placeholder so the UI does not look stuck after send.
  const pending = document.createElement('div');
  pending.className = 'msg assistant thinking';
  pending.id = 'gen-pending';
  pending.dir = 'auto';
  pending.setAttribute('role', 'status');
  pending.setAttribute('aria-label', 'Generating response');
  pending.textContent = '…';
  chat.appendChild(pending);
  chat.scrollTop = chat.scrollHeight;

  try {
    const output = await engine.generate(text, { maxTokens: 200 });
    pending.remove();
    addMessage('assistant', output);
  } catch (e) {
    pending.remove();
    const no_model = e instanceof AgaveError
      && (e.code === 'no_model' || e.code === 'not_initialized');
    const message = e instanceof Error ? e.message : String(e);
    const lower = message.toLowerCase();
    const shown = no_model || lower === 'no model loaded' || lower === 'model not initialized'
      ? 'Load a GGUF model first.'
      : `Could not generate a reply: ${message}`;
    addMessage('error', shown);
  }

  promptInput.disabled = false;
  sendBtn.disabled = false;
  sendBtn.removeAttribute('aria-busy');
  sendBtn.textContent = 'Send';
  chat.setAttribute('aria-busy', 'false');
  statusEl.textContent = 'Ready';
  promptInput.focus();
  announceToSR('Response complete');
  isSending = false;
}

function truncateAnnounce(text: string, maxChars: number): string {
  const chars = Array.from(text);
  if (chars.length <= maxChars) {return text;}
  return `${chars.slice(0, maxChars).join('')}...`;
}

function addMessage(role: string, text: string): void {
  const empty = document.getElementById('chat-empty');
  if (empty) {empty.remove();}
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.dir = 'auto';
  const roleLabels: Record<string, string> = { user: 'You', assistant: 'Agave', system: 'System', error: 'Error' };
  const roleId = `msg-role-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const roleEl = document.createElement('span');
  roleEl.id = roleId;
  roleEl.className = 'msg-role';
  roleEl.textContent = roleLabels[role] || role;
  const body = document.createElement('span');
  body.textContent = text;
  div.appendChild(roleEl);
  div.appendChild(body);
  if (role === 'error') {
    div.setAttribute('role', 'alert');
  } else {
    div.setAttribute('role', 'group');
    div.setAttribute('aria-labelledby', roleId);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  if (role === 'error' || role === 'system') {announceToSR(text);}
  else if (role === 'assistant') {
    announceToSR(`Agave responded: ${truncateAnnounce(text, 200)}`);
  }
}

promptInput.addEventListener('keydown', (e) => {
  // Ignore Enter during IME composition (CJK input): Enter there confirms
  // the conversion, it must not send. Matches src/web/app.ts behavior.
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); void send(); }
});

const modelUrl = document.getElementById('model-url') as HTMLInputElement;
modelUrl.addEventListener('input', () => { clearUrlError(); });
modelUrl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); void loadModelFromUrl(); }
});

(globalThis as unknown as {
  loadModelFromUrl: typeof loadModelFromUrl;
  loadModelFromFile: typeof loadModelFromFile;
  send: typeof send;
  clearChat: typeof clearChat;
}).loadModelFromUrl = loadModelFromUrl;
(globalThis as unknown as { loadModelFromFile: typeof loadModelFromFile }).loadModelFromFile = loadModelFromFile;
(globalThis as unknown as { send: typeof send }).send = send;
(globalThis as unknown as { clearChat: typeof clearChat }).clearChat = clearChat;
