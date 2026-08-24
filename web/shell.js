"use strict";
/**
 * Standalone WASM chat shell. Loaded after `agave.js` by `web/index.html`.
 * Distinct from `src/web/` (HTTP --serve chat UI).
 */
const engine = new AgaveEngine();
const chat = document.getElementById('chat');
const statusEl = document.getElementById('status');
const promptInput = document.getElementById('prompt');
const dropZone = document.getElementById('drop-zone');
function announceToSR(text) {
    const el = document.getElementById('sr-announce');
    if (el) {
        el.textContent = '';
        setTimeout(() => { el.textContent = text; }, 100);
    }
}
// Drag & drop
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer?.files[0];
    if (file) {
        void loadModelFromBuffer(file);
    }
});
async function loadModelFromUrl() {
    const urlInput = document.getElementById('model-url');
    const urlError = document.getElementById('url-error');
    const url = urlInput.value.trim();
    if (!url) {
        statusEl.textContent = 'Enter a model URL first';
        urlError.textContent = 'Enter a model URL first';
        announceToSR('Enter a model URL first');
        urlInput.setAttribute('aria-invalid', 'true');
        urlInput.setAttribute('aria-describedby', 'url-error');
        urlInput.focus();
        return;
    }
    urlInput.removeAttribute('aria-invalid');
    urlInput.removeAttribute('aria-describedby');
    urlError.textContent = '';
    await initAndLoad(async () => {
        statusEl.textContent = 'Downloading model…';
        await engine.loadModel(url);
    });
}
function loadModelFromFile(event) {
    const input = event.target;
    const file = input.files?.[0];
    if (!file) {
        return;
    }
    void loadModelFromBuffer(file);
}
async function loadModelFromBuffer(file) {
    await initAndLoad(async () => {
        statusEl.textContent = `Reading ${file.name} (${(file.size / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })} MB)…`;
        const data = await file.arrayBuffer();
        await engine.loadModel(data);
    });
}
async function initAndLoad(loadFn) {
    const loadBtn = document.getElementById('load-btn');
    const dropZoneEl = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    loadBtn.disabled = true;
    loadBtn.setAttribute('aria-busy', 'true');
    loadBtn.textContent = 'Loading…';
    dropZoneEl.setAttribute('aria-disabled', 'true');
    fileInput.disabled = true;
    statusEl.textContent = 'Initializing engine…';
    try {
        if (!engine.ready) {
            await engine.init();
        }
        await loadFn();
        addMessage('system', engine.initMessage || 'Model loaded');
        statusEl.textContent = 'Ready';
        announceToSR('Model loaded. Ready to chat.');
        promptInput.disabled = false;
        promptInput.placeholder = 'Type a message...';
        document.getElementById('send-btn').disabled = false;
        const hint = document.getElementById('input-hint');
        if (hint) {
            hint.hidden = false;
        }
        promptInput.setAttribute('aria-describedby', 'input-hint');
        promptInput.focus();
    }
    catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        statusEl.textContent = `Could not load model: ${message}`;
        addMessage('error', `Could not load model: ${message}`);
        announceToSR(`Error loading model: ${message}`);
    }
    loadBtn.disabled = false;
    loadBtn.removeAttribute('aria-busy');
    loadBtn.textContent = 'Load model';
    dropZoneEl.removeAttribute('aria-disabled');
    fileInput.disabled = false;
}
let isSending = false;
async function send() {
    const text = promptInput.value.trim();
    if (!text || isSending) {
        return;
    }
    isSending = true;
    const sendBtn = document.getElementById('send-btn');
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
    }
    catch (e) {
        pending.remove();
        const message = e instanceof Error ? e.message : String(e);
        addMessage('error', `Error: ${message}`);
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
function truncateAnnounce(text, maxChars) {
    const chars = Array.from(text);
    if (chars.length <= maxChars) {
        return text;
    }
    return `${chars.slice(0, maxChars).join('')}...`;
}
function addMessage(role, text) {
    const empty = document.getElementById('chat-empty');
    if (empty) {
        empty.remove();
    }
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.dir = 'auto';
    const roleLabels = { user: 'You', assistant: 'Agave', system: 'System', error: 'Error' };
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
    }
    else {
        div.setAttribute('role', 'group');
        div.setAttribute('aria-labelledby', roleId);
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    if (role === 'error' || role === 'system') {
        announceToSR(text);
    }
    else if (role === 'assistant') {
        announceToSR(`Agave responded: ${truncateAnnounce(text, 200)}`);
    }
}
promptInput.addEventListener('keydown', (e) => {
    // Ignore Enter during IME composition (CJK input): Enter there confirms
    // the conversion, it must not send. Matches src/web/app.ts behavior.
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        void send();
    }
});
const modelUrl = document.getElementById('model-url');
modelUrl.addEventListener('input', (e) => {
    const target = e.target;
    target.removeAttribute('aria-invalid');
    target.removeAttribute('aria-describedby');
    const urlError = document.getElementById('url-error');
    if (urlError) {
        urlError.textContent = '';
    }
});
modelUrl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        void loadModelFromUrl();
    }
});
globalThis.loadModelFromUrl = loadModelFromUrl;
globalThis.loadModelFromFile = loadModelFromFile;
globalThis.send = send;
