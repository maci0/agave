"use strict";
// Server chat UI (embedded by src/server/server.zig). Not the WASM browser shell in web/.
// Source of truth: compile with `scripts/build-web.sh` (tsc) to refresh app.js.
marked.setOptions({ breaks: true, gfm: true });
/** Required document query; throws if the chat shell markup is missing a node. */
function qs(sel) {
    const el = document.querySelector(sel);
    if (!el) {
        throw new Error(`missing element ${sel}`);
    }
    return el;
}
/** Truncate by Unicode code points so surrogate pairs (emoji, some CJK) are not split. */
function truncateAnnounce(text, maxChars) {
    const chars = [...text];
    if (chars.length <= maxChars) {
        return text;
    }
    return `${chars.slice(0, maxChars).join('')}...`;
}
/** Locale-aware fixed-fraction number for tok/s, percents, and similar UI values. */
function fmtNum(n, digits) {
    return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
/** Locale-aware integer for token counts and millisecond totals. */
function fmtInt(n) {
    return Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function lastFocusable(list) {
    if (list.length === 0) {
        return undefined;
    }
    const el = list[list.length - 1];
    return el instanceof HTMLElement ? el : undefined;
}
const chat = qs('#chat');
const inp = qs('#msg');
const sendBtn = qs('#send-btn');
const stopBtn = qs('#stop-btn');
let modelName = '';
let abortCtrl = null;
let isStreaming = false;
let autoScroll = true;
let renderTimer = null;
/** Latest stream paint target, updated on every token so the throttled flush shows current text, not the stale closure from schedule time. */
let pendingStreamRender = null;
let msgRoleIdSeq = 0;
sendBtn.disabled = true;
let backendName = '';
let hasVision = false;
let visionKnown = false;
let pendingImage = null;
let streamTokenCount = 0;
let streamStartTime = 0;
// The tok/s counter is a role="status" live region; rewriting it on every token
// Floods screen readers with announcements. Refresh at most once per second.
const TOKS_UPDATE_INTERVAL_MS = 1000;
let lastToksUpdate = 0;
/** Apply model metadata from /v1/models to the header, context badge, and image attach. */
function applyModelInfo(modelData) {
    modelName = modelData.id;
    backendName = modelData.backend ?? '';
    const badge = qs('#model-name');
    badge.textContent = modelName;
    badge.title = modelName;
    updateCtxBadge(modelData);
    setVisionUi(modelData.vision === true);
}
/** Show image attach only when the loaded model has a vision encoder. */
function setVisionUi(on) {
    visionKnown = true;
    hasVision = on;
    const imgBtn = document.getElementById('img-btn');
    if (imgBtn) {
        imgBtn.hidden = !on;
    }
    if (!on && pendingImage) {
        removeImage();
        showToast('This model cannot view images.');
    }
    syncVisionHints();
}
/** Keep the empty-state image hint in sync with whether this model can see images. */
function syncVisionHints() {
    const hints = document.querySelector('#empty .hints');
    if (!hints) {
        return;
    }
    const existing = hints.querySelector('.hint-image');
    if (hasVision) {
        if (!existing) {
            const s = document.createElement('span');
            s.className = 'hint hint-image';
            s.textContent = 'Paste or drop an image';
            hints.append(s);
        }
    }
    else if (existing) {
        existing.remove();
    }
}
/** Map fetch/network failures to short, actionable copy (not the engine's exception text). */
function userFacingError(error) {
    if (error instanceof Error) {
        const lower = error.message.toLowerCase();
        if (lower === 'failed to fetch' || lower === 'load failed' || lower.includes('networkerror')) {
            return 'Could not reach the server. Check that it is still running.';
        }
        if (lower === 'empty response body') {
            return 'The server sent an empty reply. Try again.';
        }
        return error.message;
    }
    return String(error);
}
// oxlint-disable-next-line unicorn/prefer-top-level-await, promise/prefer-await-to-then -- classic script: top-level await requires module semantics
fetch('/v1/models').then(function (r) { return r.json(); }).then(function (d) {
    if (d.data?.[0]) {
        applyModelInfo(d.data[0]);
    }
    else {
        setOfflineBadge();
    }
}).catch(function () { setOfflineBadge(); });
function setOfflineBadge() {
    const imgBtn = document.getElementById('img-btn');
    if (imgBtn && !hasVision) {
        imgBtn.hidden = true;
    }
    const badge = qs('#model-name');
    // Prefer a native button over role="button" on a live region span (4.1.2)
    let btn = badge;
    if (badge.tagName !== 'BUTTON') {
        const next = document.createElement('button');
        next.type = 'button';
        btn = next;
        btn.id = 'model-name';
        btn.className = 'model-badge';
        btn.setAttribute('aria-live', 'polite');
        badge.replaceWith(btn);
    }
    btn.textContent = 'offline - click to retry';
    btn.setAttribute('aria-label', 'Offline. Activate to retry connection');
    btn.addEventListener('click', function () {
        const loading = document.createElement('span');
        loading.id = 'model-name';
        loading.className = 'model-badge';
        loading.setAttribute('aria-live', 'polite');
        loading.textContent = 'Loading…';
        btn.replaceWith(loading);
        fetch('/v1/models').then(function (r) { return r.json(); }).then(function (d) {
            if (d.data?.[0]) {
                applyModelInfo(d.data[0]);
            }
            else {
                setOfflineBadge();
            }
        }).catch(function () { setOfflineBadge(); });
    });
}
// System prompt: sessionStorage only (tab lifetime). Migrate away from older
// LocalStorage key so sensitive prompt text does not persist across sessions.
let savedSystemPrompt = sessionStorage.getItem('agave_system_prompt');
if (savedSystemPrompt === null) {
    // One-time migration of a legacy storage key; not a runtime environment fallback.
    // oxlint-disable-next-line @rikalabs/no-runtime-compat-fallbacks -- deliberate key migration, removed after one release cycle
    const legacySp = localStorage.getItem('agave_system_prompt');
    if (legacySp !== null) {
        savedSystemPrompt = legacySp;
        sessionStorage.setItem('agave_system_prompt', legacySp);
        localStorage.removeItem('agave_system_prompt');
    }
}
if (savedSystemPrompt) {
    qs('#system-prompt').value = savedSystemPrompt;
}
qs('#system-prompt').addEventListener('input', function () {
    sessionStorage.setItem('agave_system_prompt', this.value);
});
// Persist and restore sampling settings
const tempEl = qs('#temperature');
const topPEl = qs('#top-p');
const maxTokEl = qs('#max-tokens');
const savedTemp = localStorage.getItem('agave_temperature');
const savedTopP = localStorage.getItem('agave_top_p');
const savedMaxTok = localStorage.getItem('agave_max_tokens');
if (savedTemp !== null) {
    tempEl.value = savedTemp;
    qs('#temp-val').textContent = fmtNum(Number.parseFloat(savedTemp), 1);
}
if (savedTopP !== null) {
    topPEl.value = savedTopP;
    qs('#topp-val').textContent = fmtNum(Number.parseFloat(savedTopP), 2);
}
if (savedMaxTok !== null) {
    maxTokEl.value = savedMaxTok;
}
tempEl.setAttribute('aria-valuetext', fmtNum(Number.parseFloat(tempEl.value), 1));
topPEl.setAttribute('aria-valuetext', fmtNum(Number.parseFloat(topPEl.value), 2));
tempEl.addEventListener('input', function () {
    qs('#temp-val').textContent = fmtNum(Number.parseFloat(this.value), 1);
    this.setAttribute('aria-valuetext', fmtNum(Number.parseFloat(this.value), 1));
    localStorage.setItem('agave_temperature', this.value);
});
topPEl.addEventListener('input', function () {
    qs('#topp-val').textContent = fmtNum(Number.parseFloat(this.value), 2);
    this.setAttribute('aria-valuetext', fmtNum(Number.parseFloat(this.value), 2));
    localStorage.setItem('agave_top_p', this.value);
});
maxTokEl.addEventListener('input', function () {
    this.removeAttribute('aria-invalid');
    const errEl = qs('#max-tokens-error');
    if (errEl) {
        errEl.textContent = '';
        errEl.hidden = true;
    }
    this.setAttribute('aria-describedby', 'max-tokens-range');
    localStorage.setItem('agave_max_tokens', this.value);
});
maxTokEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
        e.preventDefault();
    }
});
maxTokEl.addEventListener('blur', function () {
    const raw = this.value;
    const v = Number.parseInt(raw, 10);
    let clamped = v;
    if (Number.isNaN(v) || v < 1) {
        clamped = 1;
    }
    else if (v > 4096) {
        clamped = 4096;
    }
    const errEl = qs('#max-tokens-error');
    if (String(clamped) !== String(raw).trim() || Number.isNaN(v)) {
        this.value = String(clamped);
        this.setAttribute('aria-invalid', 'true');
        const msg = `Max tokens adjusted to ${clamped} (allowed range 1 to 4096)`;
        if (errEl) {
            errEl.textContent = msg;
            errEl.hidden = false;
            this.setAttribute('aria-describedby', 'max-tokens-range max-tokens-error');
        }
        else {
            announceToSR(msg);
        }
    }
    else {
        this.removeAttribute('aria-invalid');
        if (errEl) {
            errEl.textContent = '';
            errEl.hidden = true;
        }
        this.setAttribute('aria-describedby', 'max-tokens-range');
    }
    localStorage.setItem('agave_max_tokens', this.value);
});
if (localStorage.getItem('agave_show_stats') === '1') {
    document.body.classList.add('show-stats');
}
function showToast(text, type) {
    const isError = type !== 'info';
    const toast = document.createElement('div');
    toast.className = `${isError ? 'error-msg' : 'info-msg'} toast`;
    toast.setAttribute('role', isError ? 'alert' : 'status');
    const span = document.createElement('span');
    span.textContent = text;
    span.style.flex = '1';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-dismiss';
    close.textContent = '\u00D7';
    close.setAttribute('aria-label', 'Dismiss');
    close.addEventListener('click', function () { toast.remove(); });
    toast.append(span);
    toast.append(close);
    toast.style.maxWidth = 'var(--max-w)';
    toast.style.margin = '8px auto';
    chat.append(toast);
    scrollBottom();
    // No announceToSR() here: the role="alert"/"status" on the toast already
    // Announces it; a second live region would read the same text twice.
    const reducedMotion = globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const baseTimeoutMs = (isError ? 12 : 5) * 1000;
    let timeout = baseTimeoutMs;
    if (reducedMotion) {
        timeout *= 2;
    }
    let timerId = setTimeout(function () { if (toast.parentNode) {
        toast.remove();
    } }, timeout);
    toast.addEventListener('mouseenter', function () { clearTimeout(timerId); });
    toast.addEventListener('mouseleave', function () {
        timerId = setTimeout(function () { if (toast.parentNode) {
            toast.remove();
        } }, timeout);
    });
    toast.addEventListener('focusin', function () { clearTimeout(timerId); });
    toast.addEventListener('focusout', function () {
        timerId = setTimeout(function () { if (toast.parentNode) {
            toast.remove();
        } }, timeout);
    });
}
const MAX_IMAGE_BYTES = 10_485_760;
function loadImageFile(file, label) {
    if (visionKnown && !hasVision) {
        showToast('This model cannot view images.');
        return false;
    }
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Unsupported image format. Use JPEG, PNG, GIF, or WebP.');
        return false;
    }
    if (file.size > MAX_IMAGE_BYTES) {
        showToast('Image too large (max 10 MB).');
        return false;
    }
    const reader = new FileReader();
    reader.addEventListener('load', function (ev) {
        const result = ev.target?.result;
        if (typeof result !== 'string') {
            return;
        }
        pendingImage = result;
        qs('#img-thumb').src = pendingImage;
        qs('#img-preview').style.display = '';
        sendBtn.disabled = false;
        announceToSR(label);
    });
    reader.addEventListener('error', function () {
        showToast('Could not read that image. Try another file.');
    });
    reader.readAsDataURL(file);
    return true;
}
// oxlint-disable-next-line no-unused-vars -- called from HTML onchange
function onImageSelect(e) {
    const input = e.target;
    const [file] = input.files ?? [];
    if (!file) {
        return;
    }
    if (!loadImageFile(file, 'Image attached')) {
        input.value = '';
    }
}
function removeImage() {
    pendingImage = null;
    const thumb = qs('#img-thumb');
    thumb.removeAttribute('src');
    thumb.src = '';
    qs('#img-preview').style.display = 'none';
    qs('#img-input').value = '';
    sendBtn.disabled = !inp.value.trim();
    announceToSR('Image removed');
}
function autoResize() {
    inp.style.height = 'auto';
    inp.style.height = `${Math.min(inp.scrollHeight, 200)}px`;
    sendBtn.disabled = !inp.value.trim() && !pendingImage;
}
inp.addEventListener('input', autoResize);
inp.addEventListener('keydown', function (e) {
    // Ignore Enter while an IME composition is active (CJK input): Enter there
    // Confirms the conversion, it must not send the message.
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        qs('#chat-form').requestSubmit();
    }
});
inp.addEventListener('paste', function (e) {
    const items = e.clipboardData?.items;
    if (!items) {
        return;
    }
    // oxlint-disable-next-line no-plusplus, typescript/prefer-for-of -- DataTransferItemList is not iterable in all engines; index loop required
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
            e.preventDefault();
            const file = items[i].getAsFile();
            if (file) {
                loadImageFile(file, 'Image pasted');
            }
            return;
        }
    }
});
const chatForm = qs('#chat-form');
chatForm.addEventListener('dragover', function (e) {
    e.preventDefault();
    if (!visionKnown || hasVision) {
        chatForm.classList.add('drag-over');
    }
});
chatForm.addEventListener('dragleave', function (e) {
    e.preventDefault();
    // Ignore leave events that stay inside the form (child → child flicker).
    const to = e.relatedTarget;
    if (to instanceof Node && chatForm.contains(to)) {
        return;
    }
    chatForm.classList.remove('drag-over');
});
chatForm.addEventListener('drop', function (e) {
    e.preventDefault();
    chatForm.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file?.type.startsWith('image/')) {
        loadImageFile(file, 'Image dropped');
    }
});
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        if (qs('#info-modal').classList.contains('show')) {
            hideInfo();
        }
        else if (qs('#sidebar').classList.contains('open')) {
            toggleSidebar();
        }
        else if (qs('#settings-panel').classList.contains('open')) {
            toggleSettings();
        }
        else if (isStreaming) {
            stopGen();
        }
    }
});
chat.addEventListener('scroll', function () {
    autoScroll = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
});
function scrollBottom() { if (autoScroll) {
    chat.scrollTop = chat.scrollHeight;
} }
function setStreaming(s) {
    isStreaming = s;
    sendBtn.style.display = s ? 'none' : '';
    stopBtn.style.display = s ? '' : 'none';
    inp.disabled = s;
    const imgBtn = qs('#img-btn');
    if (imgBtn) {
        imgBtn.disabled = s;
    }
    chat.setAttribute('aria-busy', s ? 'true' : 'false');
    const tc = qs('#toks-counter');
    if (s) {
        streamTokenCount = 0;
        streamStartTime = performance.now();
        lastToksUpdate = 0;
        tc.textContent = `${fmtNum(0, 1)} tok/s`;
        tc.classList.add('visible');
        announceToSR('Generating response…');
        // Move focus to Stop so keyboard users can cancel without hunting (2.4.3)
        stopBtn.focus();
    }
    else {
        tc.classList.remove('visible');
        announceToSR('Response complete.');
    }
}
function updateToksCounter() {
    const tc = qs('#toks-counter');
    if (!isStreaming) {
        return;
    }
    const now = performance.now();
    if (now - lastToksUpdate < TOKS_UPDATE_INTERVAL_MS) {
        return;
    }
    lastToksUpdate = now;
    const elapsed = (now - streamStartTime) / 1000;
    if (elapsed > 0) {
        tc.textContent = `${fmtNum(streamTokenCount / elapsed, 1)} tok/s`;
    }
}
function getSamplingParams() {
    const temp = qs('#temperature').value;
    const topP = qs('#top-p').value;
    let maxTok = Number.parseInt(qs('#max-tokens').value, 10);
    if (Number.isNaN(maxTok) || maxTok < 1) {
        maxTok = 1;
    }
    else if (maxTok > 4096) {
        maxTok = 4096;
    }
    return `&temperature=${encodeURIComponent(temp)}&top_p=${encodeURIComponent(topP)}&max_tokens=${encodeURIComponent(maxTok)}`;
}
function getSystemParam() {
    const sp = qs('#system-prompt').value.trim();
    return sp ? `&system=${encodeURIComponent(sp)}` : '';
}
function toggleSettings() {
    const panel = qs('#settings-panel');
    const btn = qs('#settings-toggle');
    const open = panel.classList.toggle('open');
    panel.hidden = !open;
    btn.classList.toggle('active', open);
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.setAttribute('aria-label', open ? 'Close sampling settings' : 'Open sampling settings');
    btn.title = open ? 'Close settings' : 'Sampling settings';
    if (open) {
        const first = panel.querySelector('input');
        if (first) {
            first.focus();
        }
    }
    else {
        btn.focus();
    }
    announceToSR(`Settings panel ${open ? 'opened' : 'closed'}`);
}
// oxlint-disable-next-line no-unused-vars -- called from HTML onclick
function clearSystemPrompt() {
    const el = qs('#system-prompt');
    el.value = '';
    sessionStorage.removeItem('agave_system_prompt');
    localStorage.removeItem('agave_system_prompt');
    announceToSR('System prompt cleared');
}
const CTX_WARN_RATIO = 0.85;
function updateCtxBadge(modelData) {
    const badge = qs('#ctx-badge');
    if (!modelData) {
        return;
    }
    const used = modelData.kv_seq_len ?? 0;
    const max = modelData.ctx_size ?? 0;
    if (max <= 0) {
        return;
    }
    const fmtCtx = function (n) { return n >= 1024 ? `${fmtInt(Math.round(n / 1024))}K` : fmtInt(n); };
    const nearFull = used / max >= CTX_WARN_RATIO;
    const label = `${nearFull ? '!\u00A0' : ''}${fmtCtx(used)}/${fmtCtx(max)}`;
    badge.textContent = label;
    badge.classList.toggle('warn', nearFull);
    const ariaText = `${nearFull ? 'Context nearly full' : 'Context'}: ${fmtInt(used)} of ${fmtInt(max)} tokens used`;
    badge.title = ariaText;
    badge.setAttribute('aria-label', ariaText);
    badge.classList.add('visible');
}
/** Map HTTP status codes to short, actionable messages for the chat UI. */
function httpErrorMessage(status) {
    if (status === 400) {
        return 'The request was rejected. Check your message and settings.';
    }
    if (status === 413) {
        return 'Message or image is too large.';
    }
    if (status === 429) {
        return 'The server is busy. Wait a moment and try again.';
    }
    if (status === 503) {
        return 'The model is not ready yet. Try again shortly.';
    }
    if (status >= 500) {
        return 'Something went wrong on the server. Try again.';
    }
    return `Could not complete the request (error ${status}).`;
}
function refreshCtxBadge() {
    fetch('/v1/models').then(function (r) { return r.json(); }).then(function (d) {
        if (d.data?.[0]) {
            updateCtxBadge(d.data[0]);
        }
    })
        .catch(function () { setOfflineBadge(); });
}
// oxlint-disable-next-line no-unused-vars -- called from HTML onclick
function exportConv() {
    const msgs = chat.querySelectorAll('.msg-wrap');
    if (msgs.length === 0) {
        showToast('Nothing to export.', 'info');
        return;
    }
    let md = '';
    for (const w of msgs) {
        const isUser = w.classList.contains('user');
        const role = isUser ? 'User' : 'Assistant';
        const msgEl = w.querySelector('.msg');
        if (!msgEl) {
            continue;
        }
        // Empty data-content must fall back to rendered text.
        // oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- empty attribute must fall back to rendered text
        const msgHtml = msgEl;
        const content = (msgHtml.dataset.content ?? msgHtml.textContent) || '';
        md += `## ${role}\n\n${content.trim()}\n\n`;
    }
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agave-chat-${new Date().toISOString().slice(0, 10)}.md`;
    document.body.append(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast('Conversation exported.', 'info');
}
function addUser(text, imageSrc) {
    const emptyEl = qs('#empty');
    if (emptyEl) {
        emptyEl.remove();
    }
    const w = document.createElement('div');
    w.className = 'msg-wrap user';
    w.setAttribute('role', 'group');
    msgRoleIdSeq += 1;
    const roleId = `msg-role-${msgRoleIdSeq}`;
    const r = document.createElement('span');
    r.className = 'role user';
    r.id = roleId;
    r.textContent = 'You';
    w.setAttribute('aria-labelledby', roleId);
    const m = document.createElement('div');
    m.className = 'msg user';
    m.dir = 'auto';
    if (imageSrc) {
        const img = document.createElement('img');
        img.className = 'msg-img';
        img.src = imageSrc;
        img.alt = 'Attached image';
        m.append(img);
    }
    const span = document.createElement('span');
    span.textContent = text;
    m.append(span);
    m.dataset.content = text;
    w.append(r);
    w.append(m);
    chat.append(w);
    scrollBottom();
}
function addAssistant() {
    const emptyEl = qs('#empty');
    if (emptyEl) {
        emptyEl.remove();
    }
    const w = document.createElement('div');
    w.className = 'msg-wrap assistant';
    w.setAttribute('role', 'group');
    msgRoleIdSeq += 1;
    const roleId = `msg-role-${msgRoleIdSeq}`;
    const r = document.createElement('span');
    r.className = 'role assistant';
    r.id = roleId;
    r.textContent = 'agave';
    w.setAttribute('aria-labelledby', roleId);
    const m = document.createElement('div');
    m.className = 'msg assistant thinking';
    m.dir = 'auto';
    m.textContent = '\u2026';
    w.append(r);
    w.append(m);
    chat.append(w);
    scrollBottom();
    return m;
}
function highlightCodeBlocks(el) {
    for (const b of el.querySelectorAll('pre code')) {
        hljs?.highlightElement(b);
        const pre = b.parentElement;
        const lang = (b.className.match(/language-(\w+)/) ?? [])[1] ?? '';
        if (lang) {
            const l = document.createElement('span');
            l.className = 'code-lang';
            l.textContent = lang;
            pre?.append(l);
        }
        const c = document.createElement('button');
        c.type = 'button';
        c.className = 'copy-btn';
        c.textContent = 'Copy';
        c.setAttribute('aria-label', lang ? `Copy ${lang} code` : 'Copy code');
        c.addEventListener('click', function () {
            navigator.clipboard.writeText(b.textContent ?? '').then(function () {
                c.textContent = 'Copied!';
                announceToSR('Code copied to clipboard');
                setTimeout(function () { c.textContent = 'Copy'; }, 2000);
            }).catch(function () { c.textContent = 'Failed'; announceToSR('Copy failed'); setTimeout(function () { c.textContent = 'Copy'; }, 2000); });
        });
        pre?.append(c);
    }
}
// NOTE: All HTML rendered via innerHTML is sanitized through DOMPurify (loaded in index.html).
// The DOMPurify.sanitize() call strips any script injection from marked.parse() output.
// This is safe because: (1) user input goes through marked.parse() which escapes HTML,
// (2) the result is then passed through DOMPurify.sanitize() before DOM insertion,
// (3) showEmpty() uses hardcoded HTML constants (no user input).
// Fallback: if DOMPurify is missing but marked is present, we escape the HTML marked
// Produced (breaks formatting but prevents XSS). If both are missing, the plain-text
// Fallback already escapes entities so no second pass is needed.
function announceToSR(text) {
    const sr = qs('#sr-announce');
    if (sr) {
        sr.textContent = '';
        setTimeout(function () { sr.textContent = text; }, 100);
    }
}
function escapeHtmlEntities(text) {
    return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}
function escapeHtmlText(text) {
    return escapeHtmlEntities(text).replaceAll('\n', '<br>');
}
function markdownToHtml(dc) {
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- optional CDN dependency feature detection
    if (typeof marked === 'undefined') {
        return escapeHtmlText(dc);
    }
    try {
        return marked.parse(dc);
    }
    catch { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- fall back to escaped plain text instead of killing the response render
        return escapeHtmlText(dc);
    }
}
function demoteHeadings(el) {
    for (const h of el.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
        const level = Number.parseInt(h.tagName.charAt(1), 10);
        const next = Math.min(level + 2, 6);
        if (next === level) {
            continue;
        }
        const nh = document.createElement(`h${next}`);
        while (h.firstChild) {
            nh.append(h.firstChild);
        }
        // Do not copy attributes from markdown headings (CWE-79).
        // Re-applying them could reintroduce event handlers if a sanitizer gap exists.
        h.parentNode?.replaceChild(nh, h);
    }
}
function wrapTables(el) {
    for (const t of el.querySelectorAll('table')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'table-wrap';
        wrapper.setAttribute('tabindex', '0');
        wrapper.setAttribute('role', 'region');
        wrapper.setAttribute('aria-label', 'Data table');
        t.parentNode?.insertBefore(wrapper, t);
        wrapper.append(t);
        for (const th of t.querySelectorAll('th')) {
            if (!th.getAttribute('scope')) {
                th.setAttribute('scope', 'col');
            }
        }
    }
}
/** Neutralize active-content URL schemes that may survive sanitizer gaps (CWE-79). */
function hardenLinks(el) {
    for (const a of el.querySelectorAll('a[href]')) {
        const href = a.getAttribute('href') ?? '';
        const lower = href.trim().toLowerCase();
        const isData = lower.startsWith('data:');
        const isSafeDataImage = lower.startsWith('data:image/') && !lower.startsWith('data:image/svg');
        if (lower.startsWith('javascript:') || lower.startsWith('vbscript:') || // oxlint-disable-line no-script-url -- scheme blocklist, not a script URL assignment
            (isData && !isSafeDataImage)) {
            a.removeAttribute('href');
            continue;
        }
        if (href && href.charAt(0) !== '#') {
            const link = a;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            if (!a.querySelector('.sr-only-newtab')) {
                const tip = document.createElement('span');
                tip.className = 'sr-only sr-only-newtab';
                tip.textContent = ' (opens in new tab)';
                a.append(tip);
            }
        }
    }
}
function attachCopyButton(el, content) {
    const cb = document.createElement('button');
    cb.type = 'button';
    cb.className = 'msg-copy';
    cb.textContent = 'Copy';
    cb.setAttribute('aria-label', 'Copy response');
    cb.addEventListener('click', function () {
        navigator.clipboard.writeText(content).then(function () {
            cb.textContent = 'Copied!';
            announceToSR('Response copied to clipboard');
            setTimeout(function () { cb.textContent = 'Copy'; }, 2000);
        }).catch(function () { cb.textContent = 'Failed'; announceToSR('Copy failed'); setTimeout(function () { cb.textContent = 'Copy'; }, 2000); });
    });
    el.append(cb);
}
function renderFinal(el, content) {
    el.classList.remove('thinking');
    // Full markdown/sanitize runs once on the final chunk.
    el.textContent = '';
    let dc = content;
    if (content.includes('<think>')) {
        let thinkIdx = 0;
        dc = content.replaceAll(/<think>([\s\S]*?)<\/think>\s*/g, function (_, p) {
            const t = p.trim();
            if (!t) {
                return '';
            }
            thinkIdx += 1;
            const escapedThink = escapeHtmlText(t);
            return `<details class="think-block"><summary>Chain of thought ${thinkIdx}</summary><div class="think-content">${escapedThink}</div></details>`;
        });
        // Unclosed <think>: strip the tag and escape so marked cannot treat the rest
        // As raw HTML (marked 11 passes HTML through; CWE-79).
        if (dc.startsWith('<think>')) {
            dc = escapeHtmlEntities(dc.slice(7));
        }
    }
    const parsed = markdownToHtml(dc);
    // DOMPurify guarantees safe HTML, render as rich content.
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- optional CDN dependency feature detection
    if (DOMPurify) {
        const sanitized = DOMPurify.sanitize(parsed, { ADD_TAGS: ['details', 'summary'] });
        const container = document.createElement('div');
        container.innerHTML = sanitized;
        while (container.firstChild) {
            el.append(container.firstChild);
        }
    }
    else {
        el.textContent = content;
    }
    highlightCodeBlocks(el);
    demoteHeadings(el);
    wrapTables(el);
    hardenLinks(el);
    el.dataset.content = content;
    // Capture before the Copy button is appended so its label is not read out
    // As part of the response announcement.
    const respondedText = truncateAnnounce(el.textContent, 200);
    attachCopyButton(el, content);
    announceToSR(`Agave responded: ${respondedText}`);
    scrollBottom();
}
function renderContent(el, content, final) {
    // Streaming: keep pending content fresh and flush at most every 60ms.
    // Closing over schedule-time content dropped later tokens while the timer
    // Was armed, so the UI lagged one throttle window behind.
    if (!final) {
        pendingStreamRender = { el: el, content: content };
        if (renderTimer) {
            return;
        }
        renderTimer = setTimeout(function () {
            renderTimer = null;
            const p = pendingStreamRender;
            pendingStreamRender = null;
            if (!p) {
                return;
            }
            p.el.classList.remove('thinking');
            p.el.textContent = p.content;
            scrollBottom();
        }, 60);
        return;
    }
    if (renderTimer) {
        clearTimeout(renderTimer);
        renderTimer = null;
    }
    pendingStreamRender = null;
    renderFinal(el, content);
}
function mkStat(label, val, unit) {
    const sp = document.createElement('span');
    sp.textContent = `${label} `;
    const v = document.createElement('span');
    v.className = 'val';
    v.textContent = val;
    sp.append(v);
    if (unit) {
        sp.append(document.createTextNode(` ${unit}`));
    }
    return sp;
}
function addStats(el, s) {
    const d = document.createElement('div');
    d.className = 'stats';
    const total = Number.parseInt(s.time, 10) + (Number.parseInt(s.pfMs, 10) || 0);
    const tps = fmtNum(Number.parseFloat(s.tps), 2);
    d.append(mkStat('decode ', `${fmtInt(s.tokens)} tok @ ${tps}`, 'tok/s'));
    if (s.pfTok && s.pfTok !== '0') {
        d.append(mkStat('prefill ', `${fmtInt(s.pfTok)} tok @ ${fmtNum(Number.parseFloat(s.pfTps), 1)}`, 'tok/s'));
    }
    if (s.pfMs && s.pfMs !== '0') {
        d.append(mkStat('TTFT ', fmtInt(s.pfMs), 'ms'));
    }
    d.append(mkStat('total ', fmtInt(total), 'ms'));
    el.append(d);
}
async function streamResponse(body, errLabel, url) {
    const el = addAssistant();
    setStreaming(true);
    abortCtrl = new AbortController();
    let content = '';
    let finalized = false;
    function finalizeStream() {
        if (finalized) {
            return;
        }
        finalized = true;
        renderContent(el, content || 'No response.', true);
        addRegenBtn(el);
        loadConvs();
        refreshCtxBadge();
    }
    try {
        const resp = await fetch(url ?? '/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: body, signal: abortCtrl.signal });
        if (!resp.ok) {
            throw new Error(httpErrorMessage(resp.status));
        }
        const stream = resp.body;
        if (!stream) {
            throw new Error('empty response body');
        }
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        for (;;) {
            const { done, value } = await reader.read();
            if (done) {
                finalizeStream();
                break;
            }
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop() ?? '';
            for (const ln of lines) {
                if (!ln.startsWith('data: ')) {
                    continue;
                }
                const d = ln.slice(6);
                if (d === '[DONE]') {
                    finalizeStream();
                    return;
                }
                try {
                    const o = JSON.parse(d);
                    if (o.t) {
                        content += o.t;
                        streamTokenCount += 1;
                        updateToksCounter();
                        renderContent(el, content, false);
                    }
                    if (o.done) {
                        addStats(el, { tokens: String(o.n), tps: o.tps.toFixed(2), time: String(o.ms), pfTok: String(o.pn), pfMs: String(o.pms), pfTps: o.ptps.toFixed(1) });
                    }
                }
                catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- one malformed SSE frame must not kill the stream
                    // oxlint-disable-next-line no-console -- stream diagnostics; toasts would spam the UI per token
                    console.warn('SSE parse:', error);
                }
            }
        }
    }
    catch (error) { // oxlint-disable-line @rikalabs/no-silent-catch-fallback -- errors are surfaced to the user as an alert region with a Retry action
        if (error instanceof Error && error.name === 'AbortError') {
            renderContent(el, content || 'Stopped.', true);
            addRegenBtn(el);
        }
        else {
            const errMsg = `${errLabel}: ${userFacingError(error)}`;
            const err = document.createElement('div');
            err.className = 'error-msg';
            err.setAttribute('role', 'alert');
            err.textContent = errMsg;
            el.textContent = '';
            el.append(err);
            announceToSR(errMsg);
            // Same server path as regenerate: last user turn is already stored when the
            // Request reached prep; Retry re-runs from that turn without retyping.
            addRegenBtn(el, 'Retry');
        }
    }
    finally {
        abortCtrl = null;
        setStreaming(false);
        refreshCtxBadge();
        if (!qs('#info-modal').classList.contains('show')) {
            inp.focus();
        }
    }
}
function sendMessage(text) {
    let body = `message=${encodeURIComponent(text)}&stream=1${getSamplingParams()}${getSystemParam()}`;
    if (pendingImage) {
        body += `&image=${encodeURIComponent(pendingImage)}`;
    }
    // Fire-and-forget: the stream paints itself; errors surface in the message area.
    void streamResponse(body, 'Failed to get response');
    if (pendingImage) {
        removeImage();
    }
}
function addRegenBtn(msgEl, actionLabel) {
    const oldBtns = chat.querySelectorAll('.regen-btn');
    for (const oldBtn of oldBtns) {
        oldBtn.remove();
    }
    const wrap = msgEl.closest('.msg-wrap');
    if (!wrap?.classList.contains('assistant')) {
        return;
    }
    const label = actionLabel ?? 'Regenerate';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'regen-btn';
    btn.textContent = `\u21BB ${label}`;
    btn.setAttribute('aria-label', label === 'Retry' ? 'Retry generating response' : 'Regenerate response');
    btn.addEventListener('click', function () { regenerate(); });
    wrap.append(btn);
}
function regenerate() {
    if (isStreaming) {
        return;
    }
    const wraps = chat.querySelectorAll('.msg-wrap.assistant');
    if (wraps.length === 0) {
        return;
    }
    wraps[wraps.length - 1]?.remove();
    void streamResponse(`stream=1${getSamplingParams()}${getSystemParam()}`, 'Failed to regenerate', '/v1/chat/regenerate');
}
function runCommand(cmd) {
    if (cmd === '/help') {
        renderContent(addAssistant(), '**Commands:**\n- `/clear` / `/reset`: Clear conversation and KV cache\n- `/stats`: Toggle generation statistics\n- `/context` / `/ctx`: Show context window usage\n- `/model`: Show model name\n- `/help`: Show this help\n\n**Shortcuts:**\n- `Enter`: Send message\n- `Shift+Enter`: New line\n- `Escape`: Stop generation or close dialog\n\nUse the \u2699 settings panel to configure temperature, top-p, max tokens, and system prompt.', true);
        return;
    }
    if (cmd === '/stats') {
        document.body.classList.toggle('show-stats');
        const on = document.body.classList.contains('show-stats');
        localStorage.setItem('agave_show_stats', on ? '1' : '0');
        renderContent(addAssistant(), `Statistics ${on ? 'enabled' : 'disabled'}.`, true);
        return;
    }
    if (cmd === '/context' || cmd === '/ctx') {
        fetch('/v1/models').then(function (r) { return r.json(); }).then(function (d) {
            if (d.data?.[0]) {
                const used = d.data[0].kv_seq_len ?? 0;
                const max = d.data[0].ctx_size ?? 0;
                const pct = max > 0 ? fmtNum(used / max * 100, 1) : fmtNum(0, 1);
                renderContent(addAssistant(), `Context: **${fmtInt(used)} / ${fmtInt(max)}** tokens (${pct}% used)`, true);
            }
            else {
                renderContent(addAssistant(), 'Could not retrieve context info.', true);
            }
        }).catch(function () { renderContent(addAssistant(), 'Failed to get context info.', true); });
        return;
    }
    if (cmd === '/model') {
        renderContent(addAssistant(), `Model: **${modelName || 'unknown'}**`, true);
        return;
    }
    if (cmd === '/reset') {
        clearChat();
        return;
    }
    if (cmd === '/clear') {
        clearChat();
        return;
    }
    // Unknown command: give feedback like the REPL does, instead of silently
    // Sending the "/..." text to the model as a chat message.
    renderContent(addAssistant(), `Unknown command: \`${cmd}\`\n\nType \`/help\` to see the available commands.`, true);
    announceToSR(`Unknown command ${cmd}`);
}
// oxlint-disable-next-line no-unused-vars -- called from HTML onsubmit
function onSubmit(e) {
    e.preventDefault();
    const text = inp.value.trim();
    if ((!text && !pendingImage) || isStreaming) {
        return false;
    }
    const imgSrc = pendingImage;
    inp.value = '';
    autoResize();
    sendBtn.disabled = true;
    // oxlint-disable-next-line unicorn/prefer-nullish-coalescing -- empty string with an attached image must render as "(image)"
    addUser(text || '(image)', imgSrc);
    if (text.startsWith('/')) {
        runCommand(text);
    }
    else {
        sendMessage(text);
    }
    return false;
}
function stopGen() { if (abortCtrl) {
    abortCtrl.abort();
} }
function showEmpty() {
    chat.replaceChildren();
    const empty = document.createElement('div');
    empty.id = 'empty';
    // Hardcoded HTML constant, no user input, safe without sanitization
    const icon = document.createElement('div');
    icon.className = 'icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '\uD83C\uDF35';
    const h2 = document.createElement('h2');
    h2.textContent = 'Start a conversation';
    const p = document.createElement('p');
    p.textContent = 'Type a message below to chat with the model.';
    const hintsEl = document.createElement('div');
    hintsEl.className = 'hints';
    // Skip "type a message" filler: the paragraph above already says it.
    for (const t of ['/help for commands', 'Enter to send']) {
        const isHelp = t === '/help for commands';
        const s = document.createElement(isHelp ? 'button' : 'span');
        s.className = 'hint';
        s.textContent = t;
        if (isHelp && s instanceof HTMLButtonElement) {
            s.type = 'button';
            s.addEventListener('click', function () { runCommand('/help'); });
        }
        hintsEl.append(s);
    }
    if (hasVision) {
        const imgHint = document.createElement('span');
        imgHint.className = 'hint hint-image';
        imgHint.textContent = 'Paste or drop an image';
        hintsEl.append(imgHint);
    }
    empty.append(icon);
    empty.append(h2);
    empty.append(p);
    empty.append(hintsEl);
    chat.append(empty);
}
function closeMobileSidebar() {
    const sb = qs('#sidebar');
    if (sb?.classList.contains('open') && globalThis.matchMedia('(max-width: 700px)').matches) {
        toggleSidebar();
    }
}
function clearChat() {
    // oxlint-disable-next-line no-implicit-coercion -- explicit boolean for the guard read
    const hasMsgs = Boolean(chat.querySelector('.msg-wrap'));
    if (hasMsgs && !confirm('Clear this conversation?')) {
        return;
    } // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    if (isStreaming) {
        stopGen();
    }
    if (pendingImage) {
        removeImage();
    }
    fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=%2Fclear' })
        .then(function () {
        loadConvs();
        showEmpty();
        closeMobileSidebar();
        inp.focus();
        announceToSR('Conversation cleared');
    })
        .catch(function () {
        showEmpty();
        closeMobileSidebar();
        inp.focus();
        showToast('Could not clear on server. Local view was reset.', 'info');
    });
}
function toggleSidebar() {
    const sb = qs('#sidebar');
    const btn = qs('#menu-btn');
    const isMobile = globalThis.matchMedia('(max-width: 700px)').matches;
    // Drawer open/close is mobile-only; desktop sidebar stays in the layout.
    if (!isMobile) {
        syncSidebarForViewport();
        return;
    }
    const isOpen = sb.classList.toggle('open');
    if (btn) {
        btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        btn.setAttribute('aria-label', isOpen ? 'Close sidebar' : 'Open sidebar');
    }
    qs('#sidebar-overlay').classList.toggle('show', isOpen);
    if (isOpen) {
        setMobileSidebarModal(true);
        const closeBtn = qs('#sidebar-close');
        if (closeBtn) {
            closeBtn.focus();
        }
        else {
            const firstBtn = sb.querySelector('.new-chat-btn');
            if (firstBtn instanceof HTMLElement) {
                firstBtn.focus();
            }
        }
    }
    else {
        setMobileSidebarModal(false);
        if (btn && btn.offsetParent !== null) {
            btn.focus();
        }
    }
}
/** Apply or clear mobile-drawer modality (inert backdrop + Tab cycle). */
function setMobileSidebarModal(on) {
    const sb = qs('#sidebar');
    const main = qs('.main');
    const hdr = qs('header');
    if (main) {
        main.inert = on;
    }
    if (hdr) {
        hdr.inert = on;
    }
    if (!sb) {
        return;
    }
    if (on) {
        if (sb._trapFocus) {
            return;
        }
        sb._trapFocus = function (e) {
            if (e.key !== 'Tab') {
                return;
            }
            const focusable = sb.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])');
            if (focusable.length === 0) {
                return;
            }
            const first = focusable[0];
            const last = lastFocusable(focusable);
            if (!(first instanceof HTMLElement) || !last) {
                return;
            }
            if (e.shiftKey) {
                if (document.activeElement === first) {
                    e.preventDefault();
                    last.focus();
                }
            }
            else if (document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        };
        sb.addEventListener('keydown', sb._trapFocus);
    }
    else if (sb._trapFocus) {
        sb.removeEventListener('keydown', sb._trapFocus);
        sb._trapFocus = null;
    }
}
/** If the viewport leaves the mobile drawer breakpoint, drop modality so chat stays usable. */
function syncSidebarForViewport() {
    const sb = qs('#sidebar');
    if (!sb) {
        return;
    }
    const isMobile = globalThis.matchMedia('(max-width: 700px)').matches;
    if (!isMobile) {
        if (sb.classList.contains('open')) {
            sb.classList.remove('open');
            qs('#sidebar-overlay').classList.remove('show');
        }
        setMobileSidebarModal(false);
        const btn = qs('#menu-btn');
        if (btn) {
            btn.setAttribute('aria-expanded', 'false');
            btn.setAttribute('aria-label', 'Open sidebar');
        }
    }
}
window.addEventListener('resize', syncSidebarForViewport);
function loadConvs() {
    fetch('/v1/conversations').then(function (r) { return r.json(); }).then(function (convs) {
        const list = qs('#conv-list');
        list.replaceChildren();
        if (convs.length === 0) {
            list.removeAttribute('role');
            const em = document.createElement('div');
            em.className = 'conv-empty';
            em.append(document.createTextNode('No conversations yet.'));
            em.append(document.createElement('br'));
            em.append(document.createTextNode('Use '));
            const strong = document.createElement('strong');
            strong.textContent = '+ New';
            em.append(strong);
            em.append(document.createTextNode(' to start.'));
            list.append(em);
            return;
        }
        list.setAttribute('role', 'list');
        for (const c of convs) {
            const item = document.createElement('div');
            item.className = `conv-item${c.active ? ' active' : ''}`;
            item.setAttribute('role', 'listitem');
            const selectBtn = document.createElement('button');
            selectBtn.type = 'button';
            selectBtn.className = 'conv-select';
            selectBtn.setAttribute('aria-label', c.title ?? 'New chat');
            if (c.active) {
                selectBtn.setAttribute('aria-current', 'true');
            }
            selectBtn.addEventListener('click', function () { selectConv(c.id); });
            const title = document.createElement('span');
            title.className = 'conv-title';
            title.textContent = c.title ?? 'New chat';
            if (c.title) {
                title.title = c.title;
            }
            selectBtn.append(title);
            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'conv-del';
            del.textContent = '\u00D7';
            del.setAttribute('aria-label', `Delete conversation: ${c.title ?? 'New chat'}`);
            del.addEventListener('click', function (e) { e.stopPropagation(); deleteConv(c.id); });
            item.append(selectBtn);
            item.append(del);
            list.append(item);
        }
    }).catch(function () {
        const list = qs('#conv-list');
        list.replaceChildren();
        list.removeAttribute('role');
        const em = document.createElement('div');
        em.className = 'conv-empty';
        em.append(document.createTextNode('Could not load conversations.'));
        em.append(document.createElement('br'));
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'conv-retry';
        retry.textContent = 'Retry';
        retry.setAttribute('aria-label', 'Retry loading conversations');
        retry.addEventListener('click', function () { loadConvs(); });
        em.append(retry);
        list.append(em);
    });
}
// oxlint-disable-next-line no-unused-vars -- called from HTML onclick
function newConv() {
    if (isStreaming) {
        stopGen();
    }
    if (pendingImage) {
        removeImage();
    }
    fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=new' })
        .then(function () {
        loadConvs();
        showEmpty();
        closeMobileSidebar();
        inp.focus();
        announceToSR('New conversation started');
    }).catch(function () {
        loadConvs();
        showToast('Could not create a new conversation. Check that the server is running.');
    });
}
let selectSeq = 0;
function selectConv(id) {
    if (isStreaming) {
        stopGen();
    }
    if (pendingImage) {
        removeImage();
    }
    selectSeq += 1;
    const mySeq = selectSeq;
    chat.replaceChildren();
    const loadEl = document.createElement('div');
    loadEl.className = 'info-msg';
    loadEl.setAttribute('role', 'status');
    loadEl.textContent = 'Loading conversation\u2026';
    chat.append(loadEl);
    announceToSR('Loading conversation…');
    fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: `action=select&id=${encodeURIComponent(id)}` })
        .then(function (r) { return r.json(); }).then(function (data) {
        if (mySeq !== selectSeq) {
            return;
        }
        chat.replaceChildren();
        if (!data.messages || data.messages.length === 0) {
            showEmpty();
            loadConvs();
            return;
        }
        for (const m of data.messages) {
            if (m.role === 'user') {
                addUser(m.content);
            }
            else {
                renderContent(addAssistant(), m.content, true);
            }
        }
        loadConvs();
        scrollBottom();
        if (qs('#sidebar').classList.contains('open')) {
            toggleSidebar();
        }
        inp.focus();
    }).catch(function () {
        if (mySeq !== selectSeq) {
            return;
        }
        chat.replaceChildren();
        const errMsg = 'Failed to load conversation. Check that the server is running.';
        const err = document.createElement('div');
        err.className = 'error-msg toast';
        err.setAttribute('role', 'alert');
        const span = document.createElement('span');
        span.style.flex = '1';
        span.textContent = errMsg;
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'conv-retry';
        retry.textContent = 'Retry';
        retry.setAttribute('aria-label', 'Retry loading conversation');
        retry.addEventListener('click', function () { selectConv(id); });
        err.append(span);
        err.append(retry);
        chat.append(err);
        scrollBottom();
        announceToSR(errMsg);
    });
}
function deleteConv(id) {
    if (!confirm('Delete this conversation?')) {
        return;
    } // oxlint-disable-line no-alert -- native confirmation dialog is intentional UX
    fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: `action=delete&id=${encodeURIComponent(id)}` })
        .then(function (r) { return r.json(); }).then(function (data) {
        loadConvs();
        if (data.cleared) {
            showEmpty();
        }
        inp.focus();
        announceToSR('Conversation deleted');
    }).catch(function () {
        showToast('Could not delete that conversation. Check that the server is running.');
    });
}
function setDialogBackdropInert(on) {
    for (const id of ['chat', 'chat-form', 'sidebar', 'sidebar-overlay']) {
        const el = qs(`#${id}`);
        if (el) {
            el.inert = on;
        }
    }
    const hdr = qs('header');
    if (hdr) {
        hdr.inert = on;
    }
    const skip = qs('.skip-link');
    if (skip) {
        skip.inert = on;
    }
}
let infoTrigger = null;
// oxlint-disable-next-line no-unused-vars -- called from HTML onclick
function showInfo() {
    const m = qs('#info-modal');
    m.classList.add('show');
    m.setAttribute('aria-hidden', 'false');
    const dlg = m.querySelector('.modal');
    infoTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    qs('#info-model').textContent = modelName || '-';
    qs('#info-backend').textContent = backendName || '-';
    setDialogBackdropInert(true);
    // Focus the dialog container so AT announces the accessible name (aria-labelledby).
    if (dlg instanceof HTMLElement) {
        dlg.focus();
    }
    else {
        const cb = m.querySelector('.modal-close');
        if (cb instanceof HTMLElement) {
            cb.focus();
        }
    }
    m._trapFocus = function (e) {
        if (e.key !== 'Tab') {
            return;
        }
        const root = dlg ?? m;
        const focusable = root.querySelectorAll('button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (focusable.length === 0) {
            if (dlg instanceof HTMLElement) {
                e.preventDefault();
                dlg.focus();
            }
            return;
        }
        const first = focusable[0];
        const last = lastFocusable(focusable);
        if (!(first instanceof HTMLElement) || !last) {
            return;
        }
        if (e.shiftKey) {
            if (document.activeElement === first || document.activeElement === dlg) {
                e.preventDefault();
                last.focus();
            }
        }
        else if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
        }
    };
    m.addEventListener('keydown', m._trapFocus);
}
function hideInfo() {
    const m = qs('#info-modal');
    m.classList.remove('show');
    m.setAttribute('aria-hidden', 'true');
    if (m._trapFocus) {
        m.removeEventListener('keydown', m._trapFocus);
        m._trapFocus = null;
    }
    setDialogBackdropInert(false);
    if (infoTrigger && infoTrigger.offsetParent !== null) {
        infoTrigger.focus();
    }
    else {
        inp.focus();
    }
    infoTrigger = null;
}
loadConvs();
