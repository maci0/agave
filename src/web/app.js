// Server chat UI (embedded by src/server/server.zig). Not the WASM browser shell in web/.
marked.setOptions({ breaks: true, gfm: true });

/** Truncate by Unicode code points so surrogate pairs (emoji, some CJK) are not split. */
function truncateAnnounce(text, maxChars) {
  var chars = Array.from(text);
  if (chars.length <= maxChars) return text;
  return chars.slice(0, maxChars).join('') + '...';
}

/** Locale-aware fixed-fraction number for tok/s, percents, and similar UI values. */
function fmtNum(n, digits) {
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Locale-aware integer for token counts and millisecond totals. */
function fmtInt(n) {
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

var chat = document.getElementById('chat');
var inp = document.getElementById('msg');
var sendBtn = document.getElementById('send-btn');
var stopBtn = document.getElementById('stop-btn');
var modelName = '', abortCtrl = null, isStreaming = false, autoScroll = true, renderTimer = null;
/** Latest stream paint target — updated on every token so the throttled flush shows current text, not the stale closure from schedule time. */
var pendingStreamRender = null;
var msgRoleIdSeq = 0;
sendBtn.disabled = true;
var backendName = '';
var streamTokenCount = 0, streamStartTime = 0;

fetch('/v1/models').then(function(r) { return r.json(); }).then(function(d) {
  if (d.data && d.data[0]) {
    modelName = d.data[0].id;
    backendName = d.data[0].backend || '';
    var badge = document.getElementById('model-name');
    badge.textContent = modelName;
    badge.title = modelName;
    updateCtxBadge(d.data[0]);
  }
}).catch(function() { setOfflineBadge(); });

function setOfflineBadge() {
  var badge = document.getElementById('model-name');
  // Prefer a native button over role="button" on a live region span (4.1.2)
  var btn = badge;
  if (badge.tagName !== 'BUTTON') {
    btn = document.createElement('button');
    btn.type = 'button';
    btn.id = 'model-name';
    btn.className = 'model-badge';
    btn.setAttribute('aria-live', 'polite');
    badge.replaceWith(btn);
  }
  btn.textContent = 'offline - click to retry';
  btn.setAttribute('aria-label', 'Offline. Activate to retry connection');
  btn.onclick = function() {
    var loading = document.createElement('span');
    loading.id = 'model-name';
    loading.className = 'model-badge';
    loading.setAttribute('aria-live', 'polite');
    loading.textContent = 'Loading…';
    btn.replaceWith(loading);
    fetch('/v1/models').then(function(r) { return r.json(); }).then(function(d) {
      var el = document.getElementById('model-name');
      if (d.data && d.data[0]) {
        modelName = d.data[0].id;
        backendName = d.data[0].backend || '';
        el.textContent = modelName;
        el.title = modelName;
        updateCtxBadge(d.data[0]);
      } else { setOfflineBadge(); }
    }).catch(function() { setOfflineBadge(); });
  };
}

// System prompt: sessionStorage only (tab lifetime). Migrate away from older
// localStorage key so sensitive prompt text does not persist across sessions.
var savedSystemPrompt = sessionStorage.getItem('agave_system_prompt');
if (savedSystemPrompt === null) {
  var legacySp = localStorage.getItem('agave_system_prompt');
  if (legacySp !== null) {
    savedSystemPrompt = legacySp;
    sessionStorage.setItem('agave_system_prompt', legacySp);
    localStorage.removeItem('agave_system_prompt');
  }
}
if (savedSystemPrompt) document.getElementById('system-prompt').value = savedSystemPrompt;
document.getElementById('system-prompt').addEventListener('input', function() {
  sessionStorage.setItem('agave_system_prompt', this.value);
});

// Persist and restore sampling settings
var tempEl = document.getElementById('temperature');
var topPEl = document.getElementById('top-p');
var maxTokEl = document.getElementById('max-tokens');
var savedTemp = localStorage.getItem('agave_temperature');
var savedTopP = localStorage.getItem('agave_top_p');
var savedMaxTok = localStorage.getItem('agave_max_tokens');
if (savedTemp !== null) { tempEl.value = savedTemp; document.getElementById('temp-val').textContent = fmtNum(parseFloat(savedTemp), 1); }
if (savedTopP !== null) { topPEl.value = savedTopP; document.getElementById('topp-val').textContent = fmtNum(parseFloat(savedTopP), 2); }
if (savedMaxTok !== null) { maxTokEl.value = savedMaxTok; }
tempEl.setAttribute('aria-valuetext', fmtNum(parseFloat(tempEl.value), 1));
topPEl.setAttribute('aria-valuetext', fmtNum(parseFloat(topPEl.value), 2));

tempEl.addEventListener('input', function() {
  document.getElementById('temp-val').textContent = fmtNum(parseFloat(this.value), 1);
  this.setAttribute('aria-valuetext', fmtNum(parseFloat(this.value), 1));
  localStorage.setItem('agave_temperature', this.value);
});
topPEl.addEventListener('input', function() {
  document.getElementById('topp-val').textContent = fmtNum(parseFloat(this.value), 2);
  this.setAttribute('aria-valuetext', fmtNum(parseFloat(this.value), 2));
  localStorage.setItem('agave_top_p', this.value);
});
maxTokEl.addEventListener('input', function() {
  this.removeAttribute('aria-invalid');
  var errEl = document.getElementById('max-tokens-error');
  if (errEl) { errEl.textContent = ''; errEl.hidden = true; }
  this.setAttribute('aria-describedby', 'max-tokens-range');
  localStorage.setItem('agave_max_tokens', this.value);
});
maxTokEl.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') e.preventDefault();
});
maxTokEl.addEventListener('blur', function() {
  var raw = this.value;
  var v = parseInt(raw, 10);
  var clamped = v;
  if (isNaN(v) || v < 1) clamped = 1;
  else if (v > 4096) clamped = 4096;
  var errEl = document.getElementById('max-tokens-error');
  if (String(clamped) !== String(raw).trim() || isNaN(v)) {
    this.value = clamped;
    this.setAttribute('aria-invalid', 'true');
    var msg = 'Max tokens adjusted to ' + clamped + ' (allowed range 1 to 4096)';
    if (errEl) {
      errEl.textContent = msg;
      errEl.hidden = false;
      this.setAttribute('aria-describedby', 'max-tokens-range max-tokens-error');
    } else {
      announceToSR(msg);
    }
  } else {
    this.removeAttribute('aria-invalid');
    if (errEl) { errEl.textContent = ''; errEl.hidden = true; }
    this.setAttribute('aria-describedby', 'max-tokens-range');
  }
  localStorage.setItem('agave_max_tokens', this.value);
});

if (localStorage.getItem('agave_show_stats') === '1') document.body.classList.add('show-stats');

var pendingImage = null;

function showToast(text, type) {
  var isError = type !== 'info';
  var toast = document.createElement('div');
  toast.className = (isError ? 'error-msg' : 'info-msg') + ' toast';
  toast.setAttribute('role', isError ? 'alert' : 'status');
  var span = document.createElement('span');
  span.textContent = text;
  span.style.flex = '1';
  var close = document.createElement('button');
  close.type = 'button';
  close.className = 'toast-dismiss';
  close.textContent = '\u00d7';
  close.setAttribute('aria-label', 'Dismiss');
  close.onclick = function() { toast.remove(); };
  toast.appendChild(span);
  toast.appendChild(close);
  toast.style.maxWidth = 'var(--max-w)';
  toast.style.margin = '8px auto';
  chat.appendChild(toast);
  scrollBottom();
  announceToSR(text);
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var timeout = isError ? 12000 : 5000;
  if (reducedMotion) timeout *= 2;
  var timerId = setTimeout(function() { if (toast.parentNode) toast.remove(); }, timeout);
  toast.addEventListener('mouseenter', function() { clearTimeout(timerId); });
  toast.addEventListener('mouseleave', function() {
    timerId = setTimeout(function() { if (toast.parentNode) toast.remove(); }, timeout);
  });
  toast.addEventListener('focusin', function() { clearTimeout(timerId); });
  toast.addEventListener('focusout', function() {
    timerId = setTimeout(function() { if (toast.parentNode) toast.remove(); }, timeout);
  });
}

function loadImageFile(file, label) {
  var allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
  if (allowedTypes.indexOf(file.type) === -1) {
    showToast('Unsupported image format. Use JPEG, PNG, GIF, or WebP.');
    return false;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('Image too large (max 10 MB).');
    return false;
  }
  var reader = new FileReader();
  reader.onload = function(ev) {
    pendingImage = ev.target.result;
    document.getElementById('img-thumb').src = pendingImage;
    document.getElementById('img-preview').style.display = '';
    sendBtn.disabled = false;
    announceToSR(label);
  };
  reader.onerror = function() {
    showToast('Could not read that image. Try another file.');
  };
  reader.readAsDataURL(file);
  return true;
}

function onImageSelect(e) {
  var file = e.target.files[0]; if (!file) return;
  if (!loadImageFile(file, 'Image attached')) e.target.value = '';
}

function removeImage() {
  pendingImage = null;
  var thumb = document.getElementById('img-thumb');
  thumb.removeAttribute('src');
  thumb.src = '';
  document.getElementById('img-preview').style.display = 'none';
  document.getElementById('img-input').value = '';
  sendBtn.disabled = !inp.value.trim();
  announceToSR('Image removed');
}

function autoResize() {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
  sendBtn.disabled = !inp.value.trim() && !pendingImage;
}

inp.addEventListener('input', autoResize);
inp.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
});

inp.addEventListener('paste', function(e) {
  var items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (var i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image/') === 0) {
      e.preventDefault();
      var file = items[i].getAsFile();
      if (file) loadImageFile(file, 'Image pasted');
      return;
    }
  }
});

var chatForm = document.getElementById('chat-form');
chatForm.addEventListener('dragover', function(e) { e.preventDefault(); chatForm.classList.add('drag-over'); });
chatForm.addEventListener('dragleave', function(e) {
  e.preventDefault();
  // Ignore leave events that stay inside the form (child → child flicker).
  var to = e.relatedTarget;
  if (to && chatForm.contains(to)) return;
  chatForm.classList.remove('drag-over');
});
chatForm.addEventListener('drop', function(e) {
  e.preventDefault();
  chatForm.classList.remove('drag-over');
  var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.indexOf('image/') === 0) loadImageFile(file, 'Image dropped');
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    if (document.getElementById('info-modal').classList.contains('show')) hideInfo();
    else if (document.getElementById('sidebar').classList.contains('open')) toggleSidebar();
    else if (document.getElementById('settings-panel').classList.contains('open')) toggleSettings();
    else if (isStreaming) stopGen();
  }
});

chat.addEventListener('scroll', function() {
  autoScroll = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
});

function scrollBottom() { if (autoScroll) chat.scrollTop = chat.scrollHeight; }

function setStreaming(s) {
  isStreaming = s;
  sendBtn.style.display = s ? 'none' : '';
  stopBtn.style.display = s ? '' : 'none';
  inp.disabled = s;
  var imgBtn = document.getElementById('img-btn');
  if (imgBtn) imgBtn.disabled = s;
  chat.setAttribute('aria-busy', s ? 'true' : 'false');
  var tc = document.getElementById('toks-counter');
  if (s) {
    streamTokenCount = 0; streamStartTime = performance.now();
    tc.textContent = fmtNum(0, 1) + ' tok/s'; tc.classList.add('visible');
    announceToSR('Generating response…');
    // Move focus to Stop so keyboard users can cancel without hunting (2.4.3)
    stopBtn.focus();
  } else {
    tc.classList.remove('visible');
    announceToSR('Response complete.');
  }
}

function updateToksCounter() {
  var tc = document.getElementById('toks-counter');
  if (!isStreaming) return;
  var elapsed = (performance.now() - streamStartTime) / 1000;
  if (elapsed > 0) tc.textContent = fmtNum(streamTokenCount / elapsed, 1) + ' tok/s';
}

function getSamplingParams() {
  var temp = document.getElementById('temperature').value;
  var topP = document.getElementById('top-p').value;
  var maxTok = parseInt(document.getElementById('max-tokens').value);
  if (isNaN(maxTok) || maxTok < 1) maxTok = 1;
  else if (maxTok > 4096) maxTok = 4096;
  return '&temperature=' + encodeURIComponent(temp) +
    '&top_p=' + encodeURIComponent(topP) +
    '&max_tokens=' + encodeURIComponent(maxTok);
}

function getSystemParam() {
  var sp = document.getElementById('system-prompt').value.trim();
  return sp ? '&system=' + encodeURIComponent(sp) : '';
}

function toggleSettings() {
  var panel = document.getElementById('settings-panel');
  var btn = document.getElementById('settings-toggle');
  var open = panel.classList.toggle('open');
  panel.hidden = !open;
  btn.classList.toggle('active', open);
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  btn.setAttribute('aria-label', open ? 'Close sampling settings' : 'Open sampling settings');
  btn.title = open ? 'Close settings' : 'Sampling settings';
  if (open) {
    var first = panel.querySelector('input');
    if (first) first.focus();
  } else {
    btn.focus();
  }
  announceToSR('Settings panel ' + (open ? 'opened' : 'closed'));
}

function clearSystemPrompt() {
  var el = document.getElementById('system-prompt');
  el.value = '';
  sessionStorage.removeItem('agave_system_prompt');
  localStorage.removeItem('agave_system_prompt');
  announceToSR('System prompt cleared');
}

var CTX_WARN_RATIO = 0.85;

function updateCtxBadge(modelData) {
  var badge = document.getElementById('ctx-badge');
  if (!modelData) return;
  var used = modelData.kv_seq_len || 0;
  var max = modelData.ctx_size || 0;
  if (max <= 0) return;
  var fmtCtx = function(n) { return n >= 1024 ? fmtInt(Math.round(n / 1024)) + 'K' : fmtInt(n); };
  var nearFull = used / max >= CTX_WARN_RATIO;
  var label = (nearFull ? '!\u00a0' : '') + fmtCtx(used) + '/' + fmtCtx(max);
  badge.textContent = label;
  badge.classList.toggle('warn', nearFull);
  var fullLabel = nearFull
    ? 'Context nearly full: ' + fmtInt(used) + ' of ' + fmtInt(max) + ' tokens used'
    : 'Context: ' + fmtInt(used) + ' of ' + fmtInt(max) + ' tokens used';
  badge.setAttribute('aria-label', fullLabel);
  badge.title = fullLabel;
  badge.classList.add('visible');
}

/** Map HTTP status codes to short, actionable messages for the chat UI. */
function httpErrorMessage(status) {
  if (status === 400) return 'The request was rejected. Check your message and settings.';
  if (status === 413) return 'Message or image is too large.';
  if (status === 429) return 'The server is busy. Wait a moment and try again.';
  if (status === 503) return 'The model is not ready yet. Try again shortly.';
  if (status >= 500) return 'Something went wrong on the server. Try again.';
  return 'Could not complete the request (error ' + status + ').';
}

function refreshCtxBadge() {
  fetch('/v1/models').then(function(r) { return r.json(); }).then(function(d) {
    if (d.data && d.data[0]) updateCtxBadge(d.data[0]);
  }).catch(function() {});
}

function exportConv() {
  var msgs = chat.querySelectorAll('.msg-wrap');
  if (!msgs.length) { showToast('Nothing to export.', 'info'); return; }
  var md = '';
  msgs.forEach(function(w) {
    var isUser = w.classList.contains('user');
    var role = isUser ? 'User' : 'Assistant';
    var msgEl = w.querySelector('.msg');
    if (!msgEl) return;
    var content = msgEl.getAttribute('data-content') || msgEl.textContent || '';
    md += '## ' + role + '\n\n' + content.trim() + '\n\n';
  });
  var blob = new Blob([md], { type: 'text/markdown' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'agave-chat-' + new Date().toISOString().slice(0, 10) + '.md';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('Conversation exported.', 'info');
}

function addUser(text, imageSrc) {
  var e = document.getElementById('empty'); if (e) e.remove();
  var w = document.createElement('div'); w.className = 'msg-wrap user';
  w.setAttribute('role', 'group');
  var roleId = 'msg-role-' + (++msgRoleIdSeq);
  w.setAttribute('aria-labelledby', roleId);
  var r = document.createElement('span'); r.className = 'role user'; r.id = roleId; r.textContent = 'You';
  var m = document.createElement('div'); m.className = 'msg user'; m.dir = 'auto';
  if (imageSrc) {
    var img = document.createElement('img'); img.className = 'msg-img'; img.src = imageSrc; img.alt = 'Attached image';
    m.appendChild(img);
  }
  var span = document.createElement('span'); span.textContent = text;
  m.appendChild(span);
  m.setAttribute('data-content', text);
  w.appendChild(r); w.appendChild(m); chat.appendChild(w); scrollBottom();
}

function addAssistant() {
  var e = document.getElementById('empty'); if (e) e.remove();
  var w = document.createElement('div'); w.className = 'msg-wrap assistant';
  w.setAttribute('role', 'group');
  var roleId = 'msg-role-' + (++msgRoleIdSeq);
  w.setAttribute('aria-labelledby', roleId);
  var r = document.createElement('span'); r.className = 'role assistant'; r.id = roleId; r.textContent = 'agave';
  var m = document.createElement('div'); m.className = 'msg assistant thinking'; m.dir = 'auto';
  m.textContent = '\u2026';
  w.appendChild(r); w.appendChild(m); chat.appendChild(w); scrollBottom();
  return m;
}

function processCode(el) {
  el.querySelectorAll('pre code').forEach(function(b) {
    hljs.highlightElement(b);
    var pre = b.parentElement, lang = (b.className.match(/language-(\w+)/) || [])[1] || '';
    if (lang) {
      var l = document.createElement('span'); l.className = 'code-lang'; l.textContent = lang;
      pre.appendChild(l);
    }
    var c = document.createElement('button'); c.type = 'button'; c.className = 'copy-btn'; c.textContent = 'Copy';
    c.setAttribute('aria-label', lang ? 'Copy ' + lang + ' code' : 'Copy code');
    c.onclick = function() {
      navigator.clipboard.writeText(b.textContent).then(function() {
        c.textContent = 'Copied!';
        announceToSR('Code copied to clipboard');
        setTimeout(function() { c.textContent = 'Copy'; }, 2000);
      }).catch(function() { c.textContent = 'Failed'; announceToSR('Copy failed'); setTimeout(function() { c.textContent = 'Copy'; }, 2000); });
    };
    pre.appendChild(c);
  });
}

// NOTE: All HTML rendered via innerHTML is sanitized through DOMPurify (loaded in index.html).
// The DOMPurify.sanitize() call strips any script injection from marked.parse() output.
// This is safe because: (1) user input goes through marked.parse() which escapes HTML,
// (2) the result is then passed through DOMPurify.sanitize() before DOM insertion,
// (3) showEmpty() uses hardcoded HTML constants (no user input).
// Fallback: if DOMPurify is missing but marked is present, we escape the HTML marked
// produced (breaks formatting but prevents XSS). If both are missing, the plain-text
// fallback already escapes entities so no second pass is needed.

function announceToSR(text) {
  var sr = document.getElementById('sr-announce');
  if (sr) { sr.textContent = ''; setTimeout(function() { sr.textContent = text; }, 100); }
}

function renderContent(el, content, final) {
  // Streaming: keep pending content fresh and flush at most every 60ms.
  // Closing over schedule-time content dropped later tokens while the timer
  // was armed, so the UI lagged one throttle window behind.
  if (!final) {
    pendingStreamRender = { el: el, content: content };
    if (renderTimer) return;
    renderTimer = setTimeout(function() {
      renderTimer = null;
      var p = pendingStreamRender;
      pendingStreamRender = null;
      if (!p) return;
      p.el.classList.remove('thinking');
      p.el.textContent = p.content;
      scrollBottom();
    }, 60);
    return;
  }
  if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
  pendingStreamRender = null;
  var doRender = function() {
    el.classList.remove('thinking');
    // Full markdown/sanitize runs once on the final chunk.
    el.textContent = '';
    var dc = content;
    if (content.indexOf('<think>') !== -1) {
      var thinkIdx = 0;
      dc = content.replace(/<think>([\s\S]*?)<\/think>\s*/g, function(_, p) {
        var t = p.trim();
        if (!t) return '';
        thinkIdx++;
        var escapedThink = t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
        return '<details class="think-block"><summary>Chain of thought ' + thinkIdx + '</summary><div class="think-content">' + escapedThink + '</div></details>';
      });
      // Unclosed <think>: strip the tag and escape so marked cannot treat the
      // remainder as raw HTML (marked 11 passes HTML through; CWE-79).
      if (dc.indexOf('<think>') === 0) {
        dc = dc.substring(7).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      }
    }
    var parsed;
    if (typeof marked !== 'undefined') {
      try { parsed = marked.parse(dc); } catch(e) { parsed = dc.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>'); }
    } else {
      parsed = dc.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    }
    if (typeof DOMPurify !== 'undefined') {
      // DOMPurify guarantees safe HTML — render as rich content
      var sanitized = DOMPurify.sanitize(parsed, {ADD_TAGS: ['details', 'summary']});
      var container = document.createElement('div'); container.innerHTML = sanitized;
      while (container.firstChild) el.appendChild(container.firstChild);
    } else {
      el.textContent = content;
    }
    processCode(el);
    // Keep page outline intact: chat markdown must not introduce competing h1/h2
    // (page already uses h1 for brand and h2 for sidebar/empty/dialog).
    el.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(function(h) {
      var level = parseInt(h.tagName.charAt(1), 10);
      var next = Math.min(level + 2, 6);
      if (next === level) return;
      var nh = document.createElement('h' + next);
      while (h.firstChild) nh.appendChild(h.firstChild);
      // Do not copy attributes from markdown headings (CWE-79): re-applying
      // attrs after sanitize can reintroduce handlers if a sanitizer gap exists.
      h.parentNode.replaceChild(nh, h);
    });
    el.querySelectorAll('table').forEach(function(t) {
      var wrapper = document.createElement('div');
      wrapper.className = 'table-wrap';
      wrapper.setAttribute('tabindex', '0');
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-label', 'Data table');
      t.parentNode.insertBefore(wrapper, t);
      wrapper.appendChild(t);
      t.querySelectorAll('th').forEach(function(th) {
        if (!th.getAttribute('scope')) th.setAttribute('scope', 'col');
      });
    });
    el.querySelectorAll('a[href]').forEach(function(a) {
      var h = a.getAttribute('href') || '';
      // Block active content schemes that may survive sanitizer gaps (CWE-79).
      var lower = h.trim().toLowerCase();
      var isData = lower.indexOf('data:') === 0;
      var isSafeDataImage = lower.indexOf('data:image/') === 0 && lower.indexOf('data:image/svg') !== 0;
      if (lower.indexOf('javascript:') === 0 || lower.indexOf('vbscript:') === 0 ||
          (isData && !isSafeDataImage)) {
        a.removeAttribute('href');
        return;
      }
      if (h && h.charAt(0) !== '#') {
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        if (!a.querySelector('.sr-only-newtab')) {
          var tip = document.createElement('span');
          tip.className = 'sr-only sr-only-newtab';
          tip.textContent = ' (opens in new tab)';
          a.appendChild(tip);
        }
      }
    });
    el.setAttribute('data-content', content);
    var cb = document.createElement('button'); cb.type = 'button'; cb.className = 'msg-copy'; cb.textContent = 'Copy';
    cb.setAttribute('aria-label', 'Copy response');
    cb.onclick = function() {
      navigator.clipboard.writeText(content).then(function() {
        cb.textContent = 'Copied!';
        announceToSR('Response copied to clipboard');
        setTimeout(function() { cb.textContent = 'Copy'; }, 2000);
      }).catch(function() { cb.textContent = 'Failed'; announceToSR('Copy failed'); setTimeout(function() { cb.textContent = 'Copy'; }, 2000); });
    };
    el.appendChild(cb);
    announceToSR('Agave responded: ' + truncateAnnounce(el.textContent, 200));
    scrollBottom();
  };
  doRender();
}

function mkStat(label, val, unit) {
  var sp = document.createElement('span'); sp.textContent = label + ' ';
  var v = document.createElement('span'); v.className = 'val'; v.textContent = val;
  sp.appendChild(v);
  if (unit) { var u = document.createTextNode(' ' + unit); sp.appendChild(u); }
  return sp;
}

function addStats(el, s) {
  var d = document.createElement('div'); d.className = 'stats';
  var total = parseInt(s.time) + (parseInt(s.pfMs) || 0);
  var tps = fmtNum(parseFloat(s.tps), 2);
  d.appendChild(mkStat('decode ', fmtInt(s.tokens) + ' tok @ ' + tps, 'tok/s'));
  if (s.pfTok && s.pfTok !== '0') {
    d.appendChild(mkStat('prefill ', fmtInt(s.pfTok) + ' tok @ ' + fmtNum(parseFloat(s.pfTps), 1), 'tok/s'));
  }
  if (s.pfMs && s.pfMs !== '0') d.appendChild(mkStat('TTFT ', fmtInt(s.pfMs), 'ms'));
  d.appendChild(mkStat('total ', fmtInt(total), 'ms'));
  el.appendChild(d);
}

function streamResponse(body, errLabel, url) {
  var el = addAssistant();
  setStreaming(true); abortCtrl = new AbortController(); var content = '', finalized = false;
  function finalizeStream() {
    if (finalized) return;
    finalized = true;
    renderContent(el, content || '*(no response)*', true); addRegenBtn(el); loadConvs(); refreshCtxBadge();
  }
  fetch(url || '/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body, signal: abortCtrl.signal })
  .then(function(resp) {
    if (!resp.ok) throw new Error(httpErrorMessage(resp.status));
    var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
    function read() {
      return reader.read().then(function(r) {
        if (r.done) { finalizeStream(); return; }
        buf += decoder.decode(r.value, { stream: true });
        var lines = buf.split('\n'); buf = lines.pop() || '';
        for (var i = 0; i < lines.length; i++) {
          var ln = lines[i];
          if (ln.indexOf('data: ') !== 0) continue;
          var d = ln.substring(6);
          if (d === '[DONE]') { finalizeStream(); return; }
          try {
            var o = JSON.parse(d);
            if (o.t) { content += o.t; streamTokenCount++; updateToksCounter(); renderContent(el, content, false); }
            if (o.done) addStats(el, { tokens: String(o.n), tps: o.tps.toFixed(2), time: String(o.ms), pfTok: String(o.pn), pfMs: String(o.pms), pfTps: o.ptps.toFixed(1) });
          } catch(e) { console.warn('SSE parse:', e); }
        }
        return read();
      });
    }
    return read();
  })
  .catch(function(e) {
    if (e.name === 'AbortError') { renderContent(el, content || '*Stopped*', true); addRegenBtn(el); }
    else {
      var errMsg = errLabel + ': ' + e.message;
      var err = document.createElement('div'); err.className = 'error-msg';
      err.setAttribute('role', 'alert');
      err.textContent = errMsg;
      el.textContent = ''; el.appendChild(err);
      announceToSR(errMsg);
      // Same server path as regenerate: last user turn is already stored when the
      // request reached prep; Retry re-runs from that turn without retyping.
      addRegenBtn(el, 'Retry');
    }
  })
  .finally(function() {
    abortCtrl = null; setStreaming(false); refreshCtxBadge();
    if (!document.getElementById('info-modal').classList.contains('show')) inp.focus();
  });
}

function sendMessage(text) {
  var body = 'message=' + encodeURIComponent(text) + '&stream=1' + getSamplingParams() + getSystemParam();
  if (pendingImage) body += '&image=' + encodeURIComponent(pendingImage);
  streamResponse(body, 'Failed to get response');
  if (pendingImage) removeImage();
}

function addRegenBtn(msgEl, actionLabel) {
  var oldBtns = chat.querySelectorAll('.regen-btn');
  for (var i = 0; i < oldBtns.length; i++) oldBtns[i].remove();
  var wrap = msgEl.closest('.msg-wrap');
  if (!wrap || !wrap.classList.contains('assistant')) return;
  var label = actionLabel || 'Regenerate';
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'regen-btn';
  btn.textContent = '\u21BB ' + label;
  btn.setAttribute('aria-label', label === 'Retry' ? 'Retry generating response' : 'Regenerate response');
  btn.onclick = function() { regenerate(); };
  wrap.appendChild(btn);
}

function regenerate() {
  if (isStreaming) return;
  var wraps = chat.querySelectorAll('.msg-wrap.assistant');
  if (!wraps.length) return;
  wraps[wraps.length - 1].remove();
  streamResponse('stream=1' + getSamplingParams() + getSystemParam(), 'Failed to regenerate', '/v1/chat/regenerate');
}

function handleCommand(cmd) {
  if (cmd === '/help') {
    var el = addAssistant();
    renderContent(el, '**Commands:**\n- `/clear` / `/reset`: Clear conversation and KV cache\n- `/stats`: Toggle generation statistics\n- `/context` / `/ctx`: Show context window usage\n- `/model`: Show model name\n- `/help`: Show this help\n\n**Shortcuts:**\n- `Enter`: Send message\n- `Shift+Enter`: New line\n- `Escape`: Stop generation or close dialog\n\nUse the \u2699 settings panel to configure temperature, top-p, max tokens, and system prompt.', true);
    return;
  }
  if (cmd === '/stats') {
    document.body.classList.toggle('show-stats');
    var on = document.body.classList.contains('show-stats');
    localStorage.setItem('agave_show_stats', on ? '1' : '0');
    var el2 = addAssistant();
    renderContent(el2, 'Statistics ' + (on ? 'enabled' : 'disabled') + '.', true);
    return;
  }
  if (cmd === '/context' || cmd === '/ctx') {
    fetch('/v1/models').then(function(r) { return r.json(); }).then(function(d) {
      var el2 = addAssistant();
      if (d.data && d.data[0]) {
        var used = d.data[0].kv_seq_len || 0;
        var max = d.data[0].ctx_size || 0;
        var pct = max > 0 ? fmtNum(used / max * 100, 1) : fmtNum(0, 1);
        renderContent(el2, 'Context: **' + fmtInt(used) + ' / ' + fmtInt(max) + '** tokens (' + pct + '% used)', true);
      } else {
        renderContent(el2, 'Could not retrieve context info.', true);
      }
    }).catch(function() { var el2 = addAssistant(); renderContent(el2, 'Failed to get context info.', true); });
    return;
  }
  if (cmd === '/model') {
    var el3 = addAssistant();
    renderContent(el3, 'Model: **' + (modelName || 'unknown') + '**', true);
    return;
  }
  if (cmd === '/reset') { clearChat(); return; }
  if (cmd === '/clear') { clearChat(); return; }
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=' + encodeURIComponent(cmd) })
  .then(function(resp) { return resp.text(); }).then(function(responseHtml) {
    var tmp = document.createElement('div');
    // DOMPurify sanitizes HTML; fallback escapes all entities to prevent XSS
    if (typeof DOMPurify !== 'undefined') {
      tmp.innerHTML = DOMPurify.sanitize(responseHtml);
    } else {
      tmp.textContent = responseHtml;
    }
    var msgEl = tmp.querySelector('.msg.assistant'); var msg = msgEl ? msgEl.textContent : 'Done';
    var el3 = addAssistant(); renderContent(el3, msg, true);
  })
  .catch(function() {
    var el4 = addAssistant();
    var err = document.createElement('div'); err.className = 'error-msg';
    err.setAttribute('role', 'alert');
    err.textContent = 'Command failed';
    el4.textContent = ''; el4.appendChild(err);
    announceToSR('Command failed');
  });
}

function onSubmit(e) {
  e.preventDefault();
  var text = inp.value.trim();
  if ((!text && !pendingImage) || isStreaming) return false;
  var imgSrc = pendingImage;
  inp.value = ''; autoResize(); sendBtn.disabled = true; addUser(text || '(image)', imgSrc);
  if (text.charAt(0) === '/') handleCommand(text); else sendMessage(text);
  return false;
}

function stopGen() { if (abortCtrl) abortCtrl.abort(); }

function showEmpty() {
  while (chat.firstChild) chat.removeChild(chat.firstChild);
  var empty = document.createElement('div'); empty.id = 'empty';
  // Hardcoded HTML constant — no user input, safe without sanitization
  var icon = document.createElement('div'); icon.className = 'icon'; icon.setAttribute('aria-hidden', 'true'); icon.textContent = '\uD83C\uDF35';
  var h2 = document.createElement('h2'); h2.textContent = 'Start a conversation';
  var p = document.createElement('p'); p.textContent = 'Type a message below to chat with the model.';
  var hints = document.createElement('div'); hints.className = 'hints';
  ['Type a message to start', '/help for commands', 'Enter to send'].forEach(function(t) {
    var isHelp = t === '/help for commands';
    var s = document.createElement(isHelp ? 'button' : 'span');
    s.className = 'hint'; s.textContent = t;
    if (isHelp) { s.type = 'button'; s.onclick = function() { handleCommand('/help'); }; }
    hints.appendChild(s);
  });
  empty.appendChild(icon); empty.appendChild(h2); empty.appendChild(p); empty.appendChild(hints);
  chat.appendChild(empty);
}

function closeMobileSidebar() {
  var sb = document.getElementById('sidebar');
  if (sb && sb.classList.contains('open') && window.matchMedia('(max-width: 700px)').matches) {
    toggleSidebar();
  }
}

function clearChat() {
  var hasMsgs = !!chat.querySelector('.msg-wrap');
  if (hasMsgs && !confirm('Clear this conversation?')) return;
  if (isStreaming) stopGen();
  if (pendingImage) removeImage();
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=%2Fclear' })
  .then(function() {
    loadConvs(); showEmpty(); closeMobileSidebar(); inp.focus();
    announceToSR('Conversation cleared');
  })
  .catch(function() {
    showEmpty(); closeMobileSidebar(); inp.focus();
    showToast('Could not clear on server. Local view was reset.', 'info');
  });
}

function toggleSidebar() {
  var sb = document.getElementById('sidebar'), btn = document.getElementById('menu-btn');
  var isMobile = window.matchMedia('(max-width: 700px)').matches;
  // Drawer open/close is mobile-only; desktop sidebar stays in the layout.
  if (!isMobile) {
    syncSidebarForViewport();
    return;
  }
  var isOpen = sb.classList.toggle('open');
  document.getElementById('sidebar-overlay').classList.toggle('show', isOpen);
  if (btn) {
    btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    btn.setAttribute('aria-label', isOpen ? 'Close sidebar' : 'Open sidebar');
  }
  if (isOpen) {
    setMobileSidebarModal(true);
    var closeBtn = document.getElementById('sidebar-close');
    if (closeBtn) closeBtn.focus();
    else {
      var firstBtn = sb.querySelector('.new-chat-btn');
      if (firstBtn) firstBtn.focus();
    }
  } else {
    setMobileSidebarModal(false);
    if (btn && btn.offsetParent !== null) btn.focus();
  }
}

/** Apply or clear mobile-drawer modality (inert backdrop + Tab cycle). */
function setMobileSidebarModal(on) {
  var sb = document.getElementById('sidebar');
  var main = document.querySelector('.main');
  var hdr = document.querySelector('header');
  if (main) main.inert = on;
  if (hdr) hdr.inert = on;
  if (!sb) return;
  if (on) {
    if (sb._trapFocus) return;
    sb._trapFocus = function(e) {
      if (e.key !== 'Tab') return;
      var focusable = sb.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])');
      if (!focusable.length) return;
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey) { if (document.activeElement === first) { e.preventDefault(); last.focus(); } }
      else { if (document.activeElement === last) { e.preventDefault(); first.focus(); } }
    };
    sb.addEventListener('keydown', sb._trapFocus);
  } else {
    if (sb._trapFocus) { sb.removeEventListener('keydown', sb._trapFocus); sb._trapFocus = null; }
  }
}

/** If the viewport leaves the mobile drawer breakpoint, drop modality so chat stays usable. */
function syncSidebarForViewport() {
  var sb = document.getElementById('sidebar');
  if (!sb) return;
  var isMobile = window.matchMedia('(max-width: 700px)').matches;
  if (!isMobile) {
    if (sb.classList.contains('open')) {
      sb.classList.remove('open');
      document.getElementById('sidebar-overlay').classList.remove('show');
    }
    setMobileSidebarModal(false);
    var btn = document.getElementById('menu-btn');
    if (btn) {
      btn.setAttribute('aria-expanded', 'false');
      btn.setAttribute('aria-label', 'Open sidebar');
    }
  }
}
window.addEventListener('resize', syncSidebarForViewport);

function loadConvs() {
  fetch('/v1/conversations').then(function(r) { return r.json(); }).then(function(convs) {
    var list = document.getElementById('conv-list');
    while (list.firstChild) list.removeChild(list.firstChild);
    if (!convs.length) {
      list.removeAttribute('role');
      var em = document.createElement('div'); em.className = 'conv-empty';
      em.appendChild(document.createTextNode('No conversations yet.'));
      em.appendChild(document.createElement('br'));
      em.appendChild(document.createTextNode('Use '));
      var strong = document.createElement('strong'); strong.textContent = '+ New';
      em.appendChild(strong);
      em.appendChild(document.createTextNode(' to start.'));
      list.appendChild(em); return;
    }
    list.setAttribute('role', 'list');
    convs.forEach(function(c) {
      var item = document.createElement('div'); item.className = 'conv-item' + (c.active ? ' active' : '');
      item.setAttribute('role', 'listitem');
      var selectBtn = document.createElement('button');
      selectBtn.type = 'button';
      selectBtn.className = 'conv-select';
      selectBtn.setAttribute('aria-label', c.title || 'New chat');
      if (c.active) selectBtn.setAttribute('aria-current', 'true');
      selectBtn.onclick = function() { selectConv(c.id); };
      var title = document.createElement('span'); title.className = 'conv-title'; title.textContent = c.title || 'New chat';
      if (c.title) title.title = c.title;
      selectBtn.appendChild(title);
      var del = document.createElement('button'); del.type = 'button'; del.className = 'conv-del'; del.textContent = '\u00d7';
      del.setAttribute('aria-label', 'Delete conversation: ' + (c.title || 'New chat'));
      del.onclick = function(e) { e.stopPropagation(); deleteConv(c.id); };
      item.appendChild(selectBtn); item.appendChild(del); list.appendChild(item);
    });
  }).catch(function() {
    var list = document.getElementById('conv-list');
    while (list.firstChild) list.removeChild(list.firstChild);
    list.removeAttribute('role');
    var em = document.createElement('div'); em.className = 'conv-empty';
    em.appendChild(document.createTextNode('Could not load conversations.'));
    em.appendChild(document.createElement('br'));
    var retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'conv-retry';
    retry.textContent = 'Retry';
    retry.setAttribute('aria-label', 'Retry loading conversations');
    retry.onclick = function() { loadConvs(); };
    em.appendChild(retry);
    list.appendChild(em);
  });
}

function newConv() {
  if (isStreaming) stopGen();
  if (pendingImage) removeImage();
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=new' })
  .then(function() {
    loadConvs(); showEmpty(); closeMobileSidebar(); inp.focus();
    announceToSR('New conversation started');
  }).catch(function() {
    loadConvs();
    showToast('Could not create a new conversation. Check that the server is running.');
  });
}

var selectSeq = 0;
function selectConv(id) {
  if (isStreaming) stopGen();
  if (pendingImage) removeImage();
  var mySeq = ++selectSeq;
  while (chat.firstChild) chat.removeChild(chat.firstChild);
  var loadEl = addAssistant();
  loadEl.textContent = 'Loading conversation\u2026';
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=select&id=' + encodeURIComponent(id) })
  .then(function(r) { return r.json(); }).then(function(data) {
    if (mySeq !== selectSeq) return;
    while (chat.firstChild) chat.removeChild(chat.firstChild);
    if (!data.messages || !data.messages.length) { showEmpty(); loadConvs(); return; }
    data.messages.forEach(function(m) {
      if (m.role === 'user') { addUser(m.content); }
      else { var el = addAssistant(); renderContent(el, m.content, true); }
    });
    loadConvs(); scrollBottom();
    if (document.getElementById('sidebar').classList.contains('open')) toggleSidebar();
    inp.focus();
  }).catch(function() {
    if (mySeq !== selectSeq) return;
    while (chat.firstChild) chat.removeChild(chat.firstChild);
    var errMsg = 'Failed to load conversation. Check that the server is running.';
    var err = document.createElement('div'); err.className = 'error-msg toast';
    err.setAttribute('role', 'alert');
    var span = document.createElement('span');
    span.style.flex = '1';
    span.textContent = errMsg;
    var retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'conv-retry';
    retry.textContent = 'Retry';
    retry.setAttribute('aria-label', 'Retry loading conversation');
    retry.onclick = function() { selectConv(id); };
    err.appendChild(span);
    err.appendChild(retry);
    chat.appendChild(err); scrollBottom();
    announceToSR(errMsg);
  });
}

function deleteConv(id) {
  if (!confirm('Delete this conversation?')) return;
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=delete&id=' + encodeURIComponent(id) })
  .then(function(r) { return r.json(); }).then(function(data) {
    loadConvs(); if (data.cleared) showEmpty(); inp.focus();
    announceToSR('Conversation deleted');
  }).catch(function() {
    var errMsg = 'Failed to delete conversation.';
    var err = document.createElement('div'); err.className = 'error-msg';
    err.setAttribute('role', 'alert');
    err.textContent = errMsg;
    chat.appendChild(err); scrollBottom();
    announceToSR(errMsg);
  });
}

function setDialogBackdropInert(on) {
  ['chat', 'chat-form', 'sidebar', 'sidebar-overlay'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.inert = on;
  });
  var hdr = document.querySelector('header');
  if (hdr) hdr.inert = on;
  var skip = document.querySelector('.skip-link');
  if (skip) skip.inert = on;
}

var infoTrigger = null;
function showInfo() {
  var m = document.getElementById('info-modal'); m.classList.add('show');
  m.setAttribute('aria-hidden', 'false');
  var dlg = m.querySelector('.modal');
  infoTrigger = document.activeElement;
  document.getElementById('info-model').textContent = modelName || '-';
  document.getElementById('info-backend').textContent = backendName || '-';
  setDialogBackdropInert(true);
  // Focus the dialog container so AT announces the accessible name (aria-labelledby).
  if (dlg) dlg.focus();
  else {
    var cb = m.querySelector('.modal-close'); if (cb) cb.focus();
  }
  m._trapFocus = function(e) {
    if (e.key !== 'Tab') return;
    var root = dlg || m;
    var focusable = root.querySelectorAll('button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (!focusable.length) {
      if (dlg) { e.preventDefault(); dlg.focus(); }
      return;
    }
    var first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first || document.activeElement === dlg) {
        e.preventDefault(); last.focus();
      }
    } else if (document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  };
  m.addEventListener('keydown', m._trapFocus);
}

function hideInfo() {
  var m = document.getElementById('info-modal'); m.classList.remove('show');
  m.setAttribute('aria-hidden', 'true');
  if (m._trapFocus) { m.removeEventListener('keydown', m._trapFocus); m._trapFocus = null; }
  setDialogBackdropInert(false);
  if (infoTrigger && infoTrigger.offsetParent !== null) infoTrigger.focus();
  else inp.focus();
  infoTrigger = null;
}

loadConvs();
