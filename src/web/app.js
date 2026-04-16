marked.setOptions({ breaks: true, gfm: true });

var chat = document.getElementById('chat');
var inp = document.getElementById('msg');
var sendBtn = document.getElementById('send-btn');
var stopBtn = document.getElementById('stop-btn');
var modelName = '', abortCtrl = null, isStreaming = false, autoScroll = true, renderTimer = null;
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
}).catch(function() { document.getElementById('model-name').textContent = 'offline'; });

// Load system prompt from localStorage
var savedSystemPrompt = localStorage.getItem('agave_system_prompt');
if (savedSystemPrompt) document.getElementById('system-prompt').value = savedSystemPrompt;
document.getElementById('system-prompt').addEventListener('input', function() {
  localStorage.setItem('agave_system_prompt', this.value);
});

// Persist and restore sampling settings
var tempEl = document.getElementById('temperature');
var topPEl = document.getElementById('top-p');
var maxTokEl = document.getElementById('max-tokens');
var savedTemp = localStorage.getItem('agave_temperature');
var savedTopP = localStorage.getItem('agave_top_p');
var savedMaxTok = localStorage.getItem('agave_max_tokens');
if (savedTemp !== null) { tempEl.value = savedTemp; document.getElementById('temp-val').textContent = parseFloat(savedTemp).toFixed(1); }
if (savedTopP !== null) { topPEl.value = savedTopP; document.getElementById('topp-val').textContent = parseFloat(savedTopP).toFixed(2); }
if (savedMaxTok !== null) { maxTokEl.value = savedMaxTok; }

tempEl.addEventListener('input', function() {
  document.getElementById('temp-val').textContent = parseFloat(this.value).toFixed(1);
  localStorage.setItem('agave_temperature', this.value);
});
topPEl.addEventListener('input', function() {
  document.getElementById('topp-val').textContent = parseFloat(this.value).toFixed(2);
  localStorage.setItem('agave_top_p', this.value);
});
maxTokEl.addEventListener('input', function() {
  localStorage.setItem('agave_max_tokens', this.value);
});
maxTokEl.addEventListener('blur', function() {
  var v = parseInt(this.value);
  if (isNaN(v) || v < 1) this.value = 1;
  else if (v > 4096) this.value = 4096;
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
  var timeout = isError ? 12000 : 5000;
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    setTimeout(function() { if (toast.parentNode) toast.remove(); }, timeout);
  }
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
  reader.readAsDataURL(file);
  return true;
}

function onImageSelect(e) {
  var file = e.target.files[0]; if (!file) return;
  if (!loadImageFile(file, 'Image attached')) e.target.value = '';
}

function removeImage() {
  pendingImage = null;
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
chatForm.addEventListener('dragleave', function(e) { e.preventDefault(); chatForm.classList.remove('drag-over'); });
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
  var tc = document.getElementById('toks-counter');
  if (s) {
    streamTokenCount = 0; streamStartTime = performance.now();
    tc.textContent = '0.0 tok/s'; tc.classList.add('visible');
  } else {
    tc.classList.remove('visible');
  }
}

function updateToksCounter() {
  var tc = document.getElementById('toks-counter');
  if (!isStreaming) return;
  var elapsed = (performance.now() - streamStartTime) / 1000;
  if (elapsed > 0) tc.textContent = (streamTokenCount / elapsed).toFixed(1) + ' tok/s';
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
  btn.classList.toggle('active', open);
  btn.setAttribute('aria-expanded', open);
}

function clearSystemPrompt() {
  var el = document.getElementById('system-prompt');
  el.value = '';
  localStorage.removeItem('agave_system_prompt');
}

function updateCtxBadge(modelData) {
  var badge = document.getElementById('ctx-badge');
  if (!modelData) return;
  var used = modelData.kv_seq_len || 0;
  var max = modelData.ctx_size || 0;
  if (max <= 0) return;
  var fmtNum = function(n) { return n >= 1024 ? Math.round(n / 1024) + 'K' : String(n); };
  var label = fmtNum(used) + '/' + fmtNum(max);
  badge.textContent = label;
  badge.setAttribute('aria-label', 'Context: ' + used + ' of ' + max + ' tokens used');
  badge.classList.add('visible');
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
  var r = document.createElement('span'); r.className = 'role user'; r.textContent = 'You';
  var m = document.createElement('div'); m.className = 'msg user';
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
  var r = document.createElement('span'); r.className = 'role assistant'; r.textContent = 'agave';
  var m = document.createElement('div'); m.className = 'msg assistant thinking';
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
    var c = document.createElement('button'); c.className = 'copy-btn'; c.textContent = 'Copy';
    c.onclick = function() {
      navigator.clipboard.writeText(b.textContent).then(function() {
        c.textContent = 'Copied!';
        setTimeout(function() { c.textContent = 'Copy'; }, 2000);
      }).catch(function() { c.textContent = 'Failed'; setTimeout(function() { c.textContent = 'Copy'; }, 2000); });
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
  if (renderTimer && !final) return;
  var doRender = function() {
    el.classList.remove('thinking');
    el.textContent = '';
    var dc = content.replace(/<think>([\s\S]*?)<\/think>\s*/g, function(_, p) {
      var t = p.trim();
      return t ? '\n> ' + t.replace(/\n/g, '\n> ') + '\n\n' : '';
    });
    if (dc.indexOf('<think>') === 0) dc = dc.substring(7);
    var parsed;
    if (typeof marked !== 'undefined') {
      try { parsed = marked.parse(dc); } catch(e) { parsed = dc.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>'); }
    } else {
      parsed = dc.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    }
    var sanitized;
    if (typeof DOMPurify !== 'undefined') {
      sanitized = DOMPurify.sanitize(parsed);
    } else if (typeof marked !== 'undefined') {
      // marked produced HTML but DOMPurify is missing — escape to prevent XSS (breaks formatting)
      sanitized = parsed.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    } else {
      // Both missing — parsed is already escaped, use as-is
      sanitized = parsed;
    }
    var container = document.createElement('div'); container.innerHTML = sanitized;
    while (container.firstChild) el.appendChild(container.firstChild);
    processCode(el);
    el.querySelectorAll('a[href]').forEach(function(a) {
      var h = a.getAttribute('href');
      if (h && h.charAt(0) !== '#') { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
    });
    if (final) {
      el.setAttribute('data-content', content);
      var cb = document.createElement('button'); cb.className = 'msg-copy'; cb.textContent = 'Copy';
      cb.onclick = function() {
        navigator.clipboard.writeText(content).then(function() {
          cb.textContent = 'Copied!';
          setTimeout(function() { cb.textContent = 'Copy'; }, 2000);
        }).catch(function() { cb.textContent = 'Failed'; setTimeout(function() { cb.textContent = 'Copy'; }, 2000); });
      };
      el.appendChild(cb);
      var plain = el.textContent.substring(0, 200);
      announceToSR('Agave responded: ' + plain + (content.length > 200 ? '...' : ''));
    }
    scrollBottom();
    renderTimer = null;
  };
  if (final) { if (renderTimer) clearTimeout(renderTimer); doRender(); }
  else { renderTimer = setTimeout(doRender, 60); }
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
  d.appendChild(mkStat('decode ', s.tokens + ' tok @ ' + s.tps, 'tok/s'));
  if (s.pfTok && s.pfTok !== '0') d.appendChild(mkStat('prefill ', s.pfTok + ' tok @ ' + s.pfTps, 'tok/s'));
  if (s.pfMs && s.pfMs !== '0') d.appendChild(mkStat('TTFT ', s.pfMs, 'ms'));
  d.appendChild(mkStat('total ', String(total), 'ms'));
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
    if (!resp.ok) throw new Error('Server error: ' + resp.status);
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
    }
  })
  .finally(function() { abortCtrl = null; setStreaming(false); refreshCtxBadge(); inp.focus(); });
}

function sendMessage(text) {
  var body = 'message=' + encodeURIComponent(text) + '&stream=1' + getSamplingParams() + getSystemParam();
  if (pendingImage) body += '&image=' + encodeURIComponent(pendingImage);
  streamResponse(body, 'Failed to get response');
  if (pendingImage) removeImage();
}

function addRegenBtn(msgEl) {
  var oldBtns = chat.querySelectorAll('.regen-btn');
  for (var i = 0; i < oldBtns.length; i++) oldBtns[i].remove();
  var wrap = msgEl.closest('.msg-wrap');
  if (!wrap || !wrap.classList.contains('assistant')) return;
  var btn = document.createElement('button');
  btn.className = 'regen-btn';
  btn.textContent = '\u21BB Regenerate';
  btn.setAttribute('aria-label', 'Regenerate response');
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
    renderContent(el, '**Commands:**\n- `/clear` / `/reset` \u2014 Clear conversation and KV cache\n- `/stats` \u2014 Toggle generation statistics\n- `/context` / `/ctx` \u2014 Show context window usage\n- `/model` \u2014 Show model name\n- `/help` \u2014 Show this help\n\n**Shortcuts:**\n- `Enter` \u2014 Send message\n- `Shift+Enter` \u2014 New line\n- `Escape` \u2014 Stop generation or close dialog\n\nUse the \u2699 settings panel to configure temperature, top-p, max tokens, and system prompt.', true);
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
        var pct = max > 0 ? (used / max * 100).toFixed(1) : '0.0';
        renderContent(el2, 'Context: **' + used + ' / ' + max + '** tokens (' + pct + '% used)', true);
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
  var icon = document.createElement('div'); icon.className = 'icon'; icon.textContent = '\uD83C\uDF35';
  var h2 = document.createElement('h2'); h2.textContent = 'agave';
  var p = document.createElement('p'); p.textContent = 'High-performance LLM inference engine';
  var hints = document.createElement('div'); hints.className = 'hints';
  ['Type a message to start', '/help for commands', 'Shift+Enter for new line'].forEach(function(t) {
    var isHelp = t === '/help for commands';
    var s = document.createElement(isHelp ? 'button' : 'span');
    s.className = 'hint'; s.textContent = t;
    if (isHelp) { s.type = 'button'; s.onclick = function() { handleCommand('/help'); }; }
    hints.appendChild(s);
  });
  empty.appendChild(icon); empty.appendChild(h2); empty.appendChild(p); empty.appendChild(hints);
  chat.appendChild(empty);
}

function clearChat() {
  if (!confirm('Clear this conversation?')) return;
  if (pendingImage) removeImage();
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=%2Fclear' })
  .then(function() { loadConvs(); showEmpty(); inp.focus(); }).catch(function() { showEmpty(); inp.focus(); });
}

function toggleSidebar() {
  var sb = document.getElementById('sidebar'), btn = document.getElementById('menu-btn');
  var isOpen = sb.classList.toggle('open');
  document.getElementById('sidebar-overlay').classList.toggle('show');
  if (btn) btn.setAttribute('aria-expanded', isOpen);
  if (isOpen) {
    var firstBtn = sb.querySelector('.new-chat-btn');
    if (firstBtn) firstBtn.focus();
    sb._trapFocus = function(e) {
      if (e.key !== 'Tab') return;
      var focusable = sb.querySelectorAll('button, [href], [tabindex]:not([tabindex="-1"])');
      if (!focusable.length) return;
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey) { if (document.activeElement === first) { e.preventDefault(); last.focus(); } }
      else { if (document.activeElement === last) { e.preventDefault(); first.focus(); } }
    };
    sb.addEventListener('keydown', sb._trapFocus);
  } else {
    if (sb._trapFocus) { sb.removeEventListener('keydown', sb._trapFocus); sb._trapFocus = null; }
    if (btn && btn.offsetParent !== null) btn.focus();
  }
}

function loadConvs() {
  fetch('/v1/conversations').then(function(r) { return r.json(); }).then(function(convs) {
    var list = document.getElementById('conv-list');
    while (list.firstChild) list.removeChild(list.firstChild);
    if (!convs.length) {
      var em = document.createElement('div'); em.className = 'conv-empty'; em.textContent = 'No conversations yet';
      list.appendChild(em); return;
    }
    convs.forEach(function(c) {
      var item = document.createElement('div'); item.className = 'conv-item' + (c.active ? ' active' : '');
      item.tabIndex = 0; item.setAttribute('role', 'button'); item.setAttribute('aria-label', c.title || 'New chat');
      if (c.active) item.setAttribute('aria-current', 'true');
      item.onclick = function() { selectConv(c.id); };
      item.onkeydown = function(e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectConv(c.id); } };
      var title = document.createElement('span'); title.className = 'conv-title'; title.textContent = c.title || 'New chat';
      if (c.title) title.title = c.title;
      var del = document.createElement('button'); del.className = 'conv-del'; del.textContent = '\u00d7';
      del.setAttribute('aria-label', 'Delete conversation: ' + (c.title || 'New chat'));
      del.onclick = function(e) { e.stopPropagation(); deleteConv(c.id); };
      item.appendChild(title); item.appendChild(del); list.appendChild(item);
    });
  }).catch(function() {
    var list = document.getElementById('conv-list');
    while (list.firstChild) list.removeChild(list.firstChild);
    var em = document.createElement('div'); em.className = 'conv-empty'; em.textContent = 'Could not load conversations';
    list.appendChild(em);
  });
}

function newConv() {
  if (pendingImage) removeImage();
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=new' })
  .then(function() { loadConvs(); showEmpty(); inp.focus(); }).catch(function() { loadConvs(); });
}

var selectSeq = 0;
function selectConv(id) {
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
    var errMsg = 'Failed to load conversation. Check that the server is running.';
    var err = document.createElement('div'); err.className = 'error-msg';
    err.setAttribute('role', 'alert');
    err.textContent = errMsg;
    chat.appendChild(err); scrollBottom();
    announceToSR(errMsg);
  });
}

function deleteConv(id) {
  if (!confirm('Delete this conversation?')) return;
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=delete&id=' + encodeURIComponent(id) })
  .then(function(r) { return r.json(); }).then(function(data) {
    loadConvs(); if (data.cleared) showEmpty(); inp.focus();
  }).catch(function() {
    var errMsg = 'Failed to delete conversation.';
    var err = document.createElement('div'); err.className = 'error-msg';
    err.setAttribute('role', 'alert');
    err.textContent = errMsg;
    chat.appendChild(err); scrollBottom();
    announceToSR(errMsg);
  });
}

function showInfo() {
  var m = document.getElementById('info-modal'); m.classList.add('show');
  document.getElementById('info-model').textContent = modelName || '-';
  document.getElementById('info-backend').textContent = backendName || '-';
  var cb = m.querySelector('.modal-close'); if (cb) cb.focus();
  m._trapFocus = function(e) {
    if (e.key !== 'Tab') return;
    var focusable = m.querySelectorAll('button, [href], [tabindex]:not([tabindex="-1"])');
    if (!focusable.length) return;
    var first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey) { if (document.activeElement === first) { e.preventDefault(); last.focus(); } }
    else { if (document.activeElement === last) { e.preventDefault(); first.focus(); } }
  };
  m.addEventListener('keydown', m._trapFocus);
}

function hideInfo() {
  var m = document.getElementById('info-modal'); m.classList.remove('show');
  if (m._trapFocus) { m.removeEventListener('keydown', m._trapFocus); m._trapFocus = null; }
  inp.focus();
}

loadConvs();
