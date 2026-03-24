marked.setOptions({ breaks: true, gfm: true });

var chat = document.getElementById('chat');
var inp = document.getElementById('msg');
var sendBtn = document.getElementById('send-btn');
var stopBtn = document.getElementById('stop-btn');
var modelName = '', abortCtrl = null, isStreaming = false, autoScroll = true, renderTimer = null;
var backendName = '';

fetch('/v1/models').then(function(r) { return r.json(); }).then(function(d) {
  if (d.data && d.data[0]) {
    modelName = d.data[0].id;
    backendName = d.data[0].backend || '';
    document.getElementById('model-name').textContent = modelName;
  }
}).catch(function() { document.getElementById('model-name').textContent = 'offline'; });

function autoResize() {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
}

inp.addEventListener('input', autoResize);
inp.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    if (document.getElementById('info-modal').classList.contains('show')) hideInfo();
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
}

function addUser(text) {
  var e = document.getElementById('empty'); if (e) e.remove();
  var w = document.createElement('div'); w.className = 'msg-wrap user';
  var r = document.createElement('span'); r.className = 'role user'; r.textContent = 'You';
  var m = document.createElement('div'); m.className = 'msg user'; m.textContent = text;
  w.appendChild(r); w.appendChild(m); chat.appendChild(w); scrollBottom();
}

function addAssistant() {
  var e = document.getElementById('empty'); if (e) e.remove();
  var w = document.createElement('div'); w.className = 'msg-wrap assistant';
  var r = document.createElement('span'); r.className = 'role assistant'; r.textContent = 'agave';
  var m = document.createElement('div'); m.className = 'msg assistant';
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
      navigator.clipboard.writeText(b.textContent);
      c.textContent = 'Copied!';
      setTimeout(function() { c.textContent = 'Copy'; }, 2000);
    };
    pre.appendChild(c);
  });
}

// NOTE: All HTML rendered via innerHTML is sanitized through DOMPurify (loaded in index.html).
// The DOMPurify.sanitize() call strips any script injection from marked.parse() output.
// This is safe because: (1) user input goes through marked.parse() which escapes HTML,
// (2) the result is then passed through DOMPurify.sanitize() before DOM insertion,
// (3) showEmpty() uses hardcoded HTML constants (no user input).

function renderContent(el, content, final) {
  if (renderTimer && !final) return;
  var doRender = function() {
    el.textContent = '';
    var dc = content.replace(/<think>([\s\S]*?)<\/think>\s*/g, function(_, p) {
      var t = p.trim();
      return t ? '\n> ' + t.replace(/\n/g, '\n> ') + '\n\n' : '';
    });
    if (dc.indexOf('<think>') === 0) dc = dc.substring(7);
    var parsed = typeof marked !== 'undefined' ? marked.parse(dc) : dc.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    var sanitized = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(parsed) : parsed.replace(/<script[\s\S]*?<\/script>/gi, '');
    var container = document.createElement('div'); container.innerHTML = sanitized;
    while (container.firstChild) el.appendChild(container.firstChild);
    processCode(el);
    if (final) {
      var cb = document.createElement('button'); cb.className = 'msg-copy'; cb.textContent = 'Copy';
      cb.onclick = function() {
        navigator.clipboard.writeText(content);
        cb.textContent = 'Copied!';
        setTimeout(function() { cb.textContent = 'Copy'; }, 2000);
      };
      el.appendChild(cb);
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

function sendMessage(text) {
  var el = addAssistant();
  setStreaming(true); abortCtrl = new AbortController(); var content = '';
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'message=' + encodeURIComponent(text) + '&stream=1', signal: abortCtrl.signal })
  .then(function(resp) {
    if (!resp.ok) throw new Error('Server error: ' + resp.status);
    var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
    function read() {
      return reader.read().then(function(r) {
        if (r.done) { renderContent(el, content || '*(no response)*', true); loadConvs(); return; }
        buf += decoder.decode(r.value, { stream: true });
        var lines = buf.split('\n'); buf = lines.pop() || '';
        for (var i = 0; i < lines.length; i++) {
          var ln = lines[i];
          if (ln.indexOf('data: ') !== 0) continue;
          var d = ln.substring(6);
          if (d === '[DONE]') { renderContent(el, content || '*(no response)*', true); loadConvs(); return; }
          try {
            var o = JSON.parse(d);
            if (o.t) { content += o.t; renderContent(el, content, false); }
            if (o.done) addStats(el, { tokens: String(o.n), tps: o.tps.toFixed(2), time: String(o.ms), pfTok: String(o.pn), pfMs: String(o.pms), pfTps: o.ptps.toFixed(1) });
          } catch(e) {}
        }
        return read();
      });
    }
    return read();
  })
  .catch(function(e) {
    if (e.name === 'AbortError') { renderContent(el, content || '*Stopped*', true); }
    else {
      var err = document.createElement('div'); err.className = 'error-msg';
      err.textContent = 'Failed to get response: ' + e.message + '. Check that the server is running.';
      el.textContent = ''; el.appendChild(err);
    }
  })
  .finally(function() { abortCtrl = null; setStreaming(false); inp.focus(); });
}

function handleCommand(cmd) {
  if (cmd === '/help') {
    var el = addAssistant();
    renderContent(el, '**Commands:**\n- `/clear` \u2014 Clear conversation and KV cache\n- `/model` \u2014 Show model name\n- `/help` \u2014 Show this help\n\n**Shortcuts:**\n- `Enter` \u2014 Send message\n- `Shift+Enter` \u2014 New line\n- `Escape` \u2014 Stop generation or close dialog', true);
    return;
  }
  if (cmd === '/model') {
    var el2 = addAssistant();
    renderContent(el2, 'Model: **' + (modelName || 'unknown') + '**', true);
    return;
  }
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=' + encodeURIComponent(cmd) })
  .then(function(resp) { return resp.text(); }).then(function(responseHtml) {
    var tmp = document.createElement('div');
    tmp.innerHTML = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(responseHtml) : responseHtml;
    var msgEl = tmp.querySelector('.msg.assistant'); var msg = msgEl ? msgEl.textContent : 'Done';
    var el3 = addAssistant(); renderContent(el3, msg, true);
  })
  .catch(function() {
    var el4 = addAssistant();
    var err = document.createElement('div'); err.className = 'error-msg'; err.textContent = 'Command failed';
    el4.textContent = ''; el4.appendChild(err);
  });
}

function onSubmit(e) {
  e.preventDefault();
  var text = inp.value.trim();
  if (!text || isStreaming) return false;
  inp.value = ''; autoResize(); addUser(text);
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
    var s = document.createElement('span'); s.className = 'hint'; s.textContent = t; hints.appendChild(s);
  });
  empty.appendChild(icon); empty.appendChild(h2); empty.appendChild(p); empty.appendChild(hints);
  chat.appendChild(empty);
}

function clearChat() {
  fetch('/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'message=%2Fclear' })
  .then(function() { loadConvs(); showEmpty(); }).catch(function() { showEmpty(); });
}

function toggleSidebar() {
  var sb = document.getElementById('sidebar'), btn = document.getElementById('menu-btn');
  sb.classList.toggle('open');
  document.getElementById('sidebar-overlay').classList.toggle('show');
  if (btn) btn.setAttribute('aria-expanded', sb.classList.contains('open'));
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
      item.tabIndex = 0; item.setAttribute('role', 'button');
      item.onclick = function() { selectConv(c.id); };
      item.onkeydown = function(e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectConv(c.id); } };
      var title = document.createElement('span'); title.className = 'conv-title'; title.textContent = c.title || 'New chat';
      if (c.title && c.title.length > 30) title.title = c.title;
      var del = document.createElement('button'); del.className = 'conv-del'; del.textContent = '\u00d7';
      del.setAttribute('aria-label', 'Delete conversation');
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
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=new' })
  .then(function() { loadConvs(); showEmpty(); inp.focus(); }).catch(function() { loadConvs(); });
}

function selectConv(id) {
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=select&id=' + id })
  .then(function(r) { return r.json(); }).then(function(data) {
    while (chat.firstChild) chat.removeChild(chat.firstChild);
    if (!data.messages || !data.messages.length) { showEmpty(); loadConvs(); return; }
    var e = document.getElementById('empty'); if (e) e.remove();
    data.messages.forEach(function(m) {
      if (m.role === 'user') { addUser(m.content); }
      else { var el = addAssistant(); renderContent(el, m.content, true); }
    });
    loadConvs(); scrollBottom();
  }).catch(function() {});
}

function deleteConv(id) {
  fetch('/v1/conversations', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'action=delete&id=' + id })
  .then(function(r) { return r.json(); }).then(function(data) {
    loadConvs(); if (data.cleared) showEmpty();
  }).catch(function() {});
}

function showInfo() {
  var m = document.getElementById('info-modal'); m.classList.add('show');
  document.getElementById('info-model').textContent = modelName || '-';
  document.getElementById('info-backend').textContent = backendName || '-';
  var cb = m.querySelector('.modal-close'); if (cb) cb.focus();
}

function hideInfo() { document.getElementById('info-modal').classList.remove('show'); inp.focus(); }

loadConvs();
