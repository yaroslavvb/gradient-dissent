'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const entries = [];
  const dateFormatter = new Intl.DateTimeFormat('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
    timeZone: 'America/Los_Angeles', timeZoneName: 'short',
  });

  function linkedId() {
    try { return decodeURIComponent(location.hash.slice(1)); }
    catch { return ''; }
  }

  function attachmentView(attachment) {
    if (!attachment || typeof attachment.url !== 'string') return null;
    let resolved;
    try { resolved = new URL(attachment.url, location.href); }
    catch { return null; }
    // All exported attachments belong to this public snapshot.
    if (resolved.origin !== location.origin || !resolved.pathname.startsWith(new URL('assets/', location.href).pathname)) return null;
    const figure = document.createElement('figure');
    figure.className = 'attachment';
    const type = attachment.mimeType || '';
    const name = attachment.name || 'Attached file';
    let media;
    if (type.startsWith('image/')) {
      media = document.createElement('img');
      media.alt = name;
      media.loading = 'lazy';
      media.decoding = 'async';
    } else if (type.startsWith('video/')) {
      media = document.createElement('video');
      media.controls = true;
      media.loop = true;
      media.playsInline = true;
      media.preload = 'none';
      media.setAttribute('aria-label', name);
    } else if (type.startsWith('audio/')) {
      media = document.createElement('audio');
      media.controls = true;
      media.preload = 'none';
      media.setAttribute('aria-label', name);
    }
    if (media) { media.src = attachment.url; figure.append(media); }
    const caption = document.createElement('figcaption');
    const link = document.createElement('a');
    link.href = attachment.url;
    link.textContent = name;
    link.download = name;
    caption.append(link);
    figure.append(caption);
    return figure;
  }

  function messageView(message, index) {
    const article = document.createElement('article');
    const role = message.role === 'user' ? 'user' : 'assistant';
    article.id = message.id || `message-${String(index + 1).padStart(3, '0')}`;
    article.className = `message message-${role}`;
    article.setAttribute('aria-label', `Message ${index + 1}: ${role === 'user' ? 'User' : 'Assistant'}`);
    const header = document.createElement('header');
    header.className = 'message-header';
    const author = document.createElement('span');
    author.className = 'message-role';
    author.textContent = role === 'user' ? 'User' : 'Assistant';
    header.append(author);
    if (message.phase === 'commentary') {
      const phase = document.createElement('span');
      phase.className = 'message-phase';
      phase.textContent = 'Progress update';
      header.append(phase);
    }
    if (message.timestamp && Number.isFinite(Date.parse(message.timestamp))) {
      const time = document.createElement('time');
      time.className = 'message-time';
      time.dateTime = message.timestamp;
      time.textContent = dateFormatter.format(new Date(message.timestamp));
      header.append(time);
    }
    const permalink = document.createElement('a');
    permalink.className = 'message-permalink';
    permalink.href = `#${encodeURIComponent(article.id)}`;
    permalink.textContent = `#${index + 1}`;
    permalink.setAttribute('aria-label', `Link to message ${index + 1}`);
    header.append(permalink);
    const body = document.createElement('div');
    body.className = 'message-body';
    // The exporter generates and sanitizes this HTML; raw session text is never parsed here.
    if (typeof message.html === 'string') body.innerHTML = message.html;
    else { body.textContent = message.text || ''; body.style.whiteSpace = 'pre-wrap'; }
    article.append(header, body);
    if (Array.isArray(message.attachments) && message.attachments.length) {
      const attachments = document.createElement('div');
      attachments.className = 'attachments';
      for (const item of message.attachments) {
        const view = attachmentView(item);
        if (view) attachments.append(view);
      }
      if (attachments.childElementCount) article.append(attachments);
    }
    const note = document.createElement('p');
    note.className = 'linked-note';
    note.textContent = 'This directly linked message is shown even though it does not match the current filters.';
    note.hidden = true;
    article.append(note);
    entries.push({ article, note, progress: message.phase === 'commentary', text: `${message.text || body.textContent} ${(message.attachments || []).map(item => item.name || '').join(' ')}`.toLocaleLowerCase() });
    return article;
  }

  function applyFilters() {
    const query = $('search').value.trim().toLocaleLowerCase();
    const includeProgress = $('show-progress').checked;
    const target = linkedId();
    let visible = 0;
    let linkedException = 0;
    for (const entry of entries) {
      const matches = (!query || entry.text.includes(query)) && (includeProgress || !entry.progress);
      const forceVisible = entry.article.id === target && !matches;
      entry.article.hidden = !matches && !forceVisible;
      entry.note.hidden = !forceVisible;
      if (!entry.article.hidden) visible++;
      if (forceVisible) linkedException++;
    }
    $('clear-search').hidden = !query;
    $('empty-results').hidden = visible !== 0;
    $('message-count').textContent = `${visible === entries.length ? entries.length : `${visible} of ${entries.length}`} messages${linkedException ? ' · includes linked message' : ''}`;
  }

  function revealLinkedMessage() {
    applyFilters();
    const target = document.getElementById(linkedId());
    if (target) requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
  }

  $('search').addEventListener('input', applyFilters);
  $('show-progress').addEventListener('change', applyFilters);
  $('clear-search').addEventListener('click', () => {
    $('search').value = '';
    applyFilters();
    $('search').focus();
  });
  window.addEventListener('hashchange', revealLinkedMessage);

  fetch('transcript.json?v=20260909')
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(data => {
      if (!Array.isArray(data.messages)) throw new Error('The transcript has no message list.');
      if (data.scopeNote) $('scope-note').textContent = data.scopeNote;
      const fragment = document.createDocumentFragment();
      data.messages.forEach((message, index) => fragment.append(messageView(message, index)));
      $('transcript').append(fragment);
      revealLinkedMessage();
    })
    .catch(() => {
      $('message-count').textContent = 'Transcript unavailable';
      $('load-error').hidden = false;
      $('load-error').textContent = 'The conversation could not be loaded. Please reload this page, or use the text download above.';
    });
})();
