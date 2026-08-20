// Standalone chatbot UI. API paths are proxied to Agent System A by Nginx.
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = {
  busy: false,
  timer: null,
  image: null,
  conversationId: localStorage.getItem('firstAidConversationId'),
  notificationSource: null,
  shownNotifications: new Set(),
};
const stages = {
  queued: 5, extracting: 20, vision: 42, chunking: 62,
  embedding: 78, updating_bm25: 92, complete: 100,
};
const clean = value => String(value ?? '').replace(
  /[&<>'"]/g,
  character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'})[character],
);
const welcomeMarkup = $('#conversation').innerHTML;

function toast(message, error = false) {
  const notice = document.createElement('div');
  notice.className = `toast${error ? ' error' : ''}`;
  notice.textContent = message;
  $('#toastRegion').append(notice);
  setTimeout(() => notice.remove(), 4200);
}

function showAppointmentReminder(event) {
  if (!event?.event_id || state.shownNotifications.has(event.event_id)) return;
  state.shownNotifications.add(event.event_id);
  const popup = document.createElement('section');
  popup.className = 'appointment-reminder';
  popup.setAttribute('role', 'alertdialog');
  popup.innerHTML = `<div class="appointment-reminder-icon">!</div><div><strong>${clean(event.title || 'Appointment reminder')}</strong><p>${clean(event.message || 'Your appointment starts soon.')}</p><small>${clean(conversationAge(event.appointment_time))} · Synthetic appointment</small></div><button type="button" aria-label="Dismiss reminder">×</button>`;
  popup.querySelector('button').addEventListener('click', async () => {
    popup.remove();
    await api(`/v1/notifications/${encodeURIComponent(event.event_id)}/read`, {method: 'PATCH'}).catch(() => {});
  });
  document.body.append(popup);
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(event.title || 'Appointment reminder', {body: event.message});
  }
}

function disconnectNotifications() {
  state.notificationSource?.close();
  state.notificationSource = null;
}

async function connectNotifications() {
  disconnectNotifications();
  if (!state.conversationId) return;
  const id = encodeURIComponent(state.conversationId);
  const unread = await api(`/v1/notifications?conversation_id=${id}&unread_only=true`).catch(() => []);
  unread.forEach(showAppointmentReminder);
  const source = new EventSource(`/v1/notifications/stream?conversation_id=${id}`);
  source.addEventListener('appointment_reminder', event => {
    try { showAppointmentReminder(JSON.parse(event.data)); } catch {}
  });
  state.notificationSource = source;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function updateChatStreamStatus(message) {
  const body = $('#typingMessage .assistant-body');
  if (!body) return;
  body.innerHTML = `<div class="typing"><i></i><i></i><i></i></div><small>${clean(message)}</small>`;
}

async function streamChat(requestPayload) {
  const response = await fetch('/v1/chat/stream', {
    method: 'POST',
    headers: {
      'Accept': 'text/event-stream',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestPayload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Streaming request failed (${response.status})`);
  }
  if (!response.body) throw new Error('Streaming is not supported by this browser.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;

  const processFrame = frame => {
    let eventName = 'message';
    const dataLines = [];
    for (const line of frame.split(/\r?\n/)) {
      if (!line || line.startsWith(':')) continue;
      const separator = line.indexOf(':');
      const field = separator === -1 ? line : line.slice(0, separator);
      const value = separator === -1 ? '' : line.slice(separator + 1).replace(/^ /, '');
      if (field === 'event') eventName = value;
      if (field === 'data') dataLines.push(value);
    }
    if (!dataLines.length) return;
    let data;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      throw new Error('The server returned an invalid streaming event.');
    }
    if (eventName === 'status') updateChatStreamStatus('Routing your request…');
    if (eventName === 'heartbeat') updateChatStreamStatus('Gathering and validating evidence…');
    if (eventName === 'result') result = data;
    if (eventName === 'error') throw new Error(data.message || 'The request could not be completed safely.');
  };

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    let match = buffer.match(/\r?\n\r?\n/);
    while (match?.index !== undefined) {
      processFrame(buffer.slice(0, match.index));
      buffer = buffer.slice(match.index + match[0].length);
      match = buffer.match(/\r?\n\r?\n/);
    }
    if (done) break;
  }
  if (buffer.trim()) processFrame(buffer);
  if (!result) throw new Error('The response stream ended before returning an answer.');
  return result;
}

function setView(name) {
  $$('.nav-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.view === name));
  $$('.view').forEach(view => view.classList.toggle('active', view.id === `${name}View`));
  if (name === 'documents') loadDocuments();
}

async function health() {
  const element = $('#apiStatus');
  try {
    const result = await api('/health');
    element.className = 'api-status healthy';
    element.querySelector('strong').textContent = 'System ready';
    element.querySelector('span:last-child').textContent = `${result.qdrant_points.toLocaleString()} indexed chunks`;
  } catch {
    element.className = 'api-status error';
    element.querySelector('strong').textContent = 'Service unavailable';
    element.querySelector('span:last-child').textContent = 'Check Qdrant and Ollama';
  }
}

function resize() {
  const element = $('#queryInput');
  element.style.height = 'auto';
  element.style.height = `${Math.min(element.scrollHeight, 150)}px`;
}

function userMessage(text) {
  $('#welcome')?.remove();
  $('#conversation').insertAdjacentHTML(
    'beforeend',
    `<div class="message user"><div class="user-bubble">${clean(text)}</div></div>`,
  );
}

function typing() {
  $('#conversation').insertAdjacentHTML(
    'beforeend',
    '<div class="message" id="typingMessage"><div class="assistant-message"><div class="assistant-avatar">+</div><div class="assistant-body"><div class="typing"><i></i><i></i><i></i></div></div></div></div>',
  );
}

function answer(payload, {scroll = true} = {}) {
  $('#typingMessage')?.remove();
  const generation = payload.generation || {};
  const citations = generation.citations || [];
  const visuals = payload.visuals || [];
  const resultMap = new Map((payload.retrieval?.results || []).map(result => [result.chunk_id, result]));
  const text = clean(
    generation.answer || generation.insufficient_evidence_reason || 'I could not produce a grounded answer.',
  ).replace(/\[(S\d+)\]/g, '<span class="citation-link">[$1]</span>');
  const sources = citations.map(citation => {
    const result = resultMap.get(citation.chunk_id) || {};
    return `<article class="source-card"><strong>${clean(citation.label)}</strong><span>${clean(citation.doc_id)} - page ${clean(citation.pdf_page || citation.page)}</span><p>${clean(result.text || citation.section || 'Referenced source passage')}</p></article>`;
  }).join('');
  const rendered = visuals
    .filter(visual => visual.encoding === 'base64' && visual.data_base64 && visual.mime_type)
    .map(visual => `<figure class="manual-visual"><img src="data:${clean(visual.mime_type)};base64,${visual.data_base64}" alt="${clean(visual.alt_text || 'Rendered manual page')}" loading="lazy"><figcaption><strong>${clean(visual.doc_id)}</strong> - PDF page ${clean(visual.pdf_page)}${visual.section ? ` - ${clean(visual.section)}` : ''}</figcaption></figure>`)
    .join('');
  const route = (payload.route || generation.agent_route || [])
    .map(agent => agent.replaceAll('_', ' ')).join(' + ');
  const status = route
    ? `Handled by ${clean(route)}`
    : (generation.status === 'answered' ? 'Grounded answer' : clean(generation.abstention_category || generation.status || 'Response'));
  $('#conversation').insertAdjacentHTML(
    'beforeend',
    `<div class="message"><div class="assistant-message"><div class="assistant-avatar">+</div><div class="assistant-body"><div class="answer-status">${status}</div><div class="answer-text">${text}</div>${rendered ? `<div class="manual-visuals">${rendered}</div>` : ''}${sources ? `<details class="source-block"><summary>${citations.length} cited source${citations.length === 1 ? '' : 's'}</summary><div class="source-cards">${sources}</div></details>` : ''}</div></div></div>`,
  );
  if (scroll) window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}

function clearAttachment() {
  state.image = null;
  $('#imageInput').value = '';
  $('#attachmentPreview').classList.add('hidden');
}

function historyAssistant(message) {
  const metadata = message.metadata || {};
  answer({
    route: metadata.route || [],
    retrieval: null,
    visuals: [],
    generation: {
      status: metadata.status || 'answered',
      answer: message.content,
      citations: metadata.citations || [],
      abstention_category: metadata.abstention_category,
    },
  }, {scroll: false});
}

function resetConversationView() {
  $('#conversation').innerHTML = welcomeMarkup;
  $('#chatTitle').textContent = 'How can I help?';
  clearAttachment();
}

function conversationAge(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  }).format(date);
}

async function loadConversations() {
  const rows = await api('/v1/conversations?limit=50');
  const list = $('#conversationList');
  if (!rows.length) {
    list.innerHTML = '<span class="conversation-list-empty">No previous conversations</span>';
    return rows;
  }
  list.innerHTML = rows.map(row => `
    <article class="conversation-row${row.conversation_id === state.conversationId ? ' active' : ''}">
      <button class="conversation-select" type="button" data-conversation="${clean(row.conversation_id)}">
        <strong>${clean(row.title || 'New conversation')}</strong>
        <span>${row.message_count || 0} messages${row.updated_at ? ` - ${clean(conversationAge(row.updated_at))}` : ''}</span>
      </button>
      <button class="conversation-delete" type="button" data-delete-conversation="${clean(row.conversation_id)}" aria-label="Delete ${clean(row.title || 'conversation')}" title="Delete conversation">x</button>
    </article>
  `).join('');
  return rows;
}

async function openConversation(id) {
  if (state.busy) return;
  const conversation = await api(`/v1/conversations/${encodeURIComponent(id)}`);
  state.conversationId = id;
  localStorage.setItem('firstAidConversationId', id);
  $('#conversation').innerHTML = '';
  $('#chatTitle').textContent = conversation.title || 'Conversation';
  for (const message of conversation.messages || []) {
    if (message.role === 'user') userMessage(message.content);
    if (message.role === 'assistant') historyAssistant(message);
  }
  if (!(conversation.messages || []).length) resetConversationView();
  await loadConversations();
  await connectNotifications();
  window.scrollTo({top: document.body.scrollHeight});
}

async function startNewConversation() {
  if (state.busy) return;
  state.conversationId = null;
  localStorage.removeItem('firstAidConversationId');
  disconnectNotifications();
  resetConversationView();
  await loadConversations();
  $('#queryInput').focus();
}

async function deleteConversation(id) {
  if (!confirm('Delete this conversation permanently?')) return;
  await api(`/v1/conversations/${encodeURIComponent(id)}`, {method: 'DELETE'});
  if (state.conversationId === id) {
    state.conversationId = null;
    localStorage.removeItem('firstAidConversationId');
    resetConversationView();
  }
  await loadConversations();
  toast('Conversation deleted.');
}

async function bootstrapConversations() {
  try {
    const rows = await loadConversations();
    const selected = state.conversationId || rows[0]?.conversation_id || null;
    if (selected) await openConversation(selected);
  } catch {
    state.conversationId = null;
    localStorage.removeItem('firstAidConversationId');
    resetConversationView();
    toast('Could not restore conversation history.', true);
  }
}

async function ask(question) {
  const text = question.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $('#sendButton').disabled = true;
  userMessage(text + (state.image ? `\n[Image: ${state.image.name}]` : ''));
  typing();
  $('#queryInput').value = '';
  resize();
  try {
    let payload;
    if (state.image) {
      const form = new FormData();
      form.append('query', text);
      form.append('image', state.image);
      if (state.conversationId) form.append('conversation_id', state.conversationId);
      payload = await api('/v1/chat/visual', {method: 'POST', body: form});
    } else {
      payload = await streamChat({
        query: text,
        doc_id: null,
        conversation_id: state.conversationId,
      });
    }
    if (payload.conversation_id) {
      state.conversationId = payload.conversation_id;
      localStorage.setItem('firstAidConversationId', state.conversationId);
      await connectNotifications();
    }
    if (payload.appointment?.status === 'booked' && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().catch(() => {});
    }
    answer(payload);
    clearAttachment();
    await loadConversations();
  } catch (error) {
    $('#typingMessage')?.remove();
    toast(error.message, true);
  } finally {
    state.busy = false;
    $('#sendButton').disabled = false;
    $('#queryInput').focus();
  }
}

function docId(name) {
  return name.replace(/\.pdf$/i, '').replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '').toUpperCase().slice(0, 80);
}

function selectFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) return toast('Please choose a PDF file.', true);
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $('#pdfFile').files = transfer.files;
  $('#fileLabel').textContent = file.name;
  if (!$('#documentId').value) $('#documentId').value = docId(file.name);
}

async function loadDocuments() {
  try {
    const documents = await api('/v1/documents');
    $('#documentCount').textContent = documents.length;
    if (!documents.length) {
      $('#documentList').innerHTML = '<div class="empty-state"><span>+</span><strong>No managed documents yet</strong><p>Your original seven manuals remain safely indexed.</p></div>';
      return;
    }
    $('#documentList').innerHTML = documents.map(document => `<article class="document-row"><div class="doc-icon">PDF</div><div class="doc-info"><strong>${clean(document.filename)}</strong><span>${clean(document.doc_id)} - ${document.pages || 0} pages - ${document.chunks || 0} chunks</span></div><button class="delete-button" type="button" data-delete="${clean(document.doc_id)}">Delete</button></article>`).join('');
  } catch (error) {
    toast(error.message, true);
  }
}

function showJob(job) {
  const stage = job.detail?.stage || job.status;
  $('#jobCard').classList.remove('hidden');
  $('#jobProgress').style.width = `${stages[stage] || 10}%`;
  $('#jobPercent').textContent = job.status === 'complete' ? 'Complete' : job.status === 'failed' ? 'Failed' : 'Working';
  $('#jobDetail').textContent = job.status === 'failed'
    ? (job.detail?.error || 'Processing failed')
    : `Current stage: ${String(stage).replaceAll('_', ' ')}`;
}

async function poll(id) {
  clearTimeout(state.timer);
  try {
    const job = await api(`/v1/documents/jobs/${id}`);
    showJob(job);
    if (job.status === 'complete') {
      toast('Document is indexed and ready for questions.');
      $('#uploadButton').disabled = false;
      await loadDocuments();
      health();
    } else if (job.status === 'failed') {
      toast(job.detail?.error || 'Document processing failed.', true);
      $('#uploadButton').disabled = false;
    } else {
      state.timer = setTimeout(() => poll(id), 1800);
    }
  } catch (error) {
    toast(error.message, true);
    $('#uploadButton').disabled = false;
  }
}

async function upload(event) {
  event.preventDefault();
  const file = $('#pdfFile').files[0];
  if (!file) return toast('Choose a PDF first.', true);
  const form = new FormData();
  form.append('file', file);
  form.append('doc_id', $('#documentId').value);
  form.append('ocr_lang', $('#ocrLanguage').value);
  form.append('scope', 'managed');
  form.append('replace', $('#replaceDocument').checked ? 'true' : 'false');
  $('#uploadButton').disabled = true;
  try {
    const job = await api('/v1/documents', {method: 'POST', body: form});
    showJob(job);
    poll(job.job_id);
  } catch (error) {
    toast(error.message, true);
    $('#uploadButton').disabled = false;
  }
}

async function removeDocument(id) {
  if (!confirm(`Delete ${id} from Qdrant, BM25, and managed files?`)) return;
  try {
    await api(`/v1/documents/${encodeURIComponent(id)}`, {method: 'DELETE'});
    toast(`${id} was deleted.`);
    loadDocuments();
    health();
  } catch (error) {
    toast(error.message, true);
  }
}

$$('.nav-tab').forEach(tab => tab.addEventListener('click', () => setView(tab.dataset.view)));
$$('[data-question]').forEach(button => button.addEventListener('click', () => ask(button.dataset.question)));
$('#chatForm').addEventListener('submit', event => { event.preventDefault(); ask($('#queryInput').value); });
$('#queryInput').addEventListener('input', resize);
$('#queryInput').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#chatForm').requestSubmit();
  }
});
$('#imageInput').addEventListener('change', event => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 15 * 1024 * 1024) {
    toast('Image must be smaller than 15 MB.', true);
    return clearAttachment();
  }
  state.image = file;
  $('#attachmentName').textContent = file.name;
  $('#attachmentPreview').classList.remove('hidden');
});
$('#removeAttachment').addEventListener('click', clearAttachment);
$('#newConversation').addEventListener('click', startNewConversation);
$('#conversationList').addEventListener('click', event => {
  const select = event.target.closest('[data-conversation]');
  const remove = event.target.closest('[data-delete-conversation]');
  if (select) openConversation(select.dataset.conversation).catch(error => toast(error.message, true));
  if (remove) deleteConversation(remove.dataset.deleteConversation).catch(error => toast(error.message, true));
});
$('#pdfFile').addEventListener('change', event => selectFile(event.target.files[0]));
$('#uploadForm').addEventListener('submit', upload);
$('#refreshDocuments').addEventListener('click', loadDocuments);
$('#documentList').addEventListener('click', event => {
  if (event.target.dataset.delete) removeDocument(event.target.dataset.delete);
});

const zone = $('#dropZone');
['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => {
  event.preventDefault();
  zone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => {
  event.preventDefault();
  zone.classList.remove('dragover');
}));
zone.addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));

health();
loadDocuments();
bootstrapConversations();
