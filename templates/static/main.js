const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
let history = [];

async function send() {
  const message = inputEl.value.trim();
  if (!message) return;
  append('You', message, 'user');
  inputEl.value = '';

  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history })
  });
  const data = await res.json();
  const reply = data.reply || data.error;
  append('GPT', reply, 'assistant');
  history.push({ role: 'user', content: message });
  history.push({ role: 'assistant', content: reply });
}

function append(sender, text, cls) {
  const div = document.createElement('div');
  div.classList.add('message', cls);
  div.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keypress', e => e.key === 'Enter' && send());