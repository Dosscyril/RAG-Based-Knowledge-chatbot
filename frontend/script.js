const API_URL = 'http://localhost:8000';
        let selectedFiles = [];
        let hasDocuments = false;
        function formatText(text) {
            if (!text) return "";
            return text
                .replace(/\n/g, "<br>")
                .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        }
        function escapeHTML(text) {
            if (!text) return "";
            return text.replace(/[&<>"']/g, (m) => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;',
            }[m]));
        }
        const fileInput = document.getElementById('fileInput');
        const selectedFilesDiv = document.getElementById('selectedFiles');
        const uploadBtn = document.getElementById('uploadBtn');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');
        const messagesDiv = document.getElementById('messages');
        const docStatus = document.getElementById('docStatus');
        const chunksCount = document.getElementById('chunksCount');
        async function checkHealth() {
            try {
                const response = await fetch(`${API_URL}/health`);
                const data = await response.json();
                if (data.has_documents) {
                    hasDocuments = true;
                    updateUIState();
                }
            } catch (error) {
                showNotification('Cannot connect to backend.', 'error');
            }
        }
        fileInput.addEventListener('change', (e) => {
            selectedFiles = Array.from(e.target.files);
            displaySelectedFiles();
            uploadBtn.disabled = selectedFiles.length === 0;
        });
        function displaySelectedFiles() {
            if (selectedFiles.length === 0) {
                selectedFilesDiv.innerHTML = '<p style="color: #9ca3af;">No files selected</p>';
            } else {
                selectedFilesDiv.innerHTML = selectedFiles.map(file =>
                    `<div class="file-item">📄 ${file.name}</div>`
                ).join('');
            }
        }
        uploadBtn.addEventListener('click', async () => {
            if (selectedFiles.length === 0) return;
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="loading"></span>';
            const formData = new FormData();
            selectedFiles.forEach(file => formData.append('files', file));
            try {
                const response = await fetch(`${API_URL}/upload`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (response.ok) {
                    showNotification(`Processed ${selectedFiles.length} files!`, 'success');
                    chunksCount.textContent = data.chunks_created;
                    hasDocuments = true;
                    updateUIState();
                    selectedFiles = [];
                    fileInput.value = '';
                    displaySelectedFiles();
                } else {
                    showNotification('Upload failed: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error uploading: ' + error.message, 'error');
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Upload & Process';
            }
        });
        async function sendQuery() {
            const question = queryInput.value.trim();
            if (!question) return;
            addMessage(question, 'user');
            queryInput.value = '';
            queryInput.disabled = true;
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="loading"></span>';
            try {
                const response = await fetch(`${API_URL}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, k: 4 })
                });
                const data = await response.json();
                if (response.ok) {
                    addMessage(data.answer, 'assistant', data.sources);
                } else {
                    showNotification('Query failed: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error sending query: ' + error.message, 'error');
            } finally {
                queryInput.disabled = false;
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
                queryInput.focus();
            }
        }
        sendBtn.addEventListener('click', sendQuery);
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuery();
        });
        function addMessage(content, type, sources = null) {
    const emptyState = messagesDiv.querySelector('.empty-state');
    if (emptyState) emptyState.remove();
    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    let html = `<div class="message-content">${formatText(content)}</div>`;
    if (sources && sources.length > 0) {
        html += `
            <div class="sources">
                <div class="sources-title">📚 Sources:</div>
                ${sources.map(src => `
                    <div class="source-item">
                        <div class="source-filename">${escapeHTML(src.filename)}</div>
                        <div class="source-content">${escapeHTML(src.content.substring(0,150))}...</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    msg.innerHTML = html;
    messagesDiv.appendChild(msg);
    setTimeout(() => {
        msg.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
}
        function updateUIState() {
            if (hasDocuments) {
                docStatus.textContent = 'Yes';
                queryInput.disabled = false;
                sendBtn.disabled = false;
                queryInput.placeholder = 'Ask a question about your documents...';
            }
        }
        function showNotification(message, type) {
            const n = document.createElement('div');
            n.className = `notification ${type}`;
            n.textContent = message;
            document.body.appendChild(n);
            setTimeout(() => n.remove(), 4000);
        }
        checkHealth();
        displaySelectedFiles();