import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const API_BASE = 'http://localhost:8000'

const FILE_ICONS = {
  pdf: '📄', docx: '📝', txt: '📃', pptx: '📊', xlsx: '📈', default: '📎'
}

// Allowed file extensions for upload
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.pptx', '.xlsx']

const SUGGESTIONS = [
  { icon: '📚', text: 'Summarize the key points from my documents' },
  { icon: '🔍', text: 'Find specific information in uploaded files' },
  { icon: '💡', text: 'Explain a concept from the documents' },
  { icon: '📊', text: 'Compare information across documents' },
]

function getFileIcon(filename) {
  const ext = filename.split('.').pop()?.toLowerCase()
  return FILE_ICONS[ext] || FILE_ICONS.default
}



// Toast notification component
function Toast({ toasts, removeToast }) {
  return (
    <div className="toast-container">
      {toasts.map(toast => (
        <div key={toast.id} className={`toast ${toast.type}`}>
          <span className="toast-icon">{toast.type === 'success' ? '✓' : '✕'}</span>
          <span className="toast-message">{toast.message}</span>
          <button className="toast-close" onClick={() => removeToast(toast.id)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  )
}

// Copy button component
function CopyButton({ text, className = '' }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <button className={`action-btn ${copied ? 'copied' : ''} ${className}`} onClick={handleCopy} title="Copy">
      {copied ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M20 6L9 17l-5-5" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
        </svg>
      )}
    </button>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [threadId, setThreadId] = useState(() => crypto.randomUUID())
  const [currentStep, setCurrentStep] = useState('')
  const [uploadingFiles, setUploadingFiles] = useState([])
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    }
    return 'light'
  })
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [toasts, setToasts] = useState([])

  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  // Toast management
  const addToast = useCallback((message, type = 'success') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px'
    }
  }, [input])

  // Cleanup session on page unload/refresh
  useEffect(() => {
    const cleanup = () => {
      // Fire and forget - use sendBeacon for reliability
      navigator.sendBeacon(`${API_BASE}/api/session/${threadId}`, '')
    }
    window.addEventListener('beforeunload', cleanup)
    return () => window.removeEventListener('beforeunload', cleanup)
  }, [threadId])

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setCurrentStep('Thinking')

    const assistantId = Date.now()
    setMessages(prev => [...prev, { role: 'assistant', content: '', id: assistantId, metadata: null, thinking: true }])

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input, thread_id: threadId }),
      })

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'step') setCurrentStep(data.content)
              else if (data.type === 'token') {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId ? { ...msg, content: msg.content + data.content, thinking: false } : msg
                ))
              } else if (data.type === 'done') {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId ? { ...msg, metadata: data, thinking: false } : msg
                ))
                setCurrentStep('')
              } else if (data.type === 'error') {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId ? { ...msg, content: `Error: ${data.content}`, error: true, thinking: false } : msg
                ))
                addToast(data.content, 'error')
              }
            } catch (e) { console.error('Parse error:', e) }
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId ? { ...msg, content: `Connection error: ${error.message}`, error: true, thinking: false } : msg
      ))
      addToast(`Connection error: ${error.message}`, 'error')
    } finally {
      setIsLoading(false)
      setCurrentStep('')
    }
  }, [input, isLoading, threadId, addToast])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const uploadFile = (file, fileId, currentThreadId) => {
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest()
      const formData = new FormData()
      formData.append('file', file)
      formData.append('thread_id', currentThreadId)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 50)
          setUploadingFiles(prev => prev.map(f =>
            f.id === fileId ? { ...f, progress: percent, status: 'uploading' } : f
          ))
        }
      }

      xhr.onload = () => {
        try {
          const result = JSON.parse(xhr.responseText)
          let ingestProgress = 50
          const ingestInterval = setInterval(() => {
            ingestProgress += 10
            if (ingestProgress >= 100) {
              clearInterval(ingestInterval)
              setUploadingFiles(prev => prev.map(f =>
                f.id === fileId ? {
                  ...f, progress: 100,
                  status: result.success ? 'success' : 'error',
                  result: result.success ? `${result.chunks} chunks` : result.message
                } : f
              ))
              if (result.success) {
                addToast(`${file.name} uploaded successfully`, 'success')
                // Add assistant welcome message for the uploaded file
                const fileExt = file.name.split('.').pop()?.toUpperCase() || 'file'
                const welcomeMsg = {
                  role: 'assistant',
                  content: `I've successfully processed your ${fileExt} file **"${file.name}"** and indexed ${result.chunks} chunks into my knowledge base.\n\nYou can now ask me questions about the content of this document. For example:\n- "Summarize the main points"\n- "What are the key findings?"\n- "Explain the section about..."`,
                  id: Date.now(),
                  metadata: { source: 'system', route: 'upload' }
                }
                setMessages(prev => [...prev, welcomeMsg])
              } else {
                addToast(`Failed to upload ${file.name}`, 'error')
              }
              setTimeout(() => {
                setUploadingFiles(prev => prev.filter(f => f.id !== fileId))
              }, 3000)
              resolve()
            } else {
              setUploadingFiles(prev => prev.map(f =>
                f.id === fileId ? { ...f, progress: ingestProgress, status: 'ingesting' } : f
              ))
            }
          }, 100)
        } catch {
          setUploadingFiles(prev => prev.map(f =>
            f.id === fileId ? { ...f, progress: 100, status: 'error', result: 'Failed' } : f
          ))
          addToast(`Failed to upload ${file.name}`, 'error')
          resolve()
        }
      }

      xhr.onerror = () => {
        setUploadingFiles(prev => prev.map(f =>
          f.id === fileId ? { ...f, progress: 100, status: 'error', result: 'Failed' } : f
        ))
        addToast(`Failed to upload ${file.name}`, 'error')
        resolve()
      }

      xhr.open('POST', `${API_BASE}/api/ingest`)
      xhr.send(formData)
    })
  }

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    processFiles(files)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const processFiles = async (files) => {
    // Filter out unsupported files with toast notification
    const validFiles = files.filter(file => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        addToast(`Unsupported file: ${file.name}. Allowed: PDF, DOCX, TXT, PPTX, XLSX`, 'error')
        return false
      }
      return true
    })

    if (!validFiles.length) return

    const newFiles = validFiles.map((file, i) => ({
      id: Date.now() + i,
      name: file.name,
      progress: 0,
      status: 'pending',
      file
    }))

    setUploadingFiles(prev => [...prev, ...newFiles])
    await Promise.all(newFiles.map(f => uploadFile(f.file, f.id, threadId)))
  }

  const removeFile = (fileId) => {
    setUploadingFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const newConversation = async () => {
    // Cleanup old session collections
    try {
      await fetch(`${API_BASE}/api/session/${threadId}`, { method: 'DELETE' })
    } catch (e) {
      console.warn('Session cleanup failed:', e)
    }
    setMessages([])
    setUploadingFiles([])
    setThreadId(crypto.randomUUID())
    setSidebarOpen(false)
  }

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion.text)
    textareaRef.current?.focus()
  }

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const allFiles = Array.from(e.dataTransfer.files)
    const validFiles = []
    const rejectedFiles = []

    allFiles.forEach(file => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      if (ALLOWED_EXTENSIONS.includes(ext)) {
        validFiles.push(file)
      } else {
        rejectedFiles.push(file.name)
      }
    })

    if (rejectedFiles.length > 0) {
      addToast(`Unsupported file type: ${rejectedFiles.join(', ')}. Allowed: PDF, DOCX, TXT, PPTX, XLSX`, 'error')
    }

    if (validFiles.length) processFiles(validFiles)
  }

  return (
    <div className="app">
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Mobile menu button */}
      <button className="mobile-menu-btn" onClick={() => setSidebarOpen(true)}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 12h18M3 6h18M3 18h18" />
        </svg>
      </button>

      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">🔀</div>
            Agentic RAG
          </div>
          <button className="theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5" />
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
              </svg>
            )}
          </button>
        </div>
        <div className="sidebar-content">
          <button className="new-chat-btn" onClick={newConversation}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            New Chat
          </button>

        </div>
      </aside>

      <main className="chat-main">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">🔀</div>
              <h2>How can I help you today?</h2>
              <p>Upload documents and ask questions. I'll use adaptive retrieval to find the best answers.</p>
              <div className="welcome-suggestions">
                {SUGGESTIONS.map((suggestion, i) => (
                  <button
                    key={i}
                    className="suggestion-card"
                    onClick={() => handleSuggestionClick(suggestion)}
                  >
                    <span className="suggestion-icon">{suggestion.icon}</span>
                    <span className="suggestion-text">{suggestion.text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={msg.id || i} className={`message ${msg.role}`}>
              <div className="avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
              <div className="content">
                {msg.thinking && !msg.content ? (
                  <div className="thinking">
                    <div className="thinking-dots">
                      <span className="thinking-dot"></span>
                      <span className="thinking-dot"></span>
                      <span className="thinking-dot"></span>
                    </div>
                    <span className="thinking-text">{currentStep || 'Thinking...'}</span>
                  </div>
                ) : (
                  <>
                    <div className={`text ${msg.error ? 'error' : ''}`}>
                      {msg.role === 'assistant' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>
                    {msg.role === 'assistant' && msg.content && !msg.error && (
                      <div className="message-actions">
                        <CopyButton text={msg.content} />
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        <div
          className="input-container"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploadingFiles.length > 0 && (
            <div className="input-files">
              {uploadingFiles.map(file => (
                <div key={file.id} className={`file-chip ${file.status}`}>
                  <span className="file-chip-icon">{getFileIcon(file.name)}</span>
                  <span className="file-chip-name">{file.name}</span>
                  {file.status !== 'success' && file.status !== 'error' && (
                    <span className="file-chip-progress">{file.progress}%</span>
                  )}
                  {file.status === 'success' && <span className="file-chip-status file-chip-done">✓</span>}
                  {file.status === 'error' && <span className="file-chip-status file-chip-error">✕</span>}
                  <button className="file-chip-remove" onClick={() => removeFile(file.id)}>×</button>
                </div>
              ))}
            </div>
          )}

          <div className={`input-wrapper ${dragOver ? 'drag-over' : ''}`}>
            <label className="attach-btn" title="Upload documents (PDF, DOCX, TXT, PPTX, XLSX)">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt,.pptx,.xlsx"
                onChange={handleFileUpload}
                multiple
                hidden
              />
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
              </svg>
            </label>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={dragOver ? "Drop files here..." : "Ask me anything..."}
              rows="1"
              disabled={isLoading}
            />
            <button className="send-btn" onClick={sendMessage} disabled={isLoading || !input.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
        </div>
      </main>

      <Toast toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default App
