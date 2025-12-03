'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = { role: 'user', content: input.trim() }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput('')
    setIsLoading(true)

    try {
      // Prepare messages for API (including conversation history)
      const apiMessages = updatedMessages.map(msg => ({
        role: msg.role,
        content: msg.content,
      }))


      const baseUrl =
        process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'

      const response = await fetch(`${baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages: apiMessages }),
      })


      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage: Message = { role: 'assistant', content: data.reply }
      setMessages([...updatedMessages, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      }
      setMessages([...updatedMessages, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        maxWidth: '768px',
        margin: '0 auto',
        width: '100%',
      }}
    >
      {/* Chat Messages Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              color: '#8e8ea0',
              marginTop: '2rem',
              fontSize: '1.1rem',
            }}
          >
            Start a conversation with your AI assistant
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              padding: '1rem',
              backgroundColor:
                message.role === 'user' ? '#343541' : '#444654',
              borderRadius: '0.5rem',
            }}
          >
            <div
              style={{
                fontWeight: '600',
                fontSize: '0.875rem',
                color: message.role === 'user' ? '#ececf1' : '#10a37f',
                marginBottom: '0.25rem',
              }}
            >
              {message.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <div
              style={{
                color: '#ececf1',
                whiteSpace: 'pre-wrap',
                lineHeight: '1.6',
              }}
            >
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              padding: '1rem',
              backgroundColor: '#444654',
              borderRadius: '0.5rem',
            }}
          >
            <div
              style={{
                fontWeight: '600',
                fontSize: '0.875rem',
                color: '#10a37f',
                marginBottom: '0.25rem',
              }}
            >
              Assistant
            </div>
            <div style={{ color: '#8e8ea0' }}>Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div
        style={{
          padding: '1rem',
          borderTop: '1px solid #565869',
          backgroundColor: '#343541',
        }}
      >
        <div
          style={{
            display: 'flex',
            gap: '0.5rem',
            alignItems: 'flex-end',
            maxWidth: '100%',
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            disabled={isLoading}
            style={{
              flex: 1,
              minHeight: '44px',
              maxHeight: '200px',
              padding: '0.75rem',
              borderRadius: '0.5rem',
              border: '1px solid #565869',
              backgroundColor: '#40414f',
              color: '#ececf1',
              fontSize: '1rem',
              fontFamily: 'inherit',
              resize: 'none',
              outline: 'none',
            }}
            rows={1}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement
              target.style.height = 'auto'
              target.style.height = `${Math.min(target.scrollHeight, 200)}px`
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              backgroundColor: isLoading || !input.trim() ? '#565869' : '#10a37f',
              color: '#ececf1',
              fontSize: '1rem',
              fontWeight: '500',
              cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s',
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

