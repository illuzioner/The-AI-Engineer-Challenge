'use client'

import { useState } from 'react'
import ChatInterface from '@/components/ChatInterface'

export default function Home() {
  return (
    <main style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <ChatInterface />
    </main>
  )
}

