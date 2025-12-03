import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LLM Chat Interface',
  description: 'Chat interface powered by OpenAI GPT-4',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

