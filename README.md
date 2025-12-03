# LLM Chat Frontend

A Next.js frontend for the LLM chat interface that displays conversations with OpenAI GPT-4.

## Prerequisites

- Node.js 18+ and npm (or yarn/pnpm)
- The backend API running on `http://localhost:8000` (see `../api/README.md` for backend setup)

## Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

## Running the Application

### Development Mode

Start the Next.js development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

**Note:** Make sure the backend API is running on `http://localhost:8000` before starting the frontend. The frontend is configured to proxy API requests to the backend during development.

### Production Build

Build the application for production:

```bash
npm run build
npm start
```

## Features

- **Chat Interface**: Clean, ChatGPT-like interface for conversations
- **Conversation History**: Maintains context across multiple messages in a session
- **Auto-scroll**: Automatically scrolls to the latest message
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Displays user-friendly error messages

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx      # Root layout component
│   ├── page.tsx        # Main page component
│   └── globals.css     # Global styles
├── components/
│   └── ChatInterface.tsx  # Main chat interface component
├── package.json        # Dependencies and scripts
├── next.config.js     # Next.js configuration
└── tsconfig.json      # TypeScript configuration
```

## Deployment

This frontend is configured to deploy on Vercel. The `vercel.json` in the project root handles routing for both the frontend and backend.
