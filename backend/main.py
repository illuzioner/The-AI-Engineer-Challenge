from typing import List

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://localhost:3001",  # Next.js dev server if 3000 is busy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not client.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[m.dict() for m in req.messages],
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        # In real code you'd log this, not just expose it
        raise HTTPException(status_code=500, detail=str(e))
