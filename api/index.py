from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS so the frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Chat endpoint that accepts conversation history and returns AI response.
    Maintains context across multiple messages in the conversation.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        # Convert Pydantic models to dict format for OpenAI API
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add system message at the beginning if not already present
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": "You are a supportive mental coach."})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
