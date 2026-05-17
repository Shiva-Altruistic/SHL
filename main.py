from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from agent import Agent
from retriever import Retriever
import time

app = FastAPI(title="SHL Assessment Recommender API")

retriever = Retriever(catalog_path="catalog.json", index_path="faiss_index.bin")
agent = Agent(retriever=retriever)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/")
def read_root():
    return {"message": "SHL Assessment Recommender API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        result = agent.process_turn(messages_dict)
        
        # Enforce 25 second timeout buffer
        elapsed = time.time() - start_time
        if elapsed > 25:
            print(f"Warning: Request took {elapsed} seconds, nearing 30s timeout.")
            
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
