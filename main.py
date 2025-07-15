from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    expected_key = os.getenv("BACKEND_API_KEY")
    provided_key = request.headers.get("x-api-key")

    if expected_key and provided_key != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

    return await call_next(request)

# Middleware to allow CORS for testing or Glide frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bible-notes")

class AskRequest(BaseModel):
    question: str
    theme: str = None

@app.post("/ask")
async def ask_question(body: AskRequest):
    try:
        embedding_response = client.embeddings.create(
            input=[body.question],
            model="text-embedding-3-small"
        )
        embedding = embedding_response.data[0].embedding

        response = index.query(
            vector=embedding,
            top_k=12,
            include_metadata=True,
            filter={"theme": body.theme} if body.theme else {}
        )
        matches = response.matches
        context = "\n\n".join([m.metadata["text"] for m in matches if "text" in m.metadata])

        chat = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Bible-based research assistant."},
                {"role": "user", "content": f"Based on the Bible-based material below, answer the question: {body.question}\n\n{context}"}
            ],
            temperature=0.2
        )

        return {
            "question": body.question,
            "summary": chat.choices[0].message.content.strip(),
            "sources": [m.metadata for m in matches]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
