from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    expected_key = os.getenv("BACKEND_API_KEY")
    provided_key = request.headers.get("x-api-key")
    if expected_key and provided_key != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    return await call_next(request)

class QuestionInput(BaseModel):
    question: str

@app.post("/")
def ask_question(input: QuestionInput):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    namespace = os.getenv("PINECONE_NAMESPACE")

    query = input.question
    embed = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding

    results = index.query(vector=embed, top_k=8, include_metadata=True, namespace=namespace)

    sources = []
    texts = []
    for match in results.matches:
        meta = match.metadata
        if meta:
            sources.append(meta)
            if "text" in meta:
                texts.append(meta["text"])

    prompt = (
        "You are a Bible study assistant. A user asked:\n\n"
        f"{query}\n\n"
        "Based only on the scriptures and notes below, answer the question in clear bullet points:\n\n"
        + "\n".join(texts)
    )

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    summary = completion.choices[0].message.content

    return {
        "question": query,
        "summary": summary,
        "sources": sources,
    }
