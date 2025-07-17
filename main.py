import os
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def ask_question(request: Request, x_api_key: str = Header(None)):
    if x_api_key != os.getenv("API_KEY"):
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})

    data = await request.json()
    question = data.get("question", "")
    theme = data.get("theme", "")

    if not question:
        return JSONResponse(status_code=400, content={"error": "Missing question"})

    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    query_response = index.query(
        vector=client.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding,
        top_k=10,
        include_metadata=True
    )

    # Prompt with context from top sources
    context = "\n".join([match["metadata"]["text"] for match in query_response["matches"]])
    prompt = f"Based on the following context, answer the question as bullet points.\n\nContext:\n{context}\n\nQuestion: {question}\n\n-"

    chat = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Bible indexing assistant. You must only use the input context. Do not invent, infer, or paraphrase beyond the original material. "
                    "Each bullet point must directly quote or summarize a single chunk and must include the exact reference provided in that chunk’s metadata. "
                    "If the chunk has no clear answer, do not generate a point. "
                    "Format each result as:\n"
                    "- [summary or quote]\n"
                    "- Scripture: [Book Chapter:Verse from metadata]\n"
                    "- Source snippet: [matching text from the chunk]"
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    bullet_text = chat.choices[0].message.content.strip()
    bullets = [line.strip("- ").strip() for line in bullet_text.split("\n") if line.strip().startswith("-") or line.strip()]

    results = []
    for i, b in enumerate(bullets):
        if i < len(query_response["matches"]):
            meta = query_response["matches"][i]["metadata"]
            book = meta.get("title", "")
            chapter = meta.get("chapter")
            verse = meta.get("verse")
            ref = f"{book} {int(chapter)}:{int(verse)}" if book and chapter and verse else ""
            link = meta.get("jworg_link", "")
        else:
            ref = ""
            link = ""

        results.append({
            "text": b,
            "reference": ref,
            "link": link
        })

    return {"question": question, "bullets": results}
