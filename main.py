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

def jw_link(book, chapter, verse):
    book_codes = {
        "Genesis": "010", "Exodus": "020", "Leviticus": "030", "Numbers": "040",
        "Deuteronomy": "050", "Joshua": "060", "Judges": "070", "Ruth": "080", "1 Samuel": "090", "2 Samuel": "100",
        "1 Kings": "110", "2 Kings": "120", "1 Chronicles": "130", "2 Chronicles": "140", "Ezra": "150",
        "Nehemiah": "160", "Esther": "170", "Job": "180", "Psalms": "190", "Proverbs": "200",
        "Ecclesiastes": "210", "Song of Solomon": "220", "Isaiah": "230", "Jeremiah": "240", "Lamentations": "250",
        "Ezekiel": "260", "Daniel": "270", "Hosea": "280", "Joel": "290", "Amos": "300",
        "Obadiah": "310", "Jonah": "320", "Micah": "330", "Nahum": "340", "Habakkuk": "350",
        "Zephaniah": "360", "Haggai": "370", "Zechariah": "380", "Malachi": "390",
        "Matthew": "400", "Mark": "410", "Luke": "420", "John": "430", "Acts": "440",
        "Romans": "450", "1 Corinthians": "460", "2 Corinthians": "470", "Galatians": "480", "Ephesians": "490",
        "Philippians": "500", "Colossians": "510", "1 Thessalonians": "520", "2 Thessalonians": "530", "1 Timothy": "540",
        "2 Timothy": "550", "Titus": "560", "Philemon": "570", "Hebrews": "580", "James": "590",
        "1 Peter": "600", "2 Peter": "610", "1 John": "620", "2 John": "630", "3 John": "640",
        "Jude": "650", "Revelation": "660"
    }

    code = book_codes.get(book)
    if not code or not chapter or not verse:
        return ""

    chapter_str = str(int(chapter)).zfill(2)
    verse_str = str(int(verse)).zfill(3)  # <--- 3 digits here is the key fix

    return f"https://www.jw.org/finder?srcid=jwlshare&wtlocale=E&prefer=lang&bible={code}{chapter_str}{verse_str}&pub=nwtsty"
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
    query_response = index.query(vector=client.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding, top_k=10, include_metadata=True)

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
            book = meta.get("book", "")
            chapter = meta.get("chapter")
            verse = meta.get("verse")
            ref = f"{book} {int(chapter)}:{int(verse)}" if book and chapter and verse else ""
            link = jw_link(book, chapter, verse) if book and chapter and verse else ""
        else:
            ref = ""
            link = ""

        results.append({
            "text": b,
            "reference": ref,
            "link": link
        })

    return {"question": question, "bullets": results}
