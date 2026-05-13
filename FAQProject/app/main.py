from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from hybrid_search import hybrid_search
from utils import load_faq, encode_faq_questions
from faiss_client import faq_index

import uvicorn
import os

app = FastAPI()

# =========================
# 全局变量（先不初始化）
# =========================
faq_list = []
faq_vectors = []


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# startup 初始化（关键修复）
# =========================
@app.on_event("startup")
def init_system():

    global faq_list, faq_vectors

    print("🚀 Loading FAQ...")

    faq_list = load_faq()

    for idx, item in enumerate(faq_list):
        if "id" not in item:
            item["id"] = idx + 1

    print("🚀 Encoding FAQ...")

    faq_vectors = encode_faq_questions(faq_list)

    faq_ids = [item["id"] for item in faq_list]

    print("🚀 Building FAISS index...")

    faq_index.add(faq_vectors)

    print("✅ Startup complete")


# =========================
# 首页
# =========================
@app.get("/")
def home():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "templates", "index.html")

    return FileResponse(html_path)


# =========================
# 搜索接口
# =========================
@app.get("/search")
def search(q: str):

    results_raw = hybrid_search(
        q,
        faq_list,
        top_k=5
    )

    results = []

    for item in results_raw:

        results.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "score": float(item.get("score", 0))
        })

    return {"results": results}


# =========================
# 启动
# =========================
if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )