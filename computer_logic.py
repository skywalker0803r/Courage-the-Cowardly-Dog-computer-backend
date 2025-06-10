import os, requests, psycopg2
from dotenv import load_dotenv
load_dotenv()

# ── Gemini API 設定 ──────────────────────────────────────────
API_URL  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"
HEADERS  = {"Content-Type": "application/json"}

# ── LangChain RAG 依賴 ──────────────────────────────────────
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ── PostgreSQL 持久化 ───────────────────────────────────────
def load_history():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur  = conn.cursor()
    cur.execute("SELECT role, content FROM conversation_history ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close(); conn.close()

    return [{"role": r, "parts": [{"text": c}]} for r, c in rows]

def save_history(history):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur  = conn.cursor()
    cur.execute("DELETE FROM conversation_history")
    for entry in history:
        role = entry["role"]
        content = "".join(p["text"] for p in entry["parts"])
        cur.execute("INSERT INTO conversation_history (role, content) VALUES (%s, %s)", (role, content))
    conn.commit(); cur.close(); conn.close()

# ── 建立向量索引並檢索相似對話 ────────────────────────────
def retrieve_similar(history: list[dict], query_text: str, k: int = 3) -> list[dict]:
    """把整份 history 建成 FAISS 向量庫，回傳與 query_text 最相似的 k 則訊息（list[dict] 格式）"""
    # 1) 將每則訊息轉成 LangChain Document
    docs = [
        Document(
            page_content="".join(p["text"] for p in entry["parts"]),
            metadata={"role": entry["role"], "idx": i}
        )
        for i, entry in enumerate(history)
    ]

    # 2) 建立 Embeddings & VectorStore（GoogleGenerativeAIEmbeddings 會直接調用 Gemini Embedding API）
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                              google_api_key=os.environ["GEMINI_API_KEY"])
    vectordb   = FAISS.from_documents(docs, embeddings)

    # 3) 相似度搜尋
    sims = vectordb.similarity_search(query_text, k=k)

    # 4) 回傳對應的原始 dict（保持 role / parts 結構）
    return [
        {
            "role" : sim.metadata["role"],
            "parts": [{"text": sim.page_content}]
        }
        for sim in sims
    ]

# ── 核心：query ────────────────────────────────────────────
def query(user_message: str) -> str:
    full_history = load_history()

    # RAG：找出與 user_message 最相近的 3 則舊訊息
    retrieved = retrieve_similar(full_history, user_message, k=3)

    # 固定保留最近三輪對話（共 6 則）作短期記憶
    short_window = full_history[-6:] if len(full_history) >= 6 else full_history

    # 組合成「檢索到的長期記憶」+「短期記憶」+「新 user 提問」
    context_for_llm = retrieved + short_window + [
        {"role": "user", "parts": [{"text": user_message}]}
    ]

    payload = {
        "contents": context_for_llm,
        "system_instruction": {
            "parts": [{
                "text": (
                    "你是一個自大且嘲諷的AI，住在屋頂上，風格類似《膽小狗英雄》裡的毒舌電腦。"
                    "請用機智嘲諷但有用的方式回答。"
                )
            }]
        }
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
    except Exception:
        return "API 呼叫失敗"

    candidates = resp.json().get("candidates", [])
    if not candidates:
        return "No response from Gemini."

    ai_response = candidates[0]["content"]["parts"][0]["text"]

    # 將本輪 user / model 對話寫回資料庫（完整持久化）
    full_history.extend([
        {"role": "user",  "parts": [{"text": user_message}]},
        {"role": "model", "parts": [{"text": ai_response}]}
    ])
    save_history(full_history)

    return ai_response
