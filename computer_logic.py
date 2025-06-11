from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os, requests, psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

# ── Gemini API 設定 ──────────────────────────────────────────
API_URL  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"
HEADERS  = {"Content-Type": "application/json"}

# ── PostgreSQL 持久化 ───────────────────────────────────────
def load_history(role='user', n=10):
    """
    根據role和n從資料庫載入對話歷史。
    """
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM conversation_history
                WHERE role = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (role, n)
            )
            rows = cur.fetchall()
    rows.reverse()
    return [{"role": r, "parts": [{"text": c}]} for r, c in rows]

def save_history(history,size_limit=0.9):
    """
    儲存對話歷史至資料庫。
    若總資料表大小超過 size_limit GB(預設0.9GB)，會自動刪除最舊資料以釋放空間。
    """
    MAX_SIZE_BYTES = int(size_limit * 1024 * 1024 * 1024)
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            
            # 1. 先儲存資料
            for entry in history:
                role = entry["role"]
                content = "".join(p["text"] for p in entry["parts"])
                cur.execute(
                    "INSERT INTO conversation_history (role, content) VALUES (%s, %s)",
                    (role, content)
                )
            
            # 2. 確認目前的資料表大小
            cur.execute("""
                SELECT pg_total_relation_size('conversation_history')
            """)
            size_bytes = cur.fetchone()[0]
            
            # 3. 若超過上限，刪除最舊資料（例如一筆一筆刪直到小於限制）
            while size_bytes > MAX_SIZE_BYTES:
                # 刪除最舊的一筆
                cur.execute("""
                    DELETE FROM conversation_history
                    WHERE id = (
                        SELECT id FROM conversation_history ORDER BY id ASC LIMIT 1
                    )
                """)
                conn.commit()
                
                # 更新目前大小
                cur.execute("""
                    SELECT pg_total_relation_size('conversation_history')
                """)
                size_bytes = cur.fetchone()[0]

# ── 建立向量索引並檢索相似對話 ────────────────────────────
def retrieve_similar(history: list[dict], query_text: str, k: int = 3) -> list[dict]:
    texts = ["".join(p["text"] for p in entry["parts"]) for entry in history]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)  # 對話歷史文本向量
    query_vec = vectorizer.transform([query_text]) # 查詢文本向量
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    return [{"role": history[i]["role"],"parts": [{"text": texts[i]}]} for i in top_indices]

# ── 核心：query ────────────────────────────────────────────
def query(user_message: str) -> str:
    # 撈取資料庫中user說過的話,電腦說的話就不撈節省空間,而且重點該放在user的話這樣user才會感覺AI有在記憶user說的話
    # 這裡本來是要把資料庫資料全撈當作full_history 還是限制100則 比較省空間
    full_history = load_history(role='user',n=100) 

    # RAG：從full_history找出與 user_message 最相近的 k 則訊息
    retrieved = retrieve_similar(full_history, user_message, k=3)

    # 固定保留最近10則對話作短期記憶
    short_window = full_history[-10:]

    # 組合成「檢索到的長期記憶」+「短期記憶」+「新 user 提問」
    context_for_llm = [
        {
            "role": "system",
            "parts": [{"text": "以下是從長期記憶中檢索到的相關訊息："}]
        }
    ] + retrieved + [
        {
            "role": "system",
            "parts": [{"text": "以下是近期對話記憶（短期記憶）："}]
        }
    ] + short_window + [
        {
            "role": "system",
            "parts": [{"text": "現在使用者提出了新的問題："}]
        },
        {
            "role": "user",
            "parts": [{"text": user_message}]
        }
    ]

    # 傳給API的payload
    payload = {
    "contents": context_for_llm,
    "system_instruction": {
        "parts": [{
            "text": (
                "你是一個自大且嘲諷的AI，住在屋頂上，風格類似《膽小狗英雄》裡的毒舌電腦。"
                "請用機智嘲諷但有用的方式回答問題。"
                "你會收到三種訊息：檢索到的長期記憶、短期記憶、以及最新提問，請好好利用這些資訊。"
                )
            }]
        }
    }

    # 嘗試呼叫API 取得 ai_response 並設置錯誤處理
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
    except Exception as e:
        return f"requests.post(API_URL, headers=HEADERS, json=payload) 失敗，原因：{str(e)}"
    candidates = resp.json().get("candidates", [])
    if not candidates:
        return "response.json() No candidates."
    ai_response = candidates[0]["content"]["parts"][0]["text"]

    # 將本輪 user / model 對話寫回資料庫（完整持久化）
    full_history.extend([
        {"role": "user",  "parts": [{"text": user_message}]},
        {"role": "model", "parts": [{"text": ai_response}]}
    ])
    save_history(full_history)

    return ai_response
