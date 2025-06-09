import requests
import os
import json
import psycopg2
from dotenv import load_dotenv
load_dotenv()

# 自定義設定
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"

HEADERS = {
    "Content-Type": "application/json"
}

MAX_INPUT_TOKENS = 50_000          # 自訂安全預算（遠小於 1M，請求速度更快）
TRIM_BACK_TO    = 40_000           # 超過就把最舊內容摘要

def split_text(text, max_bytes=900):
    import re
    # 先依句號、問號、感嘆號、換行符等斷句符號分割
    sentences = re.split(r'(?<=[。.!?？\n])', text)
    parts = []
    buffer = ""

    for s in sentences:
        # 嘗試把句子加到 buffer
        if len((buffer + s).encode('utf-8')) > max_bytes:
            # 如果加了會超長，先把現有 buffer 當一段存起來
            if buffer:
                parts.append(buffer)
                buffer = s
            else:
                # 單句超長，強制切割成多段
                bytes_s = s.encode('utf-8')
                start = 0
                while start < len(bytes_s):
                    chunk = bytes_s[start:start + max_bytes]
                    # decode時忽略破字元
                    parts.append(chunk.decode('utf-8', errors='ignore'))
                    start += max_bytes
                buffer = ""
        else:
            buffer += s

    if buffer:
        parts.append(buffer)
    return parts


def load_history():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM conversation_history ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    history = []
    for role, content in rows:
        parts = split_text(content)
        history.append({
            "role": role,
            "parts": [{"text": part} for part in parts]
        })
    return history

def save_history(history):
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    # 清空舊資料
    cur.execute("DELETE FROM conversation_history")
    # 重新插入全部歷史紀錄
    for entry in history:
        role = entry.get("role", "")
        content = "".join(p["text"] for p in entry.get("parts", []))
        cur.execute(
            "INSERT INTO conversation_history (role, content) VALUES (%s, %s)",
            (role, content)
        )
    conn.commit()
    cur.close()
    conn.close()

# 單獨用來呼叫 Gemini API 做摘要，避免呼叫 query 避免遞迴。
def generate_summary(text: str) -> str:
    """
    單獨用來呼叫 Gemini API 做摘要，避免呼叫 query 避免遞迴。
    """
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"請用 150 字以內摘要下列對話，保留結論或事項：\n{text}"
                    }
                ]
            }
        ],
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "你是一個自大且嘲諷的AI，住在屋頂上，風格類似《膽小狗英雄》裡的那台講話非常毒蛇的電腦。"
                        "你會用機智且嘲諷的回應，但總是提供有用的答案。"
                    )
                }
            ]
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
    except Exception as e:
        return f"摘要API呼叫失敗: {e}"

    candidates = response.json().get("candidates", [])
    if not candidates:
        return "無法取得摘要內容。"

    return candidates[0]["content"]["parts"][0]["text"]

def count_tokens(messages: list[dict]) -> int:
    # 簡易估算：len(text)/4；要精準可呼叫 countTokens API
    return sum(len(p["text"]) // 4 for m in messages for p in m["parts"])

def trim_history(conversation_history: list[dict]):
    # 判斷是否需要裁減
    if count_tokens(conversation_history) <= MAX_INPUT_TOKENS:
        return
    # 把最舊 N 句變成一段 summary，再刪掉原文
    old_msgs, conversation_history[:] = conversation_history[:20], conversation_history[20:]
    summary_prompt = (
        "請用 150 字以內摘要下列對話，保留結論或事項：\n"
        + "\n".join(p["text"] for m in old_msgs for p in m["parts"])
    )
    summary = generate_summary(summary_prompt) # 讓模型自己產生摘要
    conversation_history.insert(
        0, {"role": "system", "parts": [{"text": f"(舊對話摘要) {summary}"}]}
    )

def query(user_message: str) -> str:
    # 載入對話歷史
    conversation_history = load_history()
    # 將使用者的輸入加入對話歷史
    conversation_history.append({"role": "user", "parts": [{"text": user_message}]})
    # 先檢查並修剪
    trim_history(conversation_history)
    # 製作payload 放 歷史對話 跟 引導
    payload = {
        "contents": conversation_history,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "你是一個自大且嘲諷的AI，住在屋頂上，風格類似《膽小狗英雄》裡的那台講話非常毒蛇的電腦。你會用機智且嘲諷的回應，但總是提供有用的答案。"
                    )
                }
            ]
        }
    }
    # 嘗試呼叫API
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
    except:
        return 'API 呼叫失敗'
    # 判斷response是否有candidates
    candidates = response.json().get("candidates", [])
    if not candidates:
        return "No response from Gemini."
    # 撈取AI回應
    ai_response = candidates[0]["content"]["parts"][0]["text"]
    # 把AI回應加入對話歷史
    conversation_history.append({"role": "model", "parts": [{"text": ai_response}]})
    # 保存對話歷史
    save_history(conversation_history)
    return ai_response
