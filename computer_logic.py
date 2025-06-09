import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

# 自定義設定
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"

HEADERS = {
    "Content-Type": "application/json"
}
HISTORY_FILE = "conversation_history.json"
MAX_INPUT_TOKENS = 50_000          # 自訂安全預算（遠小於 1M，請求速度更快）
TRIM_BACK_TO    = 40_000           # 超過就把最舊內容摘要

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

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