import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"

HEADERS = {
    "Content-Type": "application/json"
}

conversation_history = []

def query(user_message: str) -> str:
    # 將使用者的輸入加入對話歷史
    conversation_history.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "contents": conversation_history.copy(),
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

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    candidates = response.json().get("candidates", [])
    if not candidates:
        return "No response from Gemini."

    ai_response = candidates[0]["content"]["parts"][0]["text"]
    
    # 把 AI 回應加入對話歷史
    conversation_history.append({"role": "model", "parts": [{"text": ai_response}]})

    return ai_response
