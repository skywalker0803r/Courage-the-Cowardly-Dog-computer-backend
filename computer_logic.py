import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"

HEADERS = {
    "Content-Type": "application/json"
}

def query_gemini(user_message: str) -> str:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_message
                    }
                ]
            }
        ],
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
    return candidates[0]["content"]["parts"][0]["text"]
