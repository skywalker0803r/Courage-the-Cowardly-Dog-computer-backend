import requests
import os
from dotenv import load_dotenv
import redis
import json # Import the json library for serialization/deserialization

load_dotenv()

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"

HEADERS = {
    "Content-Type": "application/json"
}

# Initialize Redis connection
# Use the URL you provided: redis://red-d1utdqndiees73b35it0:6379
# decode_responses=True will ensure that retrieved data is decoded to strings
redis_client = redis.from_url("redis://red-d1utdqndiees73b35it0:6379", decode_responses=True)

def query(user_message: str, user_id: str) -> str:
    # Get or initialize user history from Redis
    # Redis stores strings, so we need to deserialize from JSON
    conversation_json = redis_client.get(user_id)
    if conversation_json:
        conversation = json.loads(conversation_json)
    else:
        conversation = []

    # 将使用者訊息加入歷史
    conversation.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "contents": conversation.copy(),  # 傳過去之前複製，避免被 Gemini 修改歷史
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

    # 把 AI 回應也加進歷史
    conversation.append({"role": "model", "parts": [{"text": ai_response}]})

    # Save the updated conversation history back to Redis
    # Redis stores strings, so we need to serialize to JSON
    redis_client.set(user_id, json.dumps(conversation))

    return ai_response