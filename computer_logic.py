import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query_fireworks(user_message: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an arrogant, sarcastic AI living on the rooftop, similar to the computer from "
                    "Courage the Cowardly Dog. You respond with witty insults but always provide useful answers."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "model": "accounts/fireworks/models/deepseek-r1-0528"
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
