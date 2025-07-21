from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from computer_logic import query # Assuming computer_logic.py is available
import os
import base64
import json
from google.cloud import texttospeech
from google.oauth2 import service_account

# APP設定
app = FastAPI()

# CORS設定
# Allow all origins for development, specify production origins later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's actual origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GOOGLE_APPLICATION_CREDENTIALS_JSON設定以使用texttospeech服務
try:
    credentials_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
except Exception as e:
    print(f"Error loading Google Cloud credentials: {e}")
    # Handle the error appropriately, perhaps by exiting or disabling TTS
    tts_client = None

# 文字轉語音函數
def text_to_speech(text: str) -> str:
    if not tts_client:
        raise RuntimeError("Text-to-Speech client is not initialized.")

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="cmn-CN",
        name="cmn-CN-Chirp3-HD-Fenrir"
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    # 將音訊內容轉 base64 字串
    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
    return audio_base64

# 問答路由 接受 user_message 返回 reply(AI回覆)
@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    user_id = data.get("user_id")
    user_message = data.get("message", "")

    if not user_message:
        raise HTTPException(status_code=400, detail="No message provided")

    try:
        reply = query(user_message, user_id)
        audio_base64 = text_to_speech(reply)
        return JSONResponse(content={"reply": reply, "audio": audio_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)