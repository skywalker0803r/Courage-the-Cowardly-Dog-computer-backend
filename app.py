from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from computer_logic import query # Assuming computer_logic.py is available
import os
import base64
import json
from google.cloud import texttospeech
from google.oauth2 import service_account
import traceback # <-- 新增這一行

# APP設定
app = FastAPI()

# CORS設定
# Allow all origins for development, specify production origins later
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://courage-the-cowardly-dog-computer.onrender.com", # 你的前端應用程式網址
        # 如果你有其他需要訪問API的來源，也可以在這裡添加
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # 明確指定允許的方法，尤其是OPTIONS
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
        # 如果 TTS 客戶端未初始化，直接返回一個預設的語音錯誤訊息
        print("Text-to-Speech client is not initialized. Skipping audio generation.")
        # 你可以返回一個空的 base64 字串或者一個預設的錯誤音訊
        return "" 
    try:
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
    except Exception as e:
        print(f"Error during Text-to-Speech synthesis: {e}")
        traceback.print_exc() # 打印 TTS 錯誤的詳細堆疊追蹤
        return "" # 返回空字串表示語音生成失敗


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
        audio_base64 = text_to_speech(reply) # 即使 TTS 失敗，reply 還是會被回傳
        return JSONResponse(content={"reply": reply, "audio": audio_base64})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}") # 打印錯誤訊息
        traceback.print_exc() # <-- 關鍵：打印完整的錯誤堆疊追蹤
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e)) # 給前端更詳細的錯誤訊息

@app.post("/set_instruction")
async def set_instruction(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        instruction = data.get("instruction")
        print(f"Received system_instruction: {instruction} for user: {user_id}") # Log the instruction
        if not instruction:
            raise HTTPException(status_code=400, detail="No instruction provided")
        
        from computer_logic import set_system_instruction
        set_system_instruction(user_id, instruction)
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        print(f"Error in /set_instruction endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000))) # 使用環境變數PORT，否則預設8000