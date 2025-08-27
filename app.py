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
import re

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
        print("Text-to-Speech client is not initialized. Skipping audio generation.")
        return ""

    MAX_TTS_BYTES = 800 # Set a slightly lower limit than 900 for safety

    # Split text by common Chinese and English sentence-ending punctuation
    # Keep the delimiters so we can re-add them or use them for splitting
    sentence_delimiters = r'([。？！；.?!])'
    sentences = re.split(sentence_delimiters, text)

    # Filter out empty strings that might result from split
    sentences = [s for s in sentences if s.strip()]

    audio_content_parts = []

    current_chunk = ""
    for i, part in enumerate(sentences):
        # If the part is a delimiter, append it to the current chunk
        if re.match(sentence_delimiters, part):
            current_chunk += part
            # If adding the delimiter makes the chunk too long, process the current_chunk
            # before adding the delimiter, then process the delimiter as a separate chunk.
            # This is a simplified approach.
            if len(current_chunk.encode('utf-8')) > MAX_TTS_BYTES and current_chunk[:-len(part)].strip():
                audio_content_parts.extend(synthesize_chunk(current_chunk[:-len(part)].strip(), tts_client))
                current_chunk = part
            
            if current_chunk.strip(): # Process the delimiter if it's a standalone chunk
                audio_content_parts.extend(synthesize_chunk(current_chunk.strip(), tts_client))
            current_chunk = "" # Reset for next sentence
            continue

        # If adding the current part makes the chunk too long, synthesize the current_chunk
        # and start a new one with the current part.
        if len((current_chunk + part).encode('utf-8')) > MAX_TTS_BYTES and current_chunk.strip():
            audio_content_parts.extend(synthesize_chunk(current_chunk.strip(), tts_client))
            current_chunk = part
        else:
            current_chunk += part

        # If the current part itself is too long, split it further
        while len(current_chunk.encode('utf-8')) > MAX_TTS_BYTES:
            # Find a safe split point (e.g., at MAX_TTS_BYTES or slightly before to avoid breaking UTF-8 chars)
            temp_segment = current_chunk[:MAX_TTS_BYTES]
            # Ensure we don't split in the middle of a multi-byte character
            while len(temp_segment.encode('utf-8')) > MAX_TTS_BYTES:
                temp_segment = temp_segment[:-1]
            
            if not temp_segment.strip(): # Avoid infinite loop if first char is too big or empty
                break

            audio_content_parts.extend(synthesize_chunk(temp_segment.strip(), tts_client))
            current_chunk = current_chunk[len(temp_segment):]

    if current_chunk.strip():
        audio_content_parts.extend(synthesize_chunk(current_chunk.strip(), tts_client))

    # Concatenate all audio content parts
    combined_audio_content = b"".join(audio_content_parts)
    audio_base64 = base64.b64encode(combined_audio_content).decode("utf-8")
    return audio_base64

def synthesize_chunk(chunk_text: str, tts_client_instance) -> list[bytes]:
    """Helper function to synthesize a single chunk of text."""
    if not chunk_text.strip():
        return []
    try:
        synthesis_input = texttospeech.SynthesisInput(text=chunk_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-CN",
            name="cmn-CN-Chirp3-HD-Fenrir"
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client_instance.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return [response.audio_content]
    except Exception as e:
        print(f"Error during Text-to-Speech synthesis for chunk: '{chunk_text[:50]}...' - {e}")
        traceback.print_exc()
        return []

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