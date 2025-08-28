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

    MAX_TTS_BYTES = 800  # Set a slightly lower limit than 900 for safety

    audio_content_parts = []
    
    # Helper to split a long string into chunks respecting MAX_TTS_BYTES
    def _split_long_text_into_chunks(long_text: str, max_bytes: int):
        current_chunk_bytes = 0
        current_chunk_chars = []
        
        for char in long_text:
            char_bytes = len(char.encode('utf-8'))
            if current_chunk_bytes + char_bytes > max_bytes:
                yield "".join(current_chunk_chars)
                current_chunk_chars = [char]
                current_chunk_bytes = char_bytes
            else:
                current_chunk_chars.append(char)
                current_chunk_bytes += char_bytes
        if current_chunk_chars:
            yield "".join(current_chunk_chars)

    # Split text by common Chinese and English sentence-ending punctuation, keeping delimiters
    sentence_delimiters_pattern = r'([。？！；.?!])'
    parts = re.split(sentence_delimiters_pattern, text)

    current_sentence_buffer = ""

    for part in parts:
        if not part.strip():
            continue

        # If the part itself is too long, split it further
        if len(part.encode('utf-8')) > MAX_TTS_BYTES:
            if current_sentence_buffer.strip(): # Synthesize anything in buffer before handling long part
                audio_content_parts.extend(synthesize_chunk(current_sentence_buffer.strip(), tts_client))
                current_sentence_buffer = ""
            
            for sub_chunk in _split_long_text_into_chunks(part, MAX_TTS_BYTES):
                if sub_chunk.strip():
                    audio_content_parts.extend(synthesize_chunk(sub_chunk.strip(), tts_client))
        else:
            # Check if adding the part makes the current buffer too long
            if len((current_sentence_buffer + part).encode('utf-8')) > MAX_TTS_BYTES:
                if current_sentence_buffer.strip():
                    audio_content_parts.extend(synthesize_chunk(current_sentence_buffer.strip(), tts_client))
                current_sentence_buffer = part
            else:
                current_sentence_buffer += part

            # If the part is a delimiter, or if the buffer is now a complete sentence, synthesize
            if re.fullmatch(sentence_delimiters_pattern, part) and current_sentence_buffer.strip():
                audio_content_parts.extend(synthesize_chunk(current_sentence_buffer.strip(), tts_client))
                current_sentence_buffer = ""

    # Synthesize any remaining buffer
    if current_sentence_buffer.strip():
        audio_content_parts.extend(synthesize_chunk(current_sentence_buffer.strip(), tts_client))

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