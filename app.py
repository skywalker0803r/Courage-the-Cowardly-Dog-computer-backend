from flask import Flask, request, jsonify
from flask_cors import CORS
from computer_logic import query
import os
import base64
import json
from google.cloud import texttospeech
from google.oauth2 import service_account

# APP設定
app = Flask(__name__)
CORS(app)

# GOOGLE_APPLICATION_CREDENTIALS_JSON設定以使用texttospeech服務
credentials = service_account.Credentials.from_service_account_info(json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")))
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# 文字轉語音函數
def text_to_speech(text):
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
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_id = data.get("user_id")
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        reply = query(user_message, user_id)
        audio_base64 = text_to_speech(reply)
        return jsonify({"reply": reply, "audio": audio_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
