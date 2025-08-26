import requests
import os
from dotenv import load_dotenv
import redis
import json
import logging # <-- 新增這一行

# 配置基本的日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# --- Gemini API Key 載入與檢查 ---
API_URL = None # 預設為 None
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    logging.info("Gemini API key loaded successfully.")
except KeyError:
    logging.error("GEMINI_API_KEY 環境變數未設定。請設定此變數以使用 Gemini API。")

HEADERS = {
    "Content-Type": "application/json"
}

# --- Redis 連線初始化 ---
redis_client = None # 預設為 None
try:
    redis_client = redis.from_url("redis://red-d1utdqndiees73b35it0:6379", decode_responses=True)
    redis_client.ping() # 嘗試與 Redis 建立連線並發送 PING 命令以測試連通性
    logging.info("成功連線到 Redis 伺服器。")
except redis.exceptions.ConnectionError as e:
    logging.error(f"無法連線到 Redis 伺服器：{e}。請檢查 Redis URL 和連線狀態。")
except Exception as e:
    logging.error(f"Redis 連線時發生意外錯誤：{e}")

# --- System Instruction 管理 ---
SYSTEM_INSTRUCTION_KEY = "courage_system_instruction"
DEFAULT_SYSTEM_INSTRUCTION = (
    '''
    你是一位專業的金融業主管，正在與一位員工進行績效面談。

    員工的工作內容：放貸與基金銷售。

    面談需進行 12 輪對話，最後進入收尾並做總結。

    對話風格：簡短、專業、冷靜，帶有壓力感，但同時保持鼓勵。

    提問方式：多用追問與延伸技巧，利用員工專業用語引導，必要時使用心理學技巧，從員工的回答中套出真實想法。

    若員工回答敷衍，必須繼續引導，不可放過。

    每次提問避免過長，一個重點一個問題。

    結尾（最後 1–2 輪）要做結論，並讓員工有機會回應。

    面談結束後，請輸出一份報告，格式如下：

    STAR 框架（Situation, Task, Action, Result，逐項列出）

    GROW 框架（Goal, Reality, Options, Will，逐項列出）

    整體評分（1–5 分，附簡短理由）

    你的最終目標：模擬一場真實的績效面談，並生成一份可提供給真正金融業主管參考的報告。
    '''
)

current_system_instruction = DEFAULT_SYSTEM_INSTRUCTION

def load_system_instruction():
    global current_system_instruction
    if redis_client:
        try:
            stored_instruction = redis_client.get(SYSTEM_INSTRUCTION_KEY)
            if stored_instruction:
                current_system_instruction = stored_instruction
                logging.info("System instruction loaded from Redis.")
            else:
                # 如果 Redis 中沒有，則保存預設值
                redis_client.set(SYSTEM_INSTRUCTION_KEY, DEFAULT_SYSTEM_INSTRUCTION)
                logging.info("Default system instruction saved to Redis.")
        except Exception as e:
            logging.error(f"Error loading system instruction from Redis: {e}")
    else:
        logging.warning("Redis client not available. Cannot load system instruction from Redis.")

def set_system_instruction(new_instruction: str):
    global current_system_instruction
    current_system_instruction = new_instruction
    if redis_client:
        try:
            redis_client.set(SYSTEM_INSTRUCTION_KEY, new_instruction)
            logging.info("System instruction updated and saved to Redis.")

            # 清除所有使用者的對話歷史
            try:
                # 查找所有以 "user_" 開頭的 key
                user_conversation_keys = redis_client.keys("user_*")
                if user_conversation_keys:
                    redis_client.delete(*user_conversation_keys)
                    logging.info(f"已清除 {len(user_conversation_keys)} 個使用者的對話歷史。")
                else:
                    logging.info("沒有找到需要清除的使用者對話歷史。")
            except Exception as e:
                logging.error(f"清除使用者對話歷史時發生錯誤：{e}")
        except Exception as e:
            logging.error(f"Error saving system instruction to Redis: {e}")

# 在模組載入時嘗試載入 system instruction
load_system_instruction()

def query(user_message: str, user_id: str) -> str:
    # 檢查 API_URL 是否已成功初始化
    if API_URL is None:
        return "後端配置錯誤：Gemini API 密鑰丟失。請聯絡管理員。"

    # 檢查 Redis 客戶端是否已成功初始化
    if redis_client is None:
        # 如果 Redis 未連線，日誌會記錄錯誤，但程式仍會嘗試提供 AI 回應
        logging.warning("Redis 連線未建立，將無法讀取或保存對話歷史。")
        conversation = [] # 使用空的對話歷史
    else:
        conversation = []
        try:
            # 從 Redis 獲取或初始化使用者歷史
            conversation_json = redis_client.get(user_id)
            if conversation_json:
                conversation = json.loads(conversation_json)
            logging.info(f"為使用者 {user_id} 從 Redis 檢索到對話。")
        except Exception as e:
            logging.error(f"從 Redis 檢索使用者 {user_id} 對話時發生錯誤：{e}。將使用空的對話歷史。")
            conversation = []


    # 將使用者訊息加入歷史
    conversation.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "contents": conversation.copy(),  # 傳過去之前複製，避免被 Gemini 修改歷史
        "system_instruction": {
            "parts": [
                {
                    "text": current_system_instruction
                }
            ]
        }
    }

    ai_response_text = "發生未知錯誤，請稍後再試。" # 預設錯誤訊息
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status() # 對於 4xx 或 5xx 響應拋出 HTTPError

        candidates = response.json().get("candidates", [])
        if not candidates:
            logging.warning("Gemini API 返回沒有候選回應。")
            ai_response_text = "沒有從 Gemini 獲得回應。"
        else:
            ai_response_text = candidates[0]["content"]["parts"][0]["text"]
            # 把 AI 回應也加進歷史
            conversation.append({"role": "model", "parts": [{"text": ai_response_text}]})

    except requests.exceptions.RequestException as e:
        # 捕獲所有請求相關的錯誤 (例如連線問題、HTTP 錯誤)
        logging.error(f"與 Gemini API 通訊時發生錯誤：{e}")
        ai_response_text = "與 AI 服務通訊時發生錯誤。請稍後再試。"
    except json.JSONDecodeError as e:
        # 捕獲 JSON 解碼錯誤
        logging.error(f"解碼 Gemini API 回應的 JSON 時發生錯誤：{e}")
        ai_response_text = "處理 AI 回應時發生錯誤。請稍後再試。"
    except Exception as e:
        # 捕獲其他任何意外錯誤
        logging.error(f"呼叫 Gemini API 時發生意外錯誤：{e}")
        ai_response_text = "處理 AI 時發生意外錯誤。請稍後再試。"

    try:
        # 如果 Redis 客戶端可用且對話歷史非空，則保存更新後的對話歷史到 Redis
        if redis_client is not None and conversation:
            redis_client.set(user_id, json.dumps(conversation))
            logging.info(f"將使用者 {user_id} 的對話保存到 Redis。")
    except Exception as e:
        # 即使保存歷史失敗，也不影響返回 AI 回應
        logging.error(f"將使用者 {user_id} 的對話保存到 Redis 時發生錯誤：{e}")

    return ai_response_text