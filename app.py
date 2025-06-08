from flask import Flask, request, jsonify
from flask_cors import CORS
from computer_logic import query_fireworks


app = Flask(__name__)
CORS(app)  # 解鎖前端跨域存取

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        reply = query_fireworks(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
