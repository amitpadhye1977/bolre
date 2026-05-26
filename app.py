from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)

# Enable CORS
CORS(app)

# -----------------------------------
# OPENAI CLIENT
# -----------------------------------

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.route("/")
def home():

    return jsonify({
        "status": "BolRe AI Server Running"
    })

# -----------------------------------
# AI TEXT ENHANCEMENT ROUTE
# -----------------------------------

@app.route("/enhance-text", methods=["POST"])
def enhance_text():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid request"
            }), 400

        text = data.get("text", "").strip()

        source_lang = data.get(
            "source_lang",
            "auto"
        )

        target_lang = data.get(
            "target_lang",
            "same"
        )

        style = data.get(
            "style",
            "casual"
        )

        if not text:

            return jsonify({
                "error": "No text provided"
            }), 400

        # -----------------------------------
        # AI PROMPT
        # -----------------------------------

        system_prompt = f"""
You are an AI assistant for the BolRe voice messaging app.

Your task is to improve speech-to-text messages.

Rules:
- Fix grammar naturally
- Add punctuation
- Improve sentence flow
- Correct slang intelligently
- Preserve original meaning
- Keep tone natural and conversational
- Do NOT over-formalize unless necessary
- Return ONLY the improved text
- No explanations
- No quotes

Source Language: {source_lang}
Target Language: {target_lang}
Style: {style}
"""

        # -----------------------------------
        # OPENAI REQUEST
        # -----------------------------------

        response = client.chat.completions.create(

            model="gpt-5.5-mini",

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": text
                }
            ],

            temperature=0.4,
            max_tokens=200
        )

        enhanced_text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        return jsonify({

            "success": True,

            "original_text": text,

            "enhanced_text": enhanced_text
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

# -----------------------------------
# HEALTH CHECK
# -----------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })

# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
