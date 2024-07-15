from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
import os
from dotenv import load_dotenv
import json
import requests
import re
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import chatbot

# Assign environment
chatbot_instance = chatbot.RecommendationChatbot(user_id=None)

app = Flask(__name__)

# Initialize API KEY
load_dotenv()
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))


# Get message and reply with line format
@app.route("/", methods=['POST'])
def webhook():
    if request.method == 'POST':
        payload = request.json
        app.logger.info(f"Received payload: {json.dumps(payload, indent=2)}")

        try:
            event = payload['events'][0]
            user_id = event['source']['userId']
        
            message_type = event['message']['type']
        
            if message_type == 'text':
                message = event['message']['text']

                chatbot_instance.user_id = user_id
                Reply_message = chatbot_instance.generate_response(message)
                
                PushMessage(user_id, Reply_message)
                app.logger.info(f"Message pushed to user {user_id}: {Reply_message}")

            else:
                app.logger.info(f"Unsupported message type: {message_type}")

            return '', 200

        except Exception as e:
            app.logger.error(f"Error in webhook: {e}")
            abort(400)
    else:
        abort(400)

def PushMessage(user_id, TextMessage):
    LINE_API = 'https://api.line.me/v2/bot/message/push'
    Authorization = f'Bearer {os.getenv("LINE_CHANNEL_ACCESS_TOKEN")}'
    app.logger.info(f"Authorization: {Authorization}")
    
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': Authorization
    }

    img_url = extract_image_url(TextMessage)
    if img_url:
        data = {
            "to": user_id,
            "messages": [{
                "type": "image",
                "originalContentUrl": img_url[0],
                "previewImageUrl": img_url[0],
            },
            {
                "type": "text",
                "text": TextMessage,
            }]
        }
    else:
        data = {
            "to": user_id,
            "messages": [
                {
                    "type": "text",
                    "text": TextMessage,
                }
            ]
        }

    data = json.dumps(data)
    
    try:
        response = requests.post(LINE_API, headers=headers, data=data)
        response.raise_for_status()
        app.logger.info(f"Message pushed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to push message: {e}")

        # Fallback: Send only the text message
        fallback_data = {
            "to": user_id,
            "messages": [
                {
                    "type": "text",
                    "text": TextMessage,
                }
            ]
        }
        fallback_data = json.dumps(fallback_data)

        try:
            fallback_response = requests.post(LINE_API, headers=headers, data=fallback_data)
            fallback_response.raise_for_status()
            app.logger.info(f"Fallback message pushed: {fallback_response.status_code}")
        except requests.exceptions.RequestException as fallback_e:
            app.logger.error(f"Failed to push fallback message: {fallback_e}")

# Line dont resive the image then this use to find image url
def extract_image_url(input_string):
    url_pattern = re.compile(r'https://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    matches = url_pattern.findall(input_string)
    return matches if matches else None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
