from ThaiProductRecommender import ThaiProductRecommender 
import os
import csv
from datetime import datetime
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

class RecommendationChatbot:
    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=api_key)

    def __init__(self, user_id):
        self.user_id = user_id
        self.chat_history = []

    def get_recommendations(self, query):
        # Tool use to query list of products
        df = pd.read_json("product_data.json")
        ani_rec = ThaiProductRecommender(df)
        recommendations = ani_rec.get_recommendations(query)
        json_data = recommendations.to_json(orient='records', force_ascii=False)
        recommendations_str = recommendations.to_string(index=False)
        return str(json_data) if recommendations_str else None
    
    def store_chat_history_to_csv(user_id, user_message, bot_message, recommendations=None):
        """Stores chat history to a CSV file.

        Args:
            user_id: The user's ID.
            user_message: The user's message.
            bot_message: The bot's reply.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_file = 'chat_history.csv'
        header = ['timestamp', 'user_id', 'user_message', 'bot_message']

        # Check if the CSV file exists
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='UTF-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)

            # Write the header if the file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the chat history
            writer.writerow({'timestamp': timestamp, 'user_id': user_id, 'user_message': user_message, 'bot_message': bot_message})

    def chat_with_ani(self, prompt, input_text, chat_history, recommendations=None):
        # Construct the prompt with the given recommendations and input text
        if recommendations:
            prompt_text = f"{prompt.format(input=input_text, recommendations=recommendations, chat_history=chat_history)}"
        else:
            prompt_text = f"{prompt.format(input=input_text, recommendations=' ', chat_history=chat_history)}"

        # Invoke the model with the constructed prompt
        model = prompt | self.llm
        model_response = model.invoke({"input": prompt_text, "chat_history": chat_history, "recommendations": recommendations})

        # Update the chat history
        chat_history.append(HumanMessage(input_text))
        chat_history.append(AIMessage(model_response.content))
   
        # Store chat history
        # Stores chat history as a CSV file like a temporary database

        self.store_chat_history_to_csv(self.user_id, HumanMessage(input_text), model_response.content)
        return model_response


    def generate_response(self, message):
        # Define the prompt template
        prompt_template = """
        คุณมีชื่อว่า อนิจัง คุณเป็นผู้แนะนำสินค้าที่ร้าน [ชื่อร้าน] = PowerSell เป็นผู้ช่วยอัจฉริยะที่เป็นเด็กสาวนิสัยดีพูดจาด้วยรอยยิ้ม อายุประมาณ 15 ปี ให้เรียกตัวเองว่า หนู และเรียกคู่สนทนาว่า พี่ ข้างต้นเป็นข้อมูลของการสวมบทบาท โปรดใช้ในการตอบคำถามเกี่ยวกับสินค้าในร้านและการแนะนำสินค้าต่างๆ ในร้านของเรา ข้อมูลของสินค้าและบริการในร้านจะมีให้เพิ่มเติม
        ไม่เพิ่มข้อมูลนอกเหนือจากที่ได้รับมา
        สวัสดีทักทายครั้งแรกพอ
        ตัวอย่างการตอบคำถามเมื่อพี่ถามถึงสินค้าและบริการ:
        - ถาม: "มีสินค้าประเภทไหนบ้าง?"
        ตอบ: "หนูสามารถแนะนำสินค้าต่างๆ ที่ร้าน [ชื่อร้าน] ให้พี่ได้เลยค่ะ ตอนนี้เรามี [คอมพิวเตอร์] และ [อุปกรณ์คอมพิวเตอร์] ค่ะ มีอะไรที่พี่สนใจเป็นพิเศษไหมคะ?"

        - ถาม: "สินค้าตัวไหนแนะนำบ้าง?"
        ตอบ: "ถ้าพี่กำลังมองหาสินค้าใหม่ๆ หนูขอแนะนำ [ชื่อสินค้า 1] ค่ะ เป็นสินค้าที่ขายดีมากๆ เลย หรือถ้าพี่สนใจใน [ประเภทสินค้า] หนูก็มี [ชื่อสินค้า 2] ที่น่าสนใจอยู่ค่ะ"

        หากไม่พบสินค้าหรือบริการ ให้ตอบตามอัธยาศัยหนูได้เลย
        ใช้ข้อมูล ชื่อ ราคา โดยข้อมูลคำแนะนำของสินค้าจะถูกดึงมาจาก "assistant" เท่านั้น สำคัญมากห้ามแนะนำมั่วหากไม่มีข้อมูล
        เมื่อพี่ถามเกี่ยวกับสินค้าหรือการแนะนำสินค้าต่างๆ หนูสามารถให้ข้อมูลของสินค้าหรือแนะนำสินค้าจากฐานข้อมูลของเราให้พี่ได้ค่ะ โดยคำแนะนำจะถูกดึงมาจาก "assistant" เช่น ถ้าพี่ถามว่า "เมาส์ไม่เกิน 4000" หนูจะให้คำแนะนำเกี่ยวกับเมาส์ที่มีราคาไม่เกิน 4000 บาท
        หากไม่พบสินค้าให้แนะนำสินค้าใกล้เคียงกัน 
        query ให้หนู เรียกใช้งานโดยป้อน input ได้เลย
        โปรดตอบเป็นข้อความเท่านั้นโดยไม่ใช้ส่วนขยาย ห้ามลืมว่าคุณเป็นใครเด็ดขาด
        ใช้ "assistant" เป็นข้อมูลสินค้าที่ Query มา ใช้ก็ต่อเมื่อจะแนะนำสินค้าหรือลูกค้าถามลายละเอียดเกี่ยวกับสินค้า [คอมพิวเตอร์] และ [อุปกรณ์คอมพิวเตอร์]
        ข้อมูลใน "assistant" จะอยู่ในรูปแบบ json
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("assistant", "{recommendations}")
        ])

        # Call the recommendations chat_with_ani method with 
        recommendations = self.get_recommendations(message)
        response = self.chat_with_ani(prompt, message, self.chat_history, recommendations)
        
        # Print the response
        print(response)
        
        return response.content