from flask import Flask, request, jsonify
import numpy as np
import keras
import pickle
from datetime import datetime
import random
import json
from flask_cors import CORS 

app = Flask(__name__)
CORS(app, supports_credentials=True)

with open("D:\clg proj\Chatbot_Keras-main\intents.json") as file:
    data = json.load(file)

def get_date():
    return datetime.now().strftime("%d-%m-%Y")

def get_time():
    return datetime.now().strftime("%I:%M:%S %p")

def get_response(intent_tag):
    intent = next((item for item in data["intents"] if item["tag"] == intent_tag), None)
    if intent:
        response = random.choice(intent["responses"])
        if "{{date}}" in response:
            response = response.replace("{{date}}", get_date())
        if "{{time}}" in response:
            response = response.replace("{{time}}", get_time())
        return response
    else:
        return "I'm sorry, I don't understand that."

@app.route('/conversation', methods=['POST', 'OPTIONS'])
def process_chat():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight request successful'})
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Allow-Headers')
        return response

    try:
        data = request.get_json()
        user_text = data['userText']

        # Load your model, tokenizer, and label encoder here
        model = keras.models.load_model('D:\clg proj\Chatbot_Keras-main\chat_model')
        with open('D:\clg proj\Chatbot_Keras-main/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('D:\clg proj\Chatbot_Keras-main/label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # Process user text and get a response
        max_len = 20
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_text]),
                                            truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        response = get_response(tag[0])

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
