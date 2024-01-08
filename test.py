# chatbot.py

import json
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime

class Chatbot:
    def __init__(self, model_path='D:\clg proj\Chatbot_Keras-main\chat_model', tokenizer_path='D:\clg proj\Chatbot_Keras-main/tokenizer.pickle',
                 label_encoder_path='D:\clg proj\Chatbot_Keras-main/label_encoder.pickle'):
        self.model = keras.models.load_model(model_path)

        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(label_encoder_path, 'rb') as enc:
            self.lbl_encoder = pickle.load(enc)

        self.max_len = 20

        with open("D:\clg proj\Chatbot_Keras-main/intents.json") as file:
            self.data = json.load(file)

    def get_date(self):
        return datetime.now().strftime("%d-%m-%Y")

    def get_time(self):
        return datetime.now().strftime("%I:%M:%S %p")

    def get_response(self, intent_tag):
        intent = next((item for item in self.data["intents"] if item["tag"] == intent_tag), None)
        if intent:
            response = np.random.choice(intent["responses"])
            if "{{date}}" in response:
                response = response.replace("{{date}}", self.get_date())
            if "{{time}}" in response:
                response = response.replace("{{time}}", self.get_time())
            return response
        else:
            return "I'm sorry, I don't understand that."

    def process_input(self, user_input):
        result = self.model.predict(keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([user_input]),
                                             truncating='post', maxlen=self.max_len))
        tag = self.lbl_encoder.inverse_transform([np.argmax(result)])
        return tag[0]

    def get_chat_response(self, user_input):
        intent_tag = self.process_input(user_input)
        return self.get_response(intent_tag)
