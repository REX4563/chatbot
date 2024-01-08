import json 
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder

import colorama
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

from datetime import datetime

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


def chat():
    # load trained model
    model = keras.models.load_model('D:\clg proj\Chatbot_Keras-main\chat_model')

    # load tokenizer object
    with open('D:\clg proj\Chatbot_Keras-main/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('D:\clg proj\Chatbot_Keras-main/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        response = get_response(tag[0])
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, response)

    print(Fore.YELLOW + "Chat ended. Have a great day!" + Style.RESET_ALL)

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
