import pickle
import json
from keras.preprocessing.text import Tokenizer

# Load data from intents.json (replace 'path/to/intents.json' with the actual path)
with open('D:\clg proj\Chatbot_Keras-main/intents.json') as file:
    data = json.load(file)

# Extract text data from patterns and responses
texts = []
for intent in data['intents']:
    patterns = intent.get('patterns', [])
    responses = intent.get('responses', [])
    texts.extend(patterns + responses)

# Create a tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Save the tokenizer to a file using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer saved to tokenizer.pickle")
