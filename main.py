from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import nltk
import numpy as np
import random
import json
import pickle

app = Flask(__name__)
CORS(app)

# Load model and data
model = load_model('/home/harsh/Desktop/chatbot/chatbot/chatbot_model.h5')
words = pickle.load(open('/home/harsh/Desktop/chatbot/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('/home/harsh/Desktop/chatbot/chatbot/classes.pkl', 'rb'))

with open('/home/harsh/Desktop/chatbot/chatbot/intents.json') as file:
    intents = json.load(file)

lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm sorry, I don't understand that. Can you rephrase?"
    return result

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    return jsonify({"response": res})

if __name__ == '__main__':
    app.run(debug=True)
