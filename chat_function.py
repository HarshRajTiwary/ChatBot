import numpy as np
import nltk
import pickle
import random
import json
from tensorflow.keras.models import load_model # type: ignore
from nltk.stem import WordNetLemmatizer

# Load the model
model = load_model('/home/harsh/Desktop/chatbot/model/chatbot_model.h5')

ext = ["Sad to see you go :(", "Talk to you later", "Goodbye!", "Come back soon"]

# Load tokenizer and classes
words = pickle.load(open('/home/harsh/Desktop/chatbot/model/words.pkl', 'rb'))
classes = pickle.load(open('/home/harsh/Desktop/chatbot/model/classes.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words array
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

# Function to predict the class
def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Function to get the response
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to get chatbot response
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, data)
    return res

# Load intents file
with open('/home/harsh/Desktop/chatbot/dataset/intents.json') as file:
    data = json.load(file)

while True:
    message = input("Enter Your Prompt: ")
    response = chatbot_response(message)
    print(response)
    # print("*********************************")
    if response in ext:
        break
