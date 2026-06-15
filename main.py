import random
import json
import pickle
import numpy as np
import sys
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

class CollegeChatBot:
    def __init__(self, intents_path='intents.json', words_path='words.pkl', classes_path='classes.pkl', model_path='chatbotmodel.h5'):
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            with open(intents_path, 'r') as file:
                self.intents = json.load(file)
            
            with open(words_path, 'rb') as file:
                self.words = pickle.load(file)
                
            with open(classes_path, 'rb') as file:
                self.classes = pickle.load(file)
                
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading required resources: {e}")
            print("Please ensure you have run 'python trainingData.py' to generate the model and pickle files.")
            sys.exit(1)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list):
        if not intents_list:
            return "I am sorry, I didn't quite understand that."
            
        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']
        
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
                
        return "I am sorry, I couldn't find a proper response."

    def get_reply(self, message):
        ints = self.predict_class(message)
        res = self.get_response(ints)
        return res, ints[0]['intent'] if ints else None

def start_chat():
    bot = CollegeChatBot()
    
    print("|============= Welcome to College Enquiry Chatbot System! =============|")
    print("|=============== Ask your any query about our college ================|")
    print("|================= Type 'quit' or 'exit' to terminate. =================|")
    
    while True:
        try:
            message = input("| You: ").strip()
            if message.lower() in ["quit", "exit"]:
                print("| Bot: Goodbye!")
                print("|===================== The Program End here! =====================|")
                break
                
            if not message:
                continue

            response, intent = bot.get_reply(message)
            print("| Bot:", response)
            
            # Additional exit condition if intent matched is 'goodbye'
            if intent == "goodbye":
                print("|===================== The Program End here! =====================|")
                break
                
        except KeyboardInterrupt:
            print("\n|===================== The Program End here! =====================|")
            break
        except Exception as e:
            print(f"\n| Bot: An error occurred: {e}")

if __name__ == "__main__":
    start_chat()