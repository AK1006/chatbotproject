import json
import pickle
import random
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

def download_nltk_data():
    """Ensure required NLTK packages are available."""
    packages = ['punkt', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package)

def train_chatbot_model():
    """Trains the chatbot model and saves it as 'chatbotmodel.h5'."""
    download_nltk_data()

    lemmatizer = WordNetLemmatizer()

    try:
        with open('intents.json', 'r') as file:
            intents = json.load(file)
    except FileNotFoundError:
        print("Error: 'intents.json' not found. Please ensure it exists in the root directory.")
        return

    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', ',', '.']

    # Process intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and sort
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save words and classes
    with open('words.pkl', 'wb') as file:
        pickle.dump(words, file)
    with open('classes.pkl', 'wb') as file:
        pickle.dump(classes, file)

    # Prepare training data
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1 if word in word_patterns else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Build model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model (using learning_rate instead of lr for newer TF versions)
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train and save model
    print("Training model...")
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    
    model.save('chatbotmodel.h5')
    print("Model successfully trained and saved as 'chatbotmodel.h5'")

if __name__ == "__main__":
    train_chatbot_model()