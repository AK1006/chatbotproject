# AI Chatbot

A simple, robust Artificial Intelligence Chatbot designed for handling college-related queries. Built using Python, NLTK, and TensorFlow/Keras.

## Overview

This project is a Neural Network-based Intent Matching Chatbot. It processes user input using Natural Language Processing (NLP) techniques like tokenization and lemmatization, generates a Bag-of-Words representation, and passes it through a deep learning Sequential model to predict the intent.

## Features
- **Intent Recognition:** Understands user inputs and maps them to predefined intents.
- **Context-Free Responses:** Provides randomized but contextual responses based on the matched intent.
- **Customizable:** Easily extend the vocabulary and intents by editing a single `intents.json` file.
- **Lightweight:** Uses an efficient fully connected neural network suitable for quick inference.

## Quick Start

Please refer to the [Setup Guide](docs/setup.md) for full installation instructions.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python trainingData.py

# 3. Start the chatbot
python main.py
```

## Documentation

- [Architecture](docs/architecture.md)
- [Setup & Installation](docs/setup.md)
- [Usage](docs/usage.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
