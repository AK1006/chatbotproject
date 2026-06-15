# Setup & Installation Guide

This guide will help you set up the AI Chatbot project on your local machine.

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)
- Git (optional, for cloning)

## Step 1: Clone or Extract

If you have the source code, open your terminal and navigate to the project root directory.

```bash
cd AI_Chatbot
```

## Step 2: Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

Install the required Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
```

This will install:
- `tensorflow`
- `nltk`
- `numpy`

## Step 4: Prepare Environment Variables (Optional)

Copy the `.env.example` file to `.env` if you wish to override default configurations.

```bash
cp .env.example .env
```

## Step 5: Train the Model

Before running the chatbot for the first time, you must generate the model and vocabulary by training it on `intents.json`.

```bash
python trainingData.py
```

This will automatically download necessary NLTK corpora, process the intents, train the neural network, and output three files:
- `chatbotmodel.h5`
- `words.pkl`
- `classes.pkl`

Your setup is now complete! Head over to the [Usage Guide](usage.md) to start the chatbot.
