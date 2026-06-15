# Usage Guide

## Starting the Chatbot

Once you have trained the model (see [Setup](setup.md)), you can start the interactive console chatbot by running:

```bash
python main.py
```

You should see an output similar to:

```text
|============= Welcome to College Enquiry Chatbot System! =============|
|=============== Ask your any query about our college ================|
|================= Type 'quit' or 'exit' to terminate. =================|
| You: 
```

## Interacting

You can ask various questions related to the college, such as:
- *"What courses do you offer?"*
- *"Where is the college located?"*
- *"What are the hostel fees?"*

## Modifying Intents

If you want to add new questions and answers, open `intents.json` and add a new block:

```json
{
  "tag": "new_topic",
  "patterns": ["question 1", "question 2"],
  "responses": ["answer 1", "answer 2"]
}
```

**Important:** After modifying `intents.json`, you **must** retrain the model.

```bash
python trainingData.py
```

## Running Tests

To run the automated test suite and ensure all components function properly:

```bash
python -m unittest discover tests/
```
