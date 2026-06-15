import unittest
import os
import sys

# Add parent directory to sys.path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import CollegeChatBot

class TestCollegeChatBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the chatbot instance once for all tests."""
        # Suppress TensorFlow logging during tests
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Verify that necessary files exist before trying to load
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        intents_path = os.path.join(base_dir, 'intents.json')
        words_path = os.path.join(base_dir, 'words.pkl')
        classes_path = os.path.join(base_dir, 'classes.pkl')
        model_path = os.path.join(base_dir, 'chatbotmodel.h5')
        
        cls.chatbot = CollegeChatBot(
            intents_path=intents_path,
            words_path=words_path,
            classes_path=classes_path,
            model_path=model_path
        )

    def test_clean_up_sentence(self):
        """Test the sentence tokenization and lemmatization."""
        sentence = "Hello! What is your name?"
        result = self.chatbot.clean_up_sentence(sentence)
        self.assertIsInstance(result, list)
        self.assertIn('hello', result)
        self.assertIn('name', result)

    def test_bag_of_words(self):
        """Test bag of words creation."""
        sentence = "Hello"
        bow = self.chatbot.bag_of_words(sentence)
        self.assertEqual(len(bow), len(self.chatbot.words))
        # Ensure it's a numpy array of integers
        self.assertTrue(all(i in [0, 1] for i in bow))

    def test_predict_class_greetings(self):
        """Test predicting the greetings intent."""
        intents = self.chatbot.predict_class("Hello")
        self.assertTrue(len(intents) > 0)
        self.assertEqual(intents[0]['intent'], 'greetings')

    def test_get_response(self):
        """Test getting a response for a known intent."""
        intents_list = [{'intent': 'greetings', 'probability': '0.9'}]
        response = self.chatbot.get_response(intents_list)
        self.assertIsInstance(response, str)
        # Verify the response is one of the greeting responses
        greeting_responses = [i['responses'] for i in self.chatbot.intents['intents'] if i['tag'] == 'greetings'][0]
        self.assertIn(response, greeting_responses)
        
    def test_get_reply_end_to_end(self):
        """Test the end-to-end reply function."""
        response, intent = self.chatbot.get_reply("Goodbye")
        self.assertIsInstance(response, str)
        self.assertEqual(intent, 'goodbye')

if __name__ == '__main__':
    unittest.main()
