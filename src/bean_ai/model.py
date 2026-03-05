import json
import os
import pickle

import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences
from keras_preprocessing.text import tokenizer_from_json


class ModelPredictor:
    """
    A reusable predictor that loads the model once for efficient repeated predictions.
    """

    def __init__(self, model_dir: str):
        """
        Initialize the predictor by loading model, tokenizer, config, and id2label.

        Args:
            model_dir: Path to the directory containing model files.
        """
        # Load config
        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
        self.maxlen = config['maxlen']

        # Load id2label mapping
        with open(os.path.join(model_dir, 'id2label.pkl'), 'rb') as f:
            self.id2label = pickle.load(f)

        # Load model
        self.model = load_model(os.path.join(model_dir, 'model.keras'))

        # Load tokenizer
        with open(os.path.join(model_dir, 'tokenizer.json')) as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)

    def predict(self, text: str) -> str:
        """
        Predict the account for a transaction description.

        Args:
            text: Transaction description text.

        Returns:
            Predicted account name.
        """
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=self.maxlen)
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        return self.id2label[predicted_label]
