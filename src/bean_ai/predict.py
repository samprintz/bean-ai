import json
import os
import pickle

import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences
from keras_preprocessing.text import tokenizer_from_json


DEFAULT_MODEL_DIR = "./models/"


def predict(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """
    Predict the account for a transaction description.

    Loads a trained model and returns the predicted account name.
    """
    # Load config
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    maxlen = config['maxlen']

    # Load id2label mapping
    with open(os.path.join(model_dir, 'id2label.pkl'), 'rb') as f:
        id2label = pickle.load(f)

    # Load model
    model = load_model(os.path.join(model_dir, 'model.keras'))

    # Load tokenizer
    with open(os.path.join(model_dir, 'tokenizer.json')) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # Preprocess input
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=maxlen)

    # Predict
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_label = np.argmax(prediction, axis=1)[0]

    return id2label[predicted_label]
