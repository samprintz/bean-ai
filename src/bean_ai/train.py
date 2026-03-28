import io
import json
import os
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical, pad_sequences
from keras_preprocessing.text import Tokenizer


def train(input_path: str, model_dir: str) -> None:
    """
    Train an LSTM model on preprocessed transaction data.

    Reads a CSV with 'text' and 'label' columns, trains an LSTM classifier,
    and saves the model, tokenizer, and label mappings.
    """
    if os.path.exists(model_dir):
        answer = input(f"{model_dir} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return

    # Read data
    df = pd.read_csv(input_path, sep="\t")
    df['label_name'] = df['label']

    # Create label mappings
    labels = df["label_name"].unique().tolist()
    print(f"Number of labels: {len(labels)}")

    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    df['label'] = df['label_name'].map(lambda x: label2id[x])
    num_classes = len(labels)
    df['label2'] = to_categorical(df["label"], num_classes=num_classes).tolist()

    # Tokenize text
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['text'])

    df['text_vectors'] = tokenizer.texts_to_sequences(df['text'])

    maxlen = max(len(seq) for seq in df['text_vectors'])
    print(f"Max sequence length: {maxlen}")

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Pad sequences
    df['text_vectors'] = pad_sequences(df['text_vectors'], padding='post', maxlen=maxlen).tolist()

    # Split dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True)

    X_train = np.array(dataset['train']['text_vectors'])
    y_train = np.array(dataset['train']['label2'])
    X_test = np.array(dataset['test']['text_vectors'])
    y_test = np.array(dataset['test']['label2'])

    # Build model
    embedding_dim = 128

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(layers.LSTM(maxlen, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        batch_size=32
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print(f"Training Accuracy: {accuracy:.4f}")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print(f"Testing Accuracy: {accuracy:.4f}")

    # Save model and artifacts
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(os.path.join(model_dir, 'model.keras'))

    tokenizer_json = tokenizer.to_json()
    with io.open(os.path.join(model_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    with open(os.path.join(model_dir, 'id2label.pkl'), 'wb') as f:
        pickle.dump(id2label, f)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump({'maxlen': maxlen}, f)

    print(f"Model saved to {model_dir}")
