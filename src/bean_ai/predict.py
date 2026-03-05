from .model import ModelPredictor


DEFAULT_MODEL_DIR = "./models/"


def predict(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """
    Predict the account for a transaction description.

    Loads a trained model and returns the predicted account name.
    """
    predictor = ModelPredictor(model_dir)
    return predictor.predict(text)
