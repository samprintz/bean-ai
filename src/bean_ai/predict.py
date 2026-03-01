DEFAULT_MODEL_DIR = "./models/"

def predict(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    print(f"Predicting account for: {text}")
    return "Expenses:Unknown"
