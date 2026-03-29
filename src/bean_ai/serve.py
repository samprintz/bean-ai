from bottle import Bottle, request, response, run

from .model import ModelPredictor

app = Bottle()
predictor: ModelPredictor = None


@app.get('/health')
def health():
    return 'ok'


@app.post('/predict')
def predict():
    response.content_type = 'application/json'

    data = request.json
    if data is None:
        response.status = 400
        return {'error': 'Invalid JSON'}

    if 'texts' not in data or not isinstance(data['texts'], list):
        response.status = 400
        return {'error': 'Missing or invalid texts array'}

    accounts = [predictor.predict(text) for text in data['texts']]
    return {'accounts': accounts}


def serve(host: str, port: int, model_dir: str):
    """
    Start the HTTP prediction server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        model_dir: Path to the model directory.
    """
    global predictor
    print(f"Loading model from {model_dir}...")
    try:
        predictor = ModelPredictor(model_dir)
    except FileNotFoundError as e:
        print(f"Error: Model directory '{model_dir}' does not contain a valid model.")
        raise SystemExit(1)

    print(f"Server running on http://{host}:{port}")
    print("Endpoints:")
    print("  POST /predict - Predict account for transaction")
    print("  GET  /health  - Health check")

    run(app, host=host, port=port, quiet=True)
