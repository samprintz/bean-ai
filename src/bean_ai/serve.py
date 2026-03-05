import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from .model import ModelPredictor


class PredictionHandler(BaseHTTPRequestHandler):
    """HTTP request handler for prediction requests."""

    predictor: ModelPredictor = None

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        if self.path == '/predict':
            content_length = self.headers.get('Content-Length')
            if content_length is None:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Missing Content-Length"}')
                return

            body = self.rfile.read(int(content_length))

            try:
                data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            if 'texts' not in data or not isinstance(data['texts'], list):
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Missing or invalid texts array"}')
                return

            predictions = [self.predictor.predict(text) for text in data['texts']]

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'accounts': predictions}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def _method_not_allowed(self):
        self.send_response(405)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Method Not Allowed')

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def serve(host: str, port: int, model_dir: str):
    """
    Start the HTTP prediction server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        model_dir: Path to the model directory.
    """
    print(f"Loading model from {model_dir}...")
    PredictionHandler.predictor = ModelPredictor(model_dir)

    server = HTTPServer((host, port), PredictionHandler)
    print(f"Server running on http://{host}:{port}")
    print("Endpoints:")
    print(f"  POST /predict - Predict account for transaction")
    print(f"  GET  /health  - Health check")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
