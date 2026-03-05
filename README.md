# bean-ai

## Installation

The current implementation requires Python 3.11.

As executable:

```bash
pipx install . --python python3.11
```

For development:

```bash
pyenv virtualenv 3.11.0 bean-ai
pyenv activate bean-ai
pip install -e .
```

## HTTP Server

Start the prediction server:

```bash
bean-ai serve --port 8080 --dir ./models/
```

Make predictions:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ACME Corp Payment", "Grocery Store"]}'
# Response: {"accounts": ["Expenses:Services:Software", "Expenses:Food"]}
```
