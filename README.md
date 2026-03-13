# bean-ai

`bean-ai` is a Python application
to classify Beancount transactions with AI.

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

## Usage

### Preprocess data

```bash
bean-ai preprocess ledger.beancount
```

### Train model

```bash
bean-ai train data.csv
```

### Predict account

```bash
bean-ai predict "Grocery Store"
```

### Serve model (HTTP Server)

```bash
bean-ai serve
```

Make predictions:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Pharmacy", "Grocery Store"]}'
# Response: {"accounts": ["Expenses:Health", "Expenses:Food"]}
```
