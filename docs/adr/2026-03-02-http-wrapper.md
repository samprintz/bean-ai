# Servable as HTTP Wrapper

`bean-ai` should be used by `bean-review` or other Python modules.
However, `bean-ai` is based on a legacy Keras and TensorFlow codebase
and requires Python 3.11.
Consequently, it cannot be directly imported as Python module.

Additionally, for prediction,
`bean-ai` requires a warm-up time for loading the model.
It could be called by using the `subprocess` module,
but it would be inefficient to start a new process for each prediction.

Calling it only once with `subprocess` and passing
the narration strings each as a command-line argument
is rather error-prone as the strings may contain special characters and white space
that need to be escaped properly.

## Decision

Create a simple HTTP wrapper around `bean-ai`.
This wrapper will start a local HTTP server that listens for incoming requests.

## Rationale

- Only one warm-up
- Isolated Python 3.11 environment
- Deployable as service (e.g. on VPS)
