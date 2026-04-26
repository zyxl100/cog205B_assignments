# Bayes Factor Test Suite

This is a small homework project for testing a simple BayesFactor class.

## Files

- `bayes_factor.py`: implementation of the BayesFactor class
- `tests/test_bayes_factor.py`: unittest test suite
- `Dockerfile`: simple Docker setup

## Install locally

```bash
pip install scipy
```

## Run tests

From inside the `bayes_factor/` folder:

```bash
python -m unittest tests/test_bayes_factor.py
```

or:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Docker

Build:

```bash
docker build -t bayes-factor-homework .
```

Run tests:

```bash
docker run --rm bayes-factor-homework
```
