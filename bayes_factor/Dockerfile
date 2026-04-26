FROM python:3.11-slim

WORKDIR /app

RUN pip install scipy

COPY bayes_factor.py /app/bayes_factor.py
COPY tests /app/tests
COPY README.md /app/README.md

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
