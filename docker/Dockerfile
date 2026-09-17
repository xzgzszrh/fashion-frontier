# Reproducible host environment for the CPU-only route.
# The board side needs nothing but numpy + onnxruntime; see requirements/board.txt.
FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "fashionfrontier", "results"]
