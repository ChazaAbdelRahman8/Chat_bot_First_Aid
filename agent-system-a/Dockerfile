# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

WORKDIR /app

COPY agent-system-a/requirements.txt /app/requirements.txt
COPY agent-system-a/docker/download_paddle_wheel.py /tmp/download_paddle_wheel.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 5 \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/wheel-cache \
    test ! -f /wheel-cache/paddlepaddle-3.2.0-cp312-linux-x86_64.whl || \
        mv /wheel-cache/paddlepaddle-3.2.0-cp312-linux-x86_64.whl \
           /wheel-cache/paddlepaddle-3.2.0-cp312-cp312-manylinux1_x86_64.whl && \
    python /tmp/download_paddle_wheel.py \
        --version 3.2.0 \
        --output /wheel-cache/paddlepaddle-3.2.0-cp312-cp312-manylinux1_x86_64.whl && \
    pip install --no-deps \
        /wheel-cache/paddlepaddle-3.2.0-cp312-cp312-manylinux1_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 5 \
    -r /app/requirements.txt

COPY agent-system-a/src /app/src

RUN mkdir -p /app/data

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
