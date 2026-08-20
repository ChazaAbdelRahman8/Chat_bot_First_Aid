FROM python:3.12-slim

WORKDIR /app
COPY mcp-server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mcp-server/src /app/src
COPY mcp-server/data /app/data

EXPOSE 8002
CMD ["python", "/app/src/server.py"]
