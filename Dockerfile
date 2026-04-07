FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install CLI entrypoint (openenv validate)
RUN pip install --no-cache-dir -e .

# Expose port for HF Space HTTP server
EXPOSE 7860

# Default: start the FastAPI server (HF Space deployment)
# Override CMD to run inference or validate instead
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
