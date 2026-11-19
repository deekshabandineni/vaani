# Build from the repo root:  docker build -f services/ml-api/Dockerfile .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY services/ml-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# inference-only install of the shared ML package (no PyTorch)
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/vaani_ml ml/vaani_ml
RUN pip install --no-cache-dir --no-deps ./ml
COPY services/ml-api/app app
ENV VAANI_MODEL_PATH=/models/vaani_fp32.onnx VAANI_CALIBRATION=/models/calibration.json
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
