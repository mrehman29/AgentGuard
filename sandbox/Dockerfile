# Sandbox image: Docker provider (python3 -I -) and Cloud Run worker (default CMD).
# Never add a secret: the code run in this container is untrusted.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chmod=0444 worker.py ./

USER 65534:65534
CMD ["python3", "worker.py"]
