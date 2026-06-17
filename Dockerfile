FROM python:3.10-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface_hub \
    && python -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Pantelis-K/glaucoma-fundus-unet', f, local_dir='.') for f in ('disc.keras','cup.keras')]"
EXPOSE 8080
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py .
COPY samples/ ./samples/
COPY app.py .
COPY index.html .
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

