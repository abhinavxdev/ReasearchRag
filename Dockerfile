FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app:app /app/data
COPY ui ./ui
USER app
EXPOSE 8000 8501
CMD ["uvicorn", "researchrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
