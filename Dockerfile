FROM python:3.14-slim

WORKDIR /app

# install uv (your package manager)
RUN pip install uv

# copy dependency files first, install deps (this layer caches)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# copy the app code, config, and the model registry
COPY src/ ./src/
COPY configs/ ./configs/
COPY mlflow.db ./
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
