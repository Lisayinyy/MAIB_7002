# Reproducible environment for the notebook and the helper package.
#
# Build:  docker build -t pricing-ml .
# Run:    docker run --rm -p 8888:8888 -v "$PWD/data:/app/data" pricing-ml
#         then open the http://127.0.0.1:8888/lab?token=... link printed in the terminal.
# The raw data (data/raw/train.parquet, data/raw/eval.parquet) is not in the image; mount it.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first so this layer is cached while the code changes.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY src ./src
COPY results ./results
COPY docs ./docs
COPY README.md discount_sales_prediction.ipynb ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
