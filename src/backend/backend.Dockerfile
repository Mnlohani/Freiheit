# Install uv
FROM python:3.11-slim

# Intsall uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Set the working directory to the `app` directory
WORKDIR /app

# Copy dependency files from root
COPY pyproject.toml uv.lock ./

# Copy the project into the image
COPY . /app

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
# THE KEY: Set PYTHONPATH to the current directory 
# This makes 'src' importable from anywhere inside /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
