FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Fix: prevent cross-filesystem hardlink I/O errors
ENV UV_LINK_MODE=copy

# Set the working directory to the `app` directory
WORKDIR /app

# Copy dependency files from root
COPY pyproject.toml uv.lock ./

# Install dependencies then clean cache to reduce image size
RUN uv sync --frozen --no-dev && uv cache clean

# Copy the project into the image
COPY . /app

# Set PYTHONPATH to the current directory
# To make 'src' importable from anywhere inside /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]