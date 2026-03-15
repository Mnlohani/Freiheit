FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Fix: prevent cross-filesystem hardlink I/O errors
ENV UV_LINK_MODE=copy
# Fix: use poll watcher — more reliable inside Docker
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll

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

EXPOSE 8501

# Start the backend as a module to match your local 'python -m' success
CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]