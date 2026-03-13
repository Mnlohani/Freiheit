FROM python:3.11-slim

# Intsall uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory to the `app` directory
WORKDIR /app

# Copy dependency files from root
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the project into the image
COPY . /app

# THE KEY: Set PYTHONPATH to the current directory 
# This makes 'src' importable from anywhere inside /app
ENV PYTHONPATH=/app

EXPOSE 8501

# Start the backend as a module to match your local 'python -m' success
CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]