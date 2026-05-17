FROM python:3.11-slim

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface

# Set the working directory
WORKDIR $HOME/app

# Copy the application code into the container
COPY --chown=user . $HOME/app

# Install dependencies (utilizing the CPU-only PyTorch config we added)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the AI model during the Docker build to ensure instant 1-second cold starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Start the FastAPI server on port 7860 (Hugging Face default)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
