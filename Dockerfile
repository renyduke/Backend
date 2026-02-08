# Use python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models/plots if they don't exist
RUN mkdir -p models plots

# Expose port (7860 is common for Hugging Face, 8000 for others)
EXPOSE 7860

# Command to run the application
# Use 0.0.0.0 to accept connections from outside container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
