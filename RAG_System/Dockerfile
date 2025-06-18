FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Pre-copy requirements.txt for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code (after pip to preserve cache)
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Default command
CMD ["./start.sh"]
