# Stage 1: Build
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app .
COPY . .
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
