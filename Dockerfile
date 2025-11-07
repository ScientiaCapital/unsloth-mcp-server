# Multi-stage Dockerfile for Unsloth MCP Server

# Stage 1: Build
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src ./src

# Build the application
RUN npm run build

# Stage 2: Python environment
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Node.js and Python
RUN apt-get update && apt-get install -y \
    curl \
    python3.11 \
    python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Unsloth and dependencies
RUN pip3 install --no-cache-dir \
    unsloth \
    torch \
    transformers \
    datasets \
    trl

WORKDIR /app

# Copy built application from builder
COPY --from=builder /app/build ./build
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Copy configuration files
COPY config.example.json ./config.json

# Create directories
RUN mkdir -p logs .cache

# Environment variables
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose no ports (MCP uses stdio)

# Run the server
CMD ["node", "build/index.js"]
