# Deployment Guide

This document provides instructions for deploying the Multimodal RAG System to various environments.

## Table of Contents

- [Docker](#docker)
- [Hugging Face Spaces](#hugging-face-spaces)
- [Cloud Platforms](#cloud-platforms)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring](#monitoring)
- [Production Checklist](#production-checklist)

---

## Docker

### Building the Image

```bash
docker build -t multimodal-rag .
```

### Running Containers

```bash
# Full application with GPU support
docker run --gpus all -p 8000:8000 -p 7860:7860 multimodal-rag

# API server only
docker run --gpus all -p 8000:8000 multimodal-rag python -m src.api.server

# Web interface only
docker run --gpus all -p 7860:7860 multimodal-rag python -m src.web.app
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Hugging Face Spaces

### Deployment Steps

1. Create a new Space at https://huggingface.co/spaces
2. Select Gradio as the SDK
3. Choose T4 Small for GPU support

### File Upload

```bash
git clone https://github.com/harshkushwaha7x/Multimodal-RAG-System-for-Document-Intelligence.git

cp app_hf.py YOUR_SPACE/app.py
cp requirements.txt YOUR_SPACE/
cp -r src/ YOUR_SPACE/

cd YOUR_SPACE
git add .
git commit -m "Initial deployment"
git push
```

### Environment Variables

Configure in Space settings:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face access token |
| `OLLAMA_HOST` | External Ollama server URL (optional) |

---

## Cloud Platforms

### AWS EC2

**Instance Configuration:**
- AMI: Deep Learning AMI (Ubuntu)
- Instance Type: g4dn.xlarge (T4 GPU)
- Storage: 50 GB minimum

**Setup Commands:**

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

git clone https://github.com/YOUR_REPO/multimodal-rag.git
cd multimodal-rag

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

screen -S rag
python -m src.api.server --host 0.0.0.0
```

**Security Group Rules:**
- Port 8000: API access
- Port 7860: Web interface

### Google Cloud Platform

```bash
gcloud compute instances create rag-server \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB
```

### Serverless Options

**Railway:**
- Connect GitHub repository
- Start command: `python -m src.api.server`
- Note: No GPU support on free tier

**Render:**
- Build command: `pip install -r requirements.txt`
- Start command: `python -m src.api.server`

---

## CI/CD Pipeline

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: your-username/multimodal-rag:latest
```

---

## Monitoring

### Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

### Log Access

```bash
# Docker
docker logs -f container_name

# Systemd
journalctl -u rag-api -f
```

---

## Production Checklist

| Item | Status |
|------|--------|
| Set strong JWT_SECRET_KEY | [ ] |
| Configure HTTPS (nginx/caddy) | [ ] |
| Set appropriate rate limits | [ ] |
| Enable Redis caching | [ ] |
| Configure log rotation | [ ] |
| Set up monitoring (Prometheus/Grafana) | [ ] |
| Configure FAISS index backup | [ ] |
